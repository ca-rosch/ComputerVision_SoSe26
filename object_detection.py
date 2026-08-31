# Use of AI tools: Claude was used throughout the project to assist with debugging, code optimization, the creation of data visualizations, some code generation and documentation for functions.


import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from monai.data.box_utils import box_iou, clip_boxes_to_image
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=str, default="local_run")
args = parser.parse_args()

JOB_ID = args.job_id
print(JOB_ID)
os.makedirs(f"results/{JOB_ID}", exist_ok=True)



CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis", "No finding",
]

def compute_class_counts(label_sets, num_classes=15):
    """label_sets: iterable of sets/lists of class ids, one per image.
    Returns the number of images each class appears in."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for labels in label_sets:
        for c in labels:
            counts[int(c)] += 1
    return counts

def compute_sample_weights(label_sets, num_classes=15, dampen=0.5):
    """Per-image weight for WeightedRandomSampler: an image is weighted by
    its rarest present class, so batches oversample rare findings instead
    of being dominated by "No finding". `dampen` in (0, 1] softens
    1/frequency so the rarest class isn't oversampled too aggressively
    (dampen=1 -> full inverse-frequency, dampen=0 -> uniform)."""
    counts = np.clip(compute_class_counts(label_sets, num_classes), 1, None)
    inv_freq = 1.0 / counts
    weights = [max(inv_freq[int(c)] for c in labels) ** dampen for labels in label_sets]
    return torch.tensor(weights, dtype=torch.double)

# --- config ---------------------------------------------------------------
NO_FINDING_ID = 14  # class_id for "No finding" in train.csv / CLASS_NAMES
DETECTION_CLASS_NAMES = CLASS_NAMES[:-1]  # drop "No finding": absence isn't a box
NUM_DET_CLASSES = len(DETECTION_CLASS_NAMES) + 1  # +1 for torchvision's background class 0
IMG_SIZE = 1024  # matches the pre-resized PNGs; see coordinate-frame note above
FUSION_IOU_THRESH = 0.4  # multi-rad boxes overlapping at/above this get merged

BATCH_SIZE = 4  # detection at 1024x1024 is far more memory-hungry per image than
                # the 224x224 classifier, hence much smaller than amia_mode_v2's 32
NUM_EPOCHS = 100
MAX_PATIENCE = 15
LR = 1e-4
WEIGHT_DECAY = 5e-4
FREEZE_UNTIL = "layer4"
WARMUP_ITERS = 1000  # linear LR warmup for epoch 0 only (capped at len(train_loader)-1).
SEED = 42
NUM_WORKERS = 4

data_path = "/storage/mi/paet02/amia_data"
train_path = os.path.join(data_path, "train", "train")
test_path = os.path.join(data_path, "test", "test")


SPLIT_DIR = os.path.join("results", "outputs")
SPLIT_PATH = os.path.join(SPLIT_DIR, "baseline_split.csv")
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_TRIALS = 300
SPLIT_RANDOM_STATE_TEST = 42
SPLIT_RANDOM_STATE_VAL = 43

torch.manual_seed(SEED)


def build_label_matrix(image_id_list, labels_by_image, num_classes):
    matrix = np.zeros((len(image_id_list), num_classes), dtype=np.float32)
    for row_index, image_id in enumerate(image_id_list):
        for class_id in labels_by_image[image_id]:
            matrix[row_index, class_id] = 1.0
    return matrix


def build_split_matrix(image_id_list, labels_by_image, num_classes):
    """Matrix used only to balance the split: the disease columns plus one
    extra No-finding column (absence of every disease label)."""
    disease_matrix = build_label_matrix(image_id_list, labels_by_image, num_classes)
    no_finding = (disease_matrix.sum(axis=1) == 0).astype(np.float32).reshape(-1, 1)
    return np.hstack([disease_matrix, no_finding])


def create_balanced_multilabel_split(image_id_list, labels_by_image, num_classes,
                                      val_ratio=0.2, random_state=42, trials=300):
    """Select a reproducible random split with similar class (and No-finding)
    prevalences on both sides -- a dependency-free approximation to
    iterative multi-label stratification."""
    ids = np.asarray(image_id_list)
    split_targets = build_split_matrix(ids, labels_by_image, num_classes)

    n_samples = len(ids)
    n_val = int(round(n_samples * val_ratio))

    rng = np.random.default_rng(random_state)

    best_score = np.inf
    best_train_indices = None
    best_val_indices = None

    total_positives = split_targets.sum(axis=0)

    for _ in range(trials):
        val_indices = rng.choice(n_samples, size=n_val, replace=False)

        val_mask = np.zeros(n_samples, dtype=bool)
        val_mask[val_indices] = True
        train_indices = np.flatnonzero(~val_mask)

        train_targets = split_targets[train_indices]
        val_targets = split_targets[val_indices]

        train_prevalence = train_targets.mean(axis=0)
        val_prevalence = val_targets.mean(axis=0)
        prevalence_difference = np.abs(train_prevalence - val_prevalence).mean()

        # Penalize splits in which a class is completely absent from one side
        missing_train = ((train_targets.sum(axis=0) == 0) & (total_positives > 0)).sum()
        missing_val = ((val_targets.sum(axis=0) == 0) & (total_positives > 0)).sum()

        score = prevalence_difference + missing_train + missing_val
        if score < best_score:
            best_score = score
            best_train_indices = train_indices
            best_val_indices = val_indices

    train_ids = ids[best_train_indices].tolist()
    val_ids = ids[best_val_indices].tolist()
    return train_ids, val_ids


def build_split_label_sets(train_csv):
    """Per-image set of disease class ids (0-13) present according to *any*
    radiologist. Kept
    separate from build_detection_targets's box-fusion below: the split only
    needs label presence, not geometry, and must match what the notebook
    used to pick the same split."""
    image_labels = {}
    for image_id, group in train_csv.groupby("image_id", sort=False):
        image_labels[image_id] = {
            int(class_id) for class_id in group["class_id"].astype(int)
            if class_id != NO_FINDING_ID
        }
    return image_labels


def load_or_create_split(image_ids, image_labels, num_classes, split_path=SPLIT_PATH,
                          val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, trials=SPLIT_TRIALS):
    """Loads `split_path` if present
    (so the classifier and detector share the exact same train/val/test
    images); otherwise recreates it with the identical procedure and random
    seeds."""
    if os.path.exists(split_path):
        print("Loading existing baseline split:", split_path)
        split_table = pd.read_csv(split_path)
        train_ids = split_table.loc[split_table["split"] == "train", "image_id"].tolist()
        val_ids = split_table.loc[split_table["split"] == "validation", "image_id"].tolist()
        test_ids = split_table.loc[split_table["split"] == "test", "image_id"].tolist()
        return train_ids, val_ids, test_ids

    print("Creating new baseline split")

    development_ids, test_ids = create_balanced_multilabel_split(
        image_ids, image_labels, num_classes,
        val_ratio=test_ratio, random_state=SPLIT_RANDOM_STATE_TEST, trials=trials)
    train_ids, val_ids = create_balanced_multilabel_split(
        development_ids, image_labels, num_classes,
        val_ratio=val_ratio / (1 - test_ratio), random_state=SPLIT_RANDOM_STATE_VAL, trials=trials)

    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    split_table = pd.concat([
        pd.DataFrame({"image_id": train_ids, "split": "train"}),
        pd.DataFrame({"image_id": val_ids, "split": "validation"}),
        pd.DataFrame({"image_id": test_ids, "split": "test"}),
    ], ignore_index=True)
    split_table.to_csv(split_path, index=False)
    print("Split saved to:", split_path)

    return train_ids, val_ids, test_ids


# --- multi-rad box fusion + coordinate rescaling ---------------------------
def fuse_rad_boxes(boxes, iou_thresh=FUSION_IOU_THRESH):
    """boxes: float32 ndarray [N,4], all one class, from up to 3 radiologists
    for one image. Greedy IoU clustering (via monai's box_iou) merges boxes
    different raters drew for the same finding into their mean; a box no one
    else agrees with is kept on its own."""
    n = len(boxes)
    if n <= 1:
        return boxes.copy()
    iou = np.asarray(box_iou(boxes, boxes))
    used = np.zeros(n, dtype=bool)
    fused = []
    for i in range(n):
        if used[i]:
            continue
        cluster = np.where((iou[i] >= iou_thresh) & (~used))[0]
        used[cluster] = True
        fused.append(boxes[cluster].mean(axis=0))
    return np.stack(fused, axis=0).astype(np.float32)


def build_detection_targets(csv_df, size_df, image_ids, target_size=IMG_SIZE,
                             iou_thresh=FUSION_IOU_THRESH):
    """Returns (targets, label_sets):
      targets: {image_id -> (boxes [N,4] float32 xyxy in the 1024x1024 PNG
                frame, labels [N] int64 in 0..13)}
      label_sets: per-image_id set of class ids present (post-fusion; {14}
                  for a boxless image), aligned to `image_ids`, for reuse
                  with compute_sample_weights.
    """
    fg = csv_df[csv_df["class_id"] != NO_FINDING_ID].merge(
        size_df[["image_id", "dim0", "dim1"]], on="image_id", how="left")
    # dim0 = height, dim1 = width; PNGs are target_size x target_size regardless
    # of original aspect ratio, so x/y need independent scale factors
    fg["x_min"] = fg["x_min"] * (target_size / fg["dim1"])
    fg["x_max"] = fg["x_max"] * (target_size / fg["dim1"])
    fg["y_min"] = fg["y_min"] * (target_size / fg["dim0"])
    fg["y_max"] = fg["y_max"] * (target_size / fg["dim0"])

    grouped = fg.groupby("image_id")
    targets = {}
    label_sets = []
    for image_id in image_ids:
        if image_id not in grouped.groups:
            targets[image_id] = (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64))
            label_sets.append({NO_FINDING_ID})
            continue

        per_image = grouped.get_group(image_id)
        boxes_out, labels_out, present = [], [], set()
        for class_id, sub in per_image.groupby("class_id"):
            boxes = sub[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=np.float32)
            fused = fuse_rad_boxes(boxes, iou_thresh=iou_thresh)
            fused, keep = clip_boxes_to_image(fused, (target_size, target_size), remove_empty=True)
            fused = np.asarray(fused)[np.asarray(keep)]
            if len(fused) == 0:
                continue
            boxes_out.append(fused)
            labels_out.append(np.full(len(fused), int(class_id), dtype=np.int64))
            present.add(int(class_id))

        if boxes_out:
            targets[image_id] = (np.concatenate(boxes_out, axis=0), np.concatenate(labels_out, axis=0))
            label_sets.append(present)
        else:
            targets[image_id] = (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64))
            label_sets.append({NO_FINDING_ID})

    return targets, label_sets


# --- dataset ----------------------------------------------------------------
class AMIADataset(Dataset):
    """One image + its fused bounding boxes, in torchvision detection's
    expected format (target dict of boxes/labels/image_id). Labels are
    stored 1-indexed (1..14) since torchvision reserves label 0 for
    background."""

    def __init__(self, image_ids, targets_by_id, img_dir, transform=None, img_size=IMG_SIZE):
        self.image_ids = image_ids
        self.targets_by_id = targets_by_id
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = Image.open(os.path.join(self.img_dir, f"{image_id}.png")).convert("RGB")
        assert image.size == (self.img_size, self.img_size), (
            f"{image_id}: expected a {self.img_size}x{self.img_size} PNG, got {image.size}. "
            "Boxes were rescaled assuming that fixed size (see build_detection_targets) -- "
            "an image at a different resolution means misaligned boxes, not just a resize."
        )

        boxes_arr, labels_arr = self.targets_by_id[image_id]
        boxes = torch.as_tensor(boxes_arr, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels_arr, dtype=torch.int64).reshape(-1) + 1  # 0-13 -> 1-14

        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY",
                                               canvas_size=(self.img_size, self.img_size)),
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        if self.transform:
            image, target = self.transform(image, target)
        else:
            image = v2.functional.to_dtype(v2.functional.to_image(image), torch.float32, scale=True)

        target["boxes"] = target["boxes"].as_subclass(torch.Tensor)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# no v2.Normalize here: fasterrcnn_resnet50_fpn_v2's own GeneralizedRCNNTransform
# already normalizes with ImageNet mean/std internally, so normalizing here too
# would double-normalize. No horizontal flip either: not clearly label-preserving for laterality-relevant
# chest X-ray findings.
train_transform = v2.Compose([
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.RandomApply([v2.RandomZoomOut(fill=0, side_range=(1.0, 1.3))], p=0.2),
    v2.RandomApply([v2.RandomIoUCrop()], p=0.5),
    v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.SanitizeBoundingBoxes(),
])

val_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])


class AMIATestDataset(Dataset):
    """The competition's held-out test.csv images -- unlabelled, so this only
    returns (image, image_id) for inference, not a target dict."""

    def __init__(self, image_ids, img_dir, transform=None, img_size=IMG_SIZE):
        self.image_ids = image_ids
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = Image.open(os.path.join(self.img_dir, f"{image_id}.png")).convert("RGB")
        assert image.size == (self.img_size, self.img_size), (
            f"{image_id}: expected a {self.img_size}x{self.img_size} PNG, got {image.size}."
        )
        image = self.transform(image) if self.transform else \
            v2.functional.to_dtype(v2.functional.to_image(image), torch.float32, scale=True)
        return image, image_id


def collate_test(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def predict_test_submission(model, device, out_path="submission.csv", score_thresh=0.05,
                             batch_size=BATCH_SIZE):
    """Runs inference over every image in test.csv and writes the competition's
    submission format: one row per unique test image_id, header
    `image_id,PredictionString`, PredictionString = "class_id score xmin ymin
    xmax ymax ..." repeated per detection, or "14 1.0 0 0 1 1" ("No finding",
    one-pixel box) if nothing clears score_thresh. Boxes are rescaled from the
    model's 1024x1024 frame back to each image's original DICOM pixel size via
    img_size.csv -- the exact inverse of the scaling build_detection_targets
    applies to training boxes -- since that's the frame test.csv/the grader
    expects, not the resized PNG's.
    """
    test_csv = pd.read_csv(os.path.join(data_path, "test.csv"))
    size_lookup = pd.read_csv(os.path.join(data_path, "img_size.csv")).set_index("image_id")
    test_ids = test_csv["image_id"].unique().tolist()

    dataset = AMIATestDataset(test_ids, test_path, transform=val_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=NUM_WORKERS, collate_fn=collate_test)

    model.eval()
    rows = []
    for images, image_ids in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for image_id, out in zip(image_ids, outputs):
            dim0, dim1 = size_lookup.loc[image_id, ["dim0", "dim1"]]  # height, width
            scale_x, scale_y = dim1 / IMG_SIZE, dim0 / IMG_SIZE

            boxes = out["boxes"].cpu().numpy()
            labels = out["labels"].cpu().numpy() - 1  # 1-14 -> 0-13, matches train.csv's class_id
            scores = out["scores"].cpu().numpy()
            keep = scores >= score_thresh

            parts = []
            for (x_min, y_min, x_max, y_max), label, score in zip(boxes[keep], labels[keep], scores[keep]):
                parts += [
                    str(int(label)), f"{score:.4f}",
                    str(int(round(x_min * scale_x))), str(int(round(y_min * scale_y))),
                    str(int(round(x_max * scale_x))), str(int(round(y_max * scale_y))),
                ]
            rows.append({
                "image_id": image_id,
                "PredictionString": " ".join(parts) if parts else "14 1.0 0 0 1 1",
            })

    submission = pd.DataFrame(rows)
    submission.to_csv(out_path, index=False)
    print(f"Wrote {len(submission)} predictions to {out_path}")
    return submission


# --- model --------------------------------------------------------------
def build_model(num_classes, device, freeze_until=FREEZE_UNTIL, box_positive_fraction=0.5):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, min_size=IMG_SIZE, max_size=IMG_SIZE,
                                        box_positive_fraction=box_positive_fraction)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # partial-freeze the backbone: keep the generic early conv blocks fixed and
    # only fine-tune the later stages plus the FPN/RPN/ROI heads.
    backbone_stages = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
    freeze_idx = backbone_stages.index(freeze_until)
    for stage_name in backbone_stages[:freeze_idx]:
        for param in getattr(model.backbone.body, stage_name).parameters():
            param.requires_grad = False

    return model.to(device)


# --- train / eval loops ---------------------------------------------------
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """Linear LR warmup over the first `warmup_iters` steps (torchvision's
    detection reference recipe). Only meant to wrap epoch 0."""
    def f(x):
        if x >= warmup_iters:
            return 1.0
        alpha = x / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(model, dataloader, optimizer, device, epoch, warmup_scheduler=None):
    model.train()
    running_loss = 0.0
    loss_components = {}
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        running_loss += loss.item()
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0.0) + v.item()

    epoch_loss = running_loss / len(dataloader)
    breakdown = " | ".join(f"{k}: {v / len(dataloader):.4f}" for k, v in loss_components.items())
    print(f"Epoch {epoch} | Train loss: {epoch_loss:.4f} ({breakdown})")
    return epoch_loss


@torch.no_grad()
def evaluate(model, dataloader, device, epoch, coco_metric):
    # torchvision detection models only return the loss dict in train() mode
    # and only return predictions in eval() mode -- so getting a comparable
    # validation loss needs a separate no-grad forward pass in train() mode
    # before switching to eval() for the actual predictions/mAP.
    model.train()
    running_loss = 0.0
    loss_components = {}
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        running_loss += sum(loss_dict.values()).item()
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0.0) + v.item()
    epoch_loss = running_loss / len(dataloader)

    model.eval()
    pred_boxes, pred_labels, pred_scores = [], [], []
    gt_boxes, gt_labels = [], []
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            pred_boxes.append(out["boxes"].cpu().numpy())
            pred_labels.append(out["labels"].cpu().numpy() - 1)  # back to 0-13
            pred_scores.append(out["scores"].cpu().numpy())
            gt_boxes.append(tgt["boxes"].cpu().numpy())
            gt_labels.append(tgt["labels"].cpu().numpy() - 1)

    results = matching_batch(
        iou_fn=box_iou,
        iou_thresholds=coco_metric.iou_thresholds,
        pred_boxes=pred_boxes, pred_classes=pred_labels, pred_scores=pred_scores,
        gt_boxes=gt_boxes, gt_classes=gt_labels,
    )
    metric_dict, _ = coco_metric(results)
    # mAP@0.4 is the competition's actual metric (PASCAL VOC 2010 mAP, IoU > 0.4);
    # mAP@0.5 is reported alongside only as a more commonly-cited reference point.
    map_40 = metric_dict["AP_IoU_0.40_MaxDet_100"]
    map_50 = metric_dict["AP_IoU_0.50_MaxDet_100"]
    print(f"Epoch {epoch} | Val loss: {epoch_loss:.4f} | mAP@0.4: {map_40:.4f} | mAP@0.5: {map_50:.4f} "
          f"| mAP@[0.1:0.5]: {metric_dict['mAP_IoU_0.10_0.50_0.05_MaxDet_100']:.4f}")
    breakdown = " | ".join(f"{k}: {v / len(dataloader):.4f}" for k, v in loss_components.items())
    print(f"  (val loss breakdown: {breakdown})")
    return epoch_loss, map_40, map_50, metric_dict


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_csv = pd.read_csv(os.path.join(data_path, "train.csv"))
    size_csv = pd.read_csv(os.path.join(data_path, "img_size.csv"))
    image_ids = train_csv["image_id"].unique().tolist()

    split_label_sets = build_split_label_sets(train_csv)
    train_ids, val_ids, test_ids = load_or_create_split(
        image_ids, split_label_sets, num_classes=len(DETECTION_CLASS_NAMES))

    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)
    assert set(train_ids + val_ids + test_ids) == set(image_ids)
    print(f"Train: {len(train_ids)} | Validation: {len(val_ids)} | Test: {len(test_ids)} "
          f"| Total: {len(train_ids) + len(val_ids) + len(test_ids)}")

    # --- fused, rescaled detection targets for every image -----------------
    targets_by_id, label_sets = build_detection_targets(train_csv, size_csv, image_ids)

    train_dataset = AMIADataset(train_ids, targets_by_id, train_path, transform=train_transform)
    val_dataset = AMIADataset(val_ids, targets_by_id, train_path, transform=val_transform)
    # Held-out labeled test split from train.csv (distinct from the competition's
    # unlabeled test.csv)
    test_dataset = AMIADataset(test_ids, targets_by_id, train_path, transform=val_transform)

    # --- rare-finding oversampling within the training split ---
    train_label_sets = [
        set(targets_by_id[i][1].tolist()) or {NO_FINDING_ID} for i in train_ids
    ]
    sample_weights = compute_sample_weights(train_label_sets, num_classes=len(CLASS_NAMES), dampen=0.3)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                               num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    model = build_model(NUM_DET_CLASSES, device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    coco_metric = COCOMetric(classes=DETECTION_CLASS_NAMES, iou_list=[0.4, 0.5], max_detection=[100])

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/best_model_detection_improved_{JOB_ID}.pth"
    history = {"train_loss": [], "val_loss": [], "val_map40": [], "val_map50": []}
    best_val_map = 0.0
    patience = 0
    start_time = time.time()
    
    print("*** Starting Training ***")
    for epoch in range(NUM_EPOCHS):
        # LR warmup only for epoch 0 -- see warmup_lr_scheduler docstring.
        warmup_scheduler = None
        if epoch == 0:
            warmup_iters = min(WARMUP_ITERS, len(train_loader) - 1)
            warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor=1.0 / 1000)

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, warmup_scheduler)
        val_loss, val_map40, val_map50, val_metrics = evaluate(model, val_loader, device, epoch, coco_metric)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_map40"].append(val_map40)
        history["val_map50"].append(val_map50)

        scheduler.step(val_map40)

        if val_map40 > best_val_map:
            best_val_map = val_map40
            torch.save(model.state_dict(), checkpoint_path)
            patience = 0
        else:
            patience += 1
            if patience > MAX_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Patience: {patience}/{MAX_PATIENCE} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Epoch {epoch} took {time.time() - start_time:.2f}s (cumulative)")

    print(f"Training time: {time.time() - start_time:.2f} seconds")
    print(f"Best validation mAP@0.4 (competition metric): {best_val_map:.4f}")
    
    
    # --- reload best checkpoint ---
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    print("\n=== Validation split: per-class AP@0.4 ===")
    _, _, _, val_final_metrics = evaluate(model, val_loader, device, "best", coco_metric)
    for name in DETECTION_CLASS_NAMES:
        print(f"  {name:22s} AP@0.4={val_final_metrics[f'{name}_AP_IoU_0.40_MaxDet_100']:.3f}")
    
    print("\n=== Held-out test split (untouched until now): per-class AP@0.4 ===")
    _, test_map40, test_map50, test_final_metrics = evaluate(model, test_loader, device, "test", coco_metric)
    for name in DETECTION_CLASS_NAMES:
        print(f"  {name:22s} AP@0.4={test_final_metrics[f'{name}_AP_IoU_0.40_MaxDet_100']:.3f}")
    print(f"Held-out test mAP@0.4: {test_map40:.4f} | mAP@0.5: {test_map50:.4f}")

    os.makedirs(SPLIT_DIR, exist_ok=True)
    test_report = pd.DataFrame({
        "class_name": DETECTION_CLASS_NAMES,
        "AP@0.4": [test_final_metrics[f"{name}_AP_IoU_0.40_MaxDet_100"] for name in DETECTION_CLASS_NAMES],
        "AP@0.5": [test_final_metrics[f"{name}_AP_IoU_0.50_MaxDet_100"] for name in DETECTION_CLASS_NAMES],
    })
    test_report_path = f"results/{JOB_ID}/detection_improved_test_per_class.csv"
    test_report.to_csv(test_report_path, index=False)
    print("Saved held-out test per-class AP:", test_report_path)

    # --- plots ---
    plt.figure()
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Detection Train and Validation Loss")
    plt.savefig(f"results/{JOB_ID}/det_loss_plot_improved.png")

    plt.figure()
    plt.plot(history["val_map40"], label="Validation mAP@0.4")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.legend()
    plt.title("Validation mAP")
    plt.savefig(f"results/{JOB_ID}/det_map_plot_improved.png")

    # --- competition submission file (Kaggle's unlabeled test.csv) ---
    predict_test_submission(model, device, out_path=f"results/{JOB_ID}/submission.csv")

    from detection_viz import run_detection_viz
    run_detection_viz(
        model, test_ids, targets_by_id, train_path, device,
        DETECTION_CLASS_NAMES, out_dir=f"results/{JOB_ID}", score_thresh=0.30,
    )