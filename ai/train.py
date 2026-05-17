"""
train.py — U-Net Training Script
LGG MRI Segmentation (Kaggle Dataset)

Dataset structure expected:
    kaggle_3m/
        TCGA_XX_XXXX/
            image.tif
            image_mask.tif
            ...

Colab usage:
    !python train.py --data_root /content/data/kaggle_3m --epochs 50
"""

import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

DEFAULT_CONFIG = {
    "encoder":          "resnet34",
    "encoder_weights":  "imagenet",
    "in_channels":      3,
    "num_classes":      1,
    "image_size":       256,
    "batch_size":       8,
    "epochs":           50,
    "lr":               1e-4,
    "val_split":        0.2,
    "model_save_path":  "models/unet_best.pth",
    "checkpoint_dir":   "models/checkpoints",
    "log_dir":          "logs",
    "seed":             42,
}


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────

class MRISegmentationDataset(Dataset):
    """
    Loads matched (image, mask) .tif pairs from the LGG Kaggle dataset.
    Images : TCGA_XX_XXXX/filename.tif
    Masks  : TCGA_XX_XXXX/filename_mask.tif
    """

    def __init__(self, image_paths: list, mask_paths: list, transform=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load as RGB (some .tif files are RGBA or grayscale)
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))

        # Load mask as grayscale and binarize
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)               # (H, W, 1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"].permute(2, 0, 1)     # (1, H, W)

        return image, mask


# ─────────────────────────────────────────────
#  Augmentations
# ─────────────────────────────────────────────

def get_transforms(image_size: int, mode: str = "train") -> A.Compose:
    if mode == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ElasticTransform(p=0.2),
            A.GridDistortion(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    return smp.Unet(
        encoder_name=config["encoder"],
        encoder_weights=config["encoder_weights"],
        in_channels=config["in_channels"],
        classes=config["num_classes"],
        activation=None,                # Raw logits; sigmoid applied in loss
    )


# ─────────────────────────────────────────────
#  Loss
# ─────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """BCE + Dice — standard choice for binary medical segmentation."""

    def __init__(self):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        return self.bce(logits, targets) + self.dice(logits, targets)


# ─────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────

def dice_score(logits: torch.Tensor, targets: torch.Tensor,
               threshold: float = 0.5) -> float:
    preds        = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum()
    return ((2.0 * intersection + 1e-6) /
            (preds.sum() + targets.sum() + 1e-6)).item()


def iou_score(logits: torch.Tensor, targets: torch.Tensor,
              threshold: float = 0.5) -> float:
    preds        = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum()
    union        = preds.sum() + targets.sum() - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).item()


# ─────────────────────────────────────────────
#  Train / Validation Loop
# ─────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, mode="train"):
    is_train = mode == "train"
    model.train() if is_train else model.eval()

    total_loss = total_dice = total_iou = 0.0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, masks in tqdm(loader, desc=f"  {mode.capitalize()}", leave=False):
            images = images.to(device)
            masks  = masks.to(device)
            logits = model(images)
            loss   = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_dice += dice_score(logits, masks)
            total_iou  += iou_score(logits, masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def collect_paths(data_root: str) -> tuple:
    """
    Walk all patient subfolders inside kaggle_3m.
    Match every .tif image to its _mask.tif counterpart.
    """
    image_paths, mask_paths = [], []

    for root, _, files in os.walk(data_root):
        for fname in sorted(files):
            if not fname.endswith(".tif"):
                continue
            if fname.endswith("_mask.tif"):
                continue                                    # Skip mask files here

            name      = os.path.splitext(fname)[0]
            mask_name = f"{name}_mask.tif"
            mask_path = os.path.join(root, mask_name)

            if os.path.exists(mask_path):
                image_paths.append(os.path.join(root, fname))
                mask_paths.append(mask_path)
            else:
                print(f"[WARN] No mask for {fname}, skipping.")

    assert len(image_paths) > 0, (
        f"No image-mask pairs found in: {data_root}\n"
        "Make sure you're pointing to the kaggle_3m folder."
    )
    print(f"[INFO] Found {len(image_paths)} image-mask pairs.")
    return image_paths, mask_paths


def save_checkpoint(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def plot_history(history: dict, log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key in zip(axes, ["loss", "dice", "iou"]):
        ax.plot(history[f"train_{key}"], label="Train")
        ax.plot(history[f"val_{key}"],   label="Val")
        ax.set_title(key.upper())
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    out = os.path.join(log_dir, "training_history.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"[INFO] Training plot saved → {out}")


def save_config(config: dict, log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def train(config: dict):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}")

    # ── Data ──────────────────────────────────
    image_paths, mask_paths = collect_paths(config["data_root"])

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths,
        test_size=config["val_split"],
        random_state=config["seed"],
    )

    train_ds = MRISegmentationDataset(
        train_imgs, train_masks, get_transforms(config["image_size"], "train")
    )
    val_ds = MRISegmentationDataset(
        val_imgs, val_masks, get_transforms(config["image_size"], "val")
    )

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
    )

    print(f"[INFO] Train : {len(train_ds)} | Val : {len(val_ds)}")

    # ── Model / Loss / Optimizer ───────────────
    model     = build_model(config).to(device)
    criterion = CombinedLoss()
    optimizer = Adam(model.parameters(), lr=config["lr"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # ── Training Loop ─────────────────────────
    best_val_loss = float("inf")
    history = {k: [] for k in
               ["train_loss", "val_loss",
                "train_dice", "val_dice",
                "train_iou",  "val_iou"]}

    for epoch in range(1, config["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")

        train_loss, train_dice, train_iou = run_epoch(
            model, train_loader, criterion, optimizer, device, "train"
        )
        val_loss, val_dice, val_iou = run_epoch(
            model, val_loader, criterion, optimizer, device, "val"
        )

        scheduler.step(val_loss)

        for k, v in zip(history.keys(),
                        [train_loss, val_loss,
                         train_dice, val_dice,
                         train_iou,  val_iou]):
            history[k].append(v)

        print(
            f"  Train → Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f}\n"
            f"  Val   → Loss: {val_loss:.4f}   | Dice: {val_dice:.4f}   | IoU: {val_iou:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, config["model_save_path"])
            print(f"  [✓] Best model saved → {config['model_save_path']}")

        if epoch % 10 == 0:
            ckpt = os.path.join(config["checkpoint_dir"], f"unet_epoch_{epoch}.pth")
            save_checkpoint(model, ckpt)

    save_config(config, config["log_dir"])
    plot_history(history, config["log_dir"])
    print(f"\n[DONE] Best val loss : {best_val_loss:.4f}")
    print(f"[DONE] Model saved   → {config['model_save_path']}")


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train U-Net on LGG MRI Kaggle dataset"
    )
    parser.add_argument("--data_root",       default="data/kaggle_3m",
                        help="Path to the kaggle_3m folder")
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",         type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--image_size", type=int,   default=DEFAULT_CONFIG["image_size"])
    parser.add_argument("--model_save_path",         default=DEFAULT_CONFIG["model_save_path"])
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    config = {**DEFAULT_CONFIG, **vars(args)}
    train(config)