# =============================================================
#  utils.py  –  Shared helpers used across train / evaluate / predict
# =============================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import config


# ── Reproducibility ───────────────────────────────────────────

def set_seed(seed: int = config.RANDOM_SEED):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Device helper ─────────────────────────────────────────────

def get_device() -> torch.device:
    """Auto-select CUDA → MPS (Apple Silicon) → CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using: {device}")
    return device


# ── Loss functions ────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Works on raw logits (applies sigmoid internally).
    Formula:  1 - (2 * |P ∩ T| + ε) / (|P| + |T| + ε)
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs   = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / \
               (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    BCE + Dice combined loss.
    Weights are set in config.py (BCE_WEIGHT, DICE_WEIGHT).
    Using BCEWithLogitsLoss for numerical stability.
    """

    def __init__(self, bce_weight: float  = config.BCE_WEIGHT,
                       dice_weight: float = config.DICE_WEIGHT):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bce_loss  = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ── Metrics (numpy-based, used during evaluation) ────────────

def dice_coefficient(pred: np.ndarray, target: np.ndarray,
                     smooth: float = 1.0) -> float:
    """Dice coefficient between two binary numpy arrays."""
    pred   = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred: np.ndarray, target: np.ndarray,
              smooth: float = 1.0) -> float:
    """Intersection over Union (Jaccard index)."""
    pred   = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Fraction of correctly classified pixels."""
    correct = (pred == target).sum()
    return correct / target.size


# ── Checkpoint helpers ────────────────────────────────────────

def save_checkpoint(state: dict, path: str = config.CHECKPOINT_PATH):
    """Save training checkpoint (model + optimiser state)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[Checkpoint] Saved → {path}")


def load_checkpoint(model, optimizer=None,
                    path: str = config.CHECKPOINT_PATH):
    """
    Load a checkpoint.  If optimizer is None, only model weights are loaded
    (useful for inference).

    Returns:
        start_epoch (int), best_dice (float)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_dice   = checkpoint.get("best_dice", 0.0)
    print(f"[Checkpoint] Loaded from epoch {start_epoch - 1}, "
          f"best Dice = {best_dice:.4f}")
    return start_epoch, best_dice


def save_model(model, path: str = config.MODEL_PATH):
    """Save only model weights (for inference / deployment)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Model] Saved → {path}")


def load_model(model, path: str = config.MODEL_PATH,
               device: torch.device = None):
    """Load model weights from a .pt file."""
    if device is None:
        device = get_device()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    print(f"[Model] Loaded from {path}")
    return model


# ── Visualisation ─────────────────────────────────────────────

def denormalize(tensor: torch.Tensor,
                mean=(0.485, 0.456, 0.406),
                std =(0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Reverse ImageNet normalisation on a [C, H, W] tensor.
    Returns a uint8 HWC numpy array.
    """
    t = tensor.clone().cpu().float()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    t = t.permute(1, 2, 0).numpy()
    t = np.clip(t * 255, 0, 255).astype(np.uint8)
    return t


def overlay_mask(image: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.4,
                 color=(255, 0, 0)) -> np.ndarray:
    """
    Blend a binary mask over an RGB image.

    Args:
        image : HWC uint8 numpy array
        mask  : HW  binary numpy array (0 or 1)
        alpha : transparency of overlay
        color : RGB tuple for mask colour
    Returns:
        HWC uint8 blended image
    """
    overlay = image.copy()
    overlay[mask == 1] = (
        (1 - alpha) * overlay[mask == 1] +
        alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return overlay


def visualize_predictions(images, true_masks, pred_masks,
                           num_samples: int = 4,
                           save_path: str = None):
    """
    Plot a grid: Original | Ground Truth | Prediction | Overlay.

    Args:
        images      : batch tensor [B, 3, H, W]
        true_masks  : batch tensor [B, 1, H, W]
        pred_masks  : batch tensor [B, 1, H, W]  (binary, after threshold)
        num_samples : how many samples to show
        save_path   : if provided, save the figure to this path
    """
    n = min(num_samples, images.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

    if n == 1:
        axes = axes[np.newaxis, :]  # keep 2-D indexing

    col_titles = ["MRI Slice", "Ground Truth", "Prediction", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold")

    for i in range(n):
        img  = denormalize(images[i])
        gt   = true_masks[i, 0].cpu().numpy()
        pred = pred_masks[i, 0].cpu().numpy()
        ovl  = overlay_mask(img, pred.astype(np.uint8))

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(gt,   cmap="gray")
        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 3].imshow(ovl)

        dice = dice_coefficient(pred, gt)
        iou  = iou_score(pred, gt)
        axes[i, 3].set_xlabel(f"Dice={dice:.3f}  IoU={iou:.3f}",
                               fontsize=9)

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[Viz] Saved → {save_path}")

    plt.show()
    plt.close()


def plot_training_curves(train_losses: list, val_losses: list,
                          train_dices: list, val_dices: list,
                          save_path: str = None):
    """Plot loss and Dice curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label="Train Loss", color="steelblue")
    axes[0].plot(epochs, val_losses,   label="Val Loss",   color="coral")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_dices, label="Train Dice", color="steelblue")
    axes[1].plot(epochs, val_dices,   label="Val Dice",   color="coral")
    axes[1].set_title("Dice Score Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[Viz] Training curves saved → {save_path}")

    plt.show()
    plt.close()