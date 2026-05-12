# =============================================================
#  predict.py  –  Load trained U-Net and run inference on a
#                 single image (used by the Flask backend too)
#
#  Usage (standalone):
#    python predict.py --image path/to/slice.tif
#    python predict.py --image path/to/slice.tif --mask path/to/mask.tif
#    python predict.py --image path/to/slice.tif --save_dir outputs/
# =============================================================

import os
import argparse
import io
import base64
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from model import UNet
from utils import (
    get_device, load_model,
    dice_coefficient, iou_score, pixel_accuracy,
    denormalize, overlay_mask,
)


# ── Inference transform (no augmentation, just resize + normalise) ──

_INFERENCE_TRANSFORM = A.Compose([
    A.Resize(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std =(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ── Image loading helpers ─────────────────────────────────────

def load_image_from_path(image_path: str) -> np.ndarray:
    """
    Load any image file supported by PIL (tif, jpg, png …).
    Returns an uint8 HWC RGB numpy array.
    """
    img = Image.open(image_path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Load an image from raw bytes (used by Flask when receiving
    an uploaded file via request.files).
    Returns an uint8 HWC RGB numpy array.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def preprocess(image_np: np.ndarray) -> torch.Tensor:
    """
    Apply the inference transform to a numpy image.
    Returns a [1, 3, H, W] float32 tensor (batch of 1).
    """
    transformed = _INFERENCE_TRANSFORM(image=image_np)
    tensor = transformed["image"]           # [3, H, W]
    return tensor.unsqueeze(0)              # [1, 3, H, W]


# ── Core prediction function ──────────────────────────────────

def predict(
    image_input,
    model       : UNet         = None,
    device      : torch.device = None,
    model_path  : str          = None,
    threshold   : float        = config.MASK_THRESHOLD,
) -> dict:
    """
    Run U-Net inference on a single image.

    Args:
        image_input : str (file path) | bytes (raw file bytes) |
                      np.ndarray (uint8 HWC RGB)
        model       : pre-loaded UNet (pass this from Flask to avoid
                      reloading the model on every request)
        device      : torch.device (auto-detected if None)
        model_path  : path to .pt weights (used only if model=None)
        threshold   : binarisation threshold (default from config)

    Returns:
        dict with keys:
            "pred_mask_np"    – binary np.ndarray (H, W) float32
            "prob_map_np"     – probability map np.ndarray (H, W) float32
            "overlay_np"      – RGB overlay np.ndarray (H, W, 3) uint8
            "original_np"     – denormalised input np.ndarray (H, W, 3) uint8
            "pred_mask_b64"   – base64 PNG of the binary mask
            "overlay_b64"     – base64 PNG of the overlay image
            "metrics"         – dict (only if ground-truth mask supplied)
    """
    # ── Setup ─────────────────────────────────────────────────
    if device is None:
        device = get_device()

    if model is None:
        path = model_path or config.MODEL_PATH
        model = UNet()
        model = load_model(model, path=path, device=device)

    # ── Load image ────────────────────────────────────────────
    if isinstance(image_input, str):
        image_np = load_image_from_path(image_input)
    elif isinstance(image_input, bytes):
        image_np = load_image_from_bytes(image_input)
    elif isinstance(image_input, np.ndarray):
        image_np = image_input
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    original_h, original_w = image_np.shape[:2]

    # ── Preprocess ────────────────────────────────────────────
    tensor = preprocess(image_np).to(device)   # [1, 3, 256, 256]

    # ── Inference ─────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(tensor)                 # [1, 1, 256, 256]
        probs  = torch.sigmoid(logits)         # [1, 1, 256, 256]
        preds  = (probs > threshold).float()   # [1, 1, 256, 256]

    # ── Post-process ──────────────────────────────────────────
    prob_map_np  = probs[0, 0].cpu().numpy()   # (256, 256) in [0,1]
    pred_mask_np = preds[0, 0].cpu().numpy()   # (256, 256) binary

    # Resize back to original image dimensions
    if (original_h, original_w) != (config.IMAGE_HEIGHT, config.IMAGE_WIDTH):
        prob_pil = Image.fromarray(
            (prob_map_np * 255).astype(np.uint8)
        ).resize((original_w, original_h), Image.BILINEAR)
        prob_map_np = np.array(prob_pil, dtype=np.float32) / 255.0

        pred_pil = Image.fromarray(
            (pred_mask_np * 255).astype(np.uint8)
        ).resize((original_w, original_h), Image.NEAREST)
        pred_mask_np = (np.array(pred_pil) > 127).astype(np.float32)

    # Denormalised original for display
    original_display = denormalize(tensor[0].cpu())  # (H_model, W_model, 3)
    original_display = np.array(
        Image.fromarray(original_display).resize(
            (original_w, original_h), Image.BILINEAR
        )
    )

    # Colour overlay
    overlay_np = overlay_mask(
        original_display,
        pred_mask_np.astype(np.uint8),
        alpha=0.45,
        color=(255, 50, 50),
    )

    # ── Encode results as base64 PNGs (for API responses) ────
    pred_mask_b64 = _ndarray_to_b64_png(
        (pred_mask_np * 255).astype(np.uint8), mode="L"
    )
    overlay_b64 = _ndarray_to_b64_png(overlay_np, mode="RGB")

    result = {
        "pred_mask_np"  : pred_mask_np,
        "prob_map_np"   : prob_map_np,
        "overlay_np"    : overlay_np,
        "original_np"   : original_display,
        "pred_mask_b64" : pred_mask_b64,
        "overlay_b64"   : overlay_b64,
        "metrics"       : {},
    }

    return result


# ── Optional: compute metrics when ground-truth is available ─

def predict_with_metrics(
    image_input,
    mask_input,
    model      = None,
    device     = None,
    model_path = None,
    threshold  = config.MASK_THRESHOLD,
) -> dict:
    """
    Same as predict() but also computes Dice, IoU, and accuracy
    against a provided ground-truth mask.

    Args:
        mask_input : str (path) | bytes | np.ndarray (binary HW)
    """
    result = predict(image_input, model, device, model_path, threshold)

    # Load ground-truth mask
    if isinstance(mask_input, str):
        gt = Image.open(mask_input).convert("L")
    elif isinstance(mask_input, bytes):
        gt = Image.open(io.BytesIO(mask_input)).convert("L")
    elif isinstance(mask_input, np.ndarray):
        gt = Image.fromarray(mask_input)
    else:
        raise TypeError(f"Unsupported mask_input type: {type(mask_input)}")

    gt_np = np.array(gt.resize(
        (result["pred_mask_np"].shape[1],
         result["pred_mask_np"].shape[0]),
        Image.NEAREST,
    ), dtype=np.float32)
    gt_np = (gt_np > 0).astype(np.float32)

    pred = result["pred_mask_np"]
    result["metrics"] = {
        "dice"    : round(dice_coefficient(pred, gt_np),  4),
        "iou"     : round(iou_score(pred, gt_np),         4),
        "accuracy": round(pixel_accuracy(pred, gt_np),    4),
    }
    result["gt_mask_np"]  = gt_np
    result["gt_mask_b64"] = _ndarray_to_b64_png(
        (gt_np * 255).astype(np.uint8), mode="L"
    )
    return result


# ── Visualise and save prediction ─────────────────────────────

def save_prediction_figure(result: dict,
                            save_path: str,
                            show: bool = False):
    """
    Save a 4-panel figure:
      Original  |  Probability Map  |  Binary Mask  |  Overlay
    """
    has_gt = "gt_mask_np" in result
    n_cols = 5 if has_gt else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    panels = [
        (result["original_np"],  "Original MRI",    "viridis"),
        (result["prob_map_np"],  "Probability Map",  "hot"),
        (result["pred_mask_np"], "Predicted Mask",   "gray"),
        (result["overlay_np"],   "Overlay",          None),
    ]
    if has_gt:
        panels.append((result["gt_mask_np"], "Ground Truth", "gray"))

    for ax, (img, title, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    # Add metrics subtitle if available
    if result.get("metrics"):
        m = result["metrics"]
        subtitle = (
            f"Dice={m.get('dice', 'N/A'):.4f}  "
            f"IoU={m.get('iou', 'N/A'):.4f}  "
            f"Accuracy={m.get('accuracy', 'N/A'):.4f}"
        )
        fig.suptitle(subtitle, fontsize=11, y=0.02)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[Predict] Figure saved → {save_path}")

    if show:
        plt.show()
    plt.close()


# ── Helper: numpy array → base64 PNG string ───────────────────

def _ndarray_to_b64_png(arr: np.ndarray, mode: str = "RGB") -> str:
    """
    Encode a numpy array as a base64 PNG string.
    Used to send images through the Flask JSON API.
    """
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── CLI entry point ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run U-Net inference on a single MRI image"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input MRI image (.tif / .jpg / .png)"
    )
    parser.add_argument(
        "--mask", type=str, default=None,
        help="(Optional) Path to ground-truth mask for metric computation"
    )
    parser.add_argument(
        "--model_path", type=str, default=config.MODEL_PATH,
        help="Path to saved model weights (.pt)"
    )
    parser.add_argument(
        "--threshold", type=float, default=config.MASK_THRESHOLD,
        help="Binarisation threshold (default: 0.5)"
    )
    parser.add_argument(
        "--save_dir", type=str, default="outputs",
        help="Directory to save output figure"
    )
    args = parser.parse_args()

    device = get_device()

    # ── Run prediction ────────────────────────────────────────
    if args.mask:
        result = predict_with_metrics(
            image_input = args.image,
            mask_input  = args.mask,
            model_path  = args.model_path,
            device      = device,
            threshold   = args.threshold,
        )
    else:
        result = predict(
            image_input = args.image,
            model_path  = args.model_path,
            device      = device,
            threshold   = args.threshold,
        )

    # ── Print metrics ─────────────────────────────────────────
    if result["metrics"]:
        print("\n[Results]")
        for k, v in result["metrics"].items():
            print(f"  {k.capitalize():<12}: {v:.4f}")
    else:
        tumor_pixels = int(result["pred_mask_np"].sum())
        total_pixels = result["pred_mask_np"].size
        coverage = tumor_pixels / total_pixels * 100
        print(f"\n[Results]")
        print(f"  Tumor pixels  : {tumor_pixels:,}")
        print(f"  Total pixels  : {total_pixels:,}")
        print(f"  Coverage      : {coverage:.2f}%")

    # ── Save figure ───────────────────────────────────────────
    stem = Path(args.image).stem
    save_path = os.path.join(args.save_dir, f"{stem}_prediction.png")
    save_prediction_figure(result, save_path, show=False)