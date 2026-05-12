# =============================================================
#  inference.py  –  Model loading + prediction logic
#
#  This file is the bridge between app.py (Flask) and the
#  ai/ folder (PyTorch model).  It is responsible for:
#    1. Loading the model exactly once at server startup
#    2. Preprocessing raw image bytes into tensors
#    3. Running forward pass and returning base64-encoded results
#    4. Computing metrics when ground-truth masks are supplied
# =============================================================

import os
import sys
import io
import base64
import logging

import numpy as np
from PIL import Image

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Make ai/ importable ───────────────────────────────────────
AI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ai')
sys.path.insert(0, os.path.abspath(AI_DIR))

import config
from model import UNet
from utils import get_device

logger = logging.getLogger(__name__)

# ── Inference transform (resize + normalise only, no augmentation) ──
_TRANSFORM = A.Compose([
    A.Resize(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std =(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])


# ═════════════════════════════════════════════════════════════
#  1.  Model loader  (called once at app startup)
# ═════════════════════════════════════════════════════════════

def load_model_once(model_path: str = None,
                    device: torch.device = None):
    """
    Load the trained U-Net weights and return (model, device).
    Called once when Flask starts — model is kept in memory for
    every subsequent request (no reload overhead).

    Args:
        model_path : path to unet_model.pt  (default from config)
        device     : torch.device  (auto-detected if None)

    Returns:
        model  : UNet in eval mode on the chosen device
        device : torch.device
    """
    if device is None:
        device = get_device()

    path = model_path or config.MODEL_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model weights not found at: {path}\n"
            f"Run  python ai/generate_model.py  to create a placeholder, or\n"
            f"download your trained weights from Kaggle and place them at {path}"
        )

    model = UNet(
        in_channels  = config.IMAGE_CHANNELS,
        out_channels = config.MASK_CHANNELS,
        features     = config.FEATURES,
        dropout      = config.DROPOUT,
    )
    state_dict = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from {path}  ({model.count_parameters():,} params)")
    return model, device


# ═════════════════════════════════════════════════════════════
#  2.  Image helpers
# ═════════════════════════════════════════════════════════════

def _bytes_to_rgb_array(file_bytes: bytes) -> np.ndarray:
    """
    Convert raw file bytes (any PIL-supported format) to
    a uint8 (H, W, 3) RGB numpy array.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    return np.array(img, dtype=np.uint8)


def _bytes_to_mask_array(file_bytes: bytes) -> np.ndarray:
    """
    Convert raw mask file bytes to a binary float32 (H, W) numpy array.
    Any non-zero pixel → 1.
    """
    mask = Image.open(io.BytesIO(file_bytes)).convert('L')
    arr  = np.array(mask, dtype=np.float32)
    return (arr > 0).astype(np.float32)


def _preprocess(image_np: np.ndarray) -> torch.Tensor:
    """
    Apply inference transform to a numpy image.
    Returns a [1, 3, H, W] float32 tensor (batch of 1).
    """
    aug    = _TRANSFORM(image=image_np)
    tensor = aug['image']           # [3, H, W]
    return tensor.unsqueeze(0)      # [1, 3, H, W]


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalisation on a [3, H, W] tensor.
    Returns a uint8 (H, W, 3) numpy array for display.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    t    = tensor.cpu().permute(1, 2, 0).numpy()
    t    = t * std + mean
    return np.clip(t * 255, 0, 255).astype(np.uint8)


# ═════════════════════════════════════════════════════════════
#  3.  Post-processing helpers
# ═════════════════════════════════════════════════════════════

def _build_overlay(image_np: np.ndarray,
                   mask_np: np.ndarray,
                   alpha: float = 0.45,
                   color=(255, 50, 50)) -> np.ndarray:
    """
    Blend a binary mask over an RGB image with a red highlight.

    Args:
        image_np : (H, W, 3) uint8
        mask_np  : (H, W)    binary float32
        alpha    : overlay transparency
        color    : RGB highlight colour

    Returns:
        (H, W, 3) uint8 blended image
    """
    overlay = image_np.copy()
    roi = mask_np == 1
    if roi.any():
        overlay[roi] = (
            (1 - alpha) * overlay[roi].astype(np.float32) +
            alpha * np.array(color, dtype=np.float32)
        ).astype(np.uint8)
    return overlay


def _build_prob_map_image(prob_np: np.ndarray) -> np.ndarray:
    """
    Convert a float [0,1] probability map to a uint8 heatmap (H, W, 3).
    Uses a red-yellow colourmap so hot areas stand out.
    """
    # Simple manual hot colourmap: black → red → yellow → white
    p = prob_np                         # (H, W) in [0, 1]
    r = np.clip(p * 3,       0, 1)
    g = np.clip(p * 3 - 1,   0, 1)
    b = np.clip(p * 3 - 2,   0, 1)
    heatmap = np.stack([r, g, b], axis=-1)
    return (heatmap * 255).astype(np.uint8)


def _resize_mask_to_original(mask_np: np.ndarray,
                              original_h: int,
                              original_w: int) -> np.ndarray:
    """Resize a mask back to the original image dimensions."""
    pil = Image.fromarray((mask_np * 255).astype(np.uint8))
    pil = pil.resize((original_w, original_h), Image.NEAREST)
    return (np.array(pil) > 127).astype(np.float32)


def _resize_prob_to_original(prob_np: np.ndarray,
                              original_h: int,
                              original_w: int) -> np.ndarray:
    """Resize a probability map back to the original image dimensions."""
    pil = Image.fromarray((prob_np * 255).astype(np.uint8))
    pil = pil.resize((original_w, original_h), Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


def _ndarray_to_b64(arr: np.ndarray, mode: str = 'RGB') -> str:
    """
    Encode a numpy array as a base64 PNG string.
    React reads this as:  <img src={`data:image/png;base64,${b64}`} />
    """
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ═════════════════════════════════════════════════════════════
#  4.  Metrics
# ═════════════════════════════════════════════════════════════

def _compute_metrics(pred: np.ndarray,
                     target: np.ndarray) -> dict:
    """
    Compute Dice, IoU, and pixel accuracy between two binary arrays.
    Both inputs should be float32 with values in {0, 1}.
    """
    p = pred.flatten()
    t = target.flatten()

    tp = float((p * t).sum())
    fp = float((p * (1 - t)).sum())
    fn = float(((1 - p) * t).sum())
    tn = float(((1 - p) * (1 - t)).sum())

    dice     = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou      = (tp + 1e-6)     / (tp + fp + fn + 1e-6)
    accuracy = (tp + tn)       / (tp + tn + fp + fn + 1e-6)

    return {
        'dice'    : round(float(dice),     4),
        'iou'     : round(float(iou),      4),
        'accuracy': round(float(accuracy), 4),
    }


# ═════════════════════════════════════════════════════════════
#  5.  Core prediction functions  (called by app.py)
# ═════════════════════════════════════════════════════════════

def run_prediction(image_bytes: bytes,
                   model: UNet,
                   device: torch.device,
                   threshold: float = config.MASK_THRESHOLD) -> dict:
    """
    Run U-Net inference on raw image bytes.

    Args:
        image_bytes : raw bytes from Flask request.files['file'].read()
        model       : pre-loaded UNet (in eval mode)
        device      : torch.device
        threshold   : sigmoid binarisation threshold

    Returns dict with keys:
        pred_mask_b64   – base64 PNG of binary mask          (grayscale)
        overlay_b64     – base64 PNG of mask overlaid on MRI (RGB)
        original_b64    – base64 PNG of the input MRI        (RGB)
        prob_map_b64    – base64 PNG of probability heatmap  (RGB)
        tumor_coverage  – float, % of pixels predicted as tumour
    """
    # ── Load & preprocess ──────────────────────────────────────
    image_np  = _bytes_to_rgb_array(image_bytes)
    orig_h, orig_w = image_np.shape[:2]

    tensor = _preprocess(image_np).to(device)   # [1, 3, 256, 256]

    # ── Forward pass ──────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(tensor)                          # [1, 1, H, W]
        probs  = torch.sigmoid(logits)                  # [1, 1, H, W]
        preds  = (probs > threshold).float()            # [1, 1, H, W]

    prob_map_256  = probs[0, 0].cpu().numpy()           # (256, 256)
    pred_mask_256 = preds[0, 0].cpu().numpy()           # (256, 256)

    # ── Resize outputs to original image dimensions ────────────
    pred_mask = _resize_mask_to_original(pred_mask_256, orig_h, orig_w)
    prob_map  = _resize_prob_to_original(prob_map_256,  orig_h, orig_w)

    # ── Build display images ───────────────────────────────────
    original_display = _denormalize(tensor[0].cpu())    # (256, 256, 3)
    original_display = np.array(
        Image.fromarray(original_display)
             .resize((orig_w, orig_h), Image.BILINEAR)
    )

    overlay      = _build_overlay(original_display, pred_mask)
    prob_map_img = _build_prob_map_image(prob_map)

    # ── Tumour coverage ───────────────────────────────────────
    tumor_coverage = float(pred_mask.sum() / pred_mask.size * 100)

    return {
        'pred_mask_b64' : _ndarray_to_b64(
                            (pred_mask * 255).astype(np.uint8), mode='L'),
        'overlay_b64'   : _ndarray_to_b64(overlay,       mode='RGB'),
        'original_b64'  : _ndarray_to_b64(original_display, mode='RGB'),
        'prob_map_b64'  : _ndarray_to_b64(prob_map_img,  mode='RGB'),
        'tumor_coverage': tumor_coverage,
    }


def run_prediction_with_mask(image_bytes: bytes,
                              mask_bytes: bytes,
                              model: UNet,
                              device: torch.device,
                              threshold: float = config.MASK_THRESHOLD
                              ) -> dict:
    """
    Same as run_prediction() but also accepts ground-truth mask bytes
    and returns Dice / IoU / Accuracy metrics.

    Extra return keys:
        gt_mask_b64  – base64 PNG of the ground-truth mask
        metrics      – dict  { dice, iou, accuracy }
    """
    result = run_prediction(image_bytes, model, device, threshold)

    # ── Load ground-truth mask ─────────────────────────────────
    gt_np = _bytes_to_mask_array(mask_bytes)

    # Resize GT to model output size for fair metric comparison
    gt_resized = _resize_mask_to_original(
        gt_np,
        *[int(x) for x in Image.open(
            io.BytesIO(image_bytes)
        ).size[::-1]]          # (orig_h, orig_w)
    )

    # Re-derive prediction at the same dimensions
    pred_np = np.array(
        Image.open(
            io.BytesIO(
                base64.b64decode(result['pred_mask_b64'])
            )
        ).convert('L'), dtype=np.float32
    )
    pred_np = (pred_np > 127).astype(np.float32)

    # ── Align dimensions ──────────────────────────────────────
    if pred_np.shape != gt_resized.shape:
        pred_pil = Image.fromarray((pred_np * 255).astype(np.uint8))
        pred_pil = pred_pil.resize(
            (gt_resized.shape[1], gt_resized.shape[0]),
            Image.NEAREST
        )
        pred_np = (np.array(pred_pil) > 127).astype(np.float32)

    metrics = _compute_metrics(pred_np, gt_resized)

    result['gt_mask_b64'] = _ndarray_to_b64(
        (gt_resized * 255).astype(np.uint8), mode='L'
    )
    result['metrics'] = metrics

    logger.info(f"Metrics → Dice={metrics['dice']}  "
                f"IoU={metrics['iou']}  "
                f"Accuracy={metrics['accuracy']}")
    return result