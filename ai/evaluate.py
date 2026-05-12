# =============================================================
#  evaluate.py  –  Evaluate trained U-Net on the test set
#
#  Metrics reported:
#    • Dice Coefficient  (primary metric)
#    • IoU / Jaccard Index
#    • Pixel Accuracy
#    • Precision & Recall
#
#  Usage:
#    python evaluate.py
#    python evaluate.py --kaggle_root path/to/kaggle_3m
#    python evaluate.py --model_path saved_models/unet_model.pt
# =============================================================

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

import config
from model   import UNet
from dataset import get_dataloaders
from utils   import (
    get_device, load_model,
    dice_coefficient, iou_score, pixel_accuracy,
    visualize_predictions, denormalize,
)


# ── Per-batch metric computation ─────────────────────────────

def compute_batch_metrics(preds: np.ndarray,
                           targets: np.ndarray) -> dict:
    """
    Compute all metrics for a single batch.
    Both arrays are expected to be binary float32 numpy arrays.

    Returns a dict with keys: dice, iou, accuracy, precision, recall
    """
    preds   = preds.flatten()
    targets = targets.flatten()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()

    dice      = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou       = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall    = (tp + 1e-6) / (tp + fn + 1e-6)

    return {
        "dice"     : float(dice),
        "iou"      : float(iou),
        "accuracy" : float(accuracy),
        "precision": float(precision),
        "recall"   : float(recall),
    }


# ── Full test-set evaluation ──────────────────────────────────

@torch.no_grad()
def evaluate(model, test_loader, device,
             save_viz: bool = True,
             viz_samples: int = 8):
    """
    Run the model over the entire test set and aggregate metrics.

    Args:
        model       : trained UNet (already on device, in eval mode)
        test_loader : DataLoader with batch_size=1
        device      : torch.device
        save_viz    : whether to save prediction visualisations
        viz_samples : how many sample images to visualise

    Returns:
        results (dict) with mean ± std for every metric
    """
    model.eval()

    all_metrics = {
        "dice"     : [],
        "iou"      : [],
        "accuracy" : [],
        "precision": [],
        "recall"   : [],
    }

    # Buffers for visualisation (first N samples)
    viz_images     = []
    viz_true_masks = []
    viz_pred_masks = []

    print("\n[Evaluate] Running inference on test set …")

    for idx, (images, masks) in enumerate(tqdm(test_loader, unit="image")):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        # Forward pass
        logits = model(images)                          # [1, 1, H, W]
        probs  = torch.sigmoid(logits)
        preds  = (probs > config.MASK_THRESHOLD).float()

        # Compute metrics (numpy)
        m = compute_batch_metrics(
            preds.cpu().numpy(),
            masks.cpu().numpy(),
        )
        for key in all_metrics:
            all_metrics[key].append(m[key])

        # Collect samples for visualisation
        if idx < viz_samples:
            viz_images.append(images.cpu())
            viz_true_masks.append(masks.cpu())
            viz_pred_masks.append(preds.cpu())

    # ── Aggregate ─────────────────────────────────────────────
    results = {}
    for key, values in all_metrics.items():
        arr = np.array(values)
        results[key] = {
            "mean": float(arr.mean()),
            "std" : float(arr.std()),
            "min" : float(arr.min()),
            "max" : float(arr.max()),
        }

    # ── Print report ──────────────────────────────────────────
    _print_report(results, n_images=len(test_loader))

    # ── Distribution histograms ───────────────────────────────
    _plot_metric_distributions(
        all_metrics,
        save_path=os.path.join(config.SAVED_MODEL_DIR, "metric_distributions.png"),
    )

    # ── Prediction visualisations ─────────────────────────────
    if save_viz and viz_images:
        images_batch     = torch.cat(viz_images,     dim=0)
        true_masks_batch = torch.cat(viz_true_masks, dim=0)
        pred_masks_batch = torch.cat(viz_pred_masks, dim=0)

        visualize_predictions(
            images_batch, true_masks_batch, pred_masks_batch,
            num_samples=viz_samples,
            save_path=os.path.join(config.SAVED_MODEL_DIR,
                                   "test_predictions.png"),
        )

    return results


# ── Pretty-print helper ───────────────────────────────────────

def _print_report(results: dict, n_images: int):
    """Print a formatted evaluation report to stdout."""
    bar = "=" * 55
    print(f"\n{bar}")
    print(f"  U-Net Evaluation Report  ({n_images} test images)")
    print(bar)
    print(f"  {'Metric':<12} {'Mean':>8}  {'Std':>8}  "
          f"{'Min':>8}  {'Max':>8}")
    print(f"  {'-'*50}")
    for metric, stats in results.items():
        print(f"  {metric.capitalize():<12} "
              f"{stats['mean']:>8.4f}  "
              f"{stats['std']:>8.4f}  "
              f"{stats['min']:>8.4f}  "
              f"{stats['max']:>8.4f}")
    print(bar)

    # Highlight primary metrics
    dice_mean = results["dice"]["mean"]
    iou_mean  = results["iou"]["mean"]
    grade = "Excellent" if dice_mean >= 0.85 else \
            "Good"      if dice_mean >= 0.70 else \
            "Fair"      if dice_mean >= 0.55 else "Needs improvement"

    print(f"\n  ✓ Dice Coefficient : {dice_mean:.4f}  ({grade})")
    print(f"  ✓ IoU / Jaccard    : {iou_mean:.4f}")
    print(f"\n  Target benchmarks: Dice ≥ 0.80 | IoU ≥ 0.70")
    print(f"{bar}\n")


# ── Metric distribution plots ─────────────────────────────────

def _plot_metric_distributions(all_metrics: dict,
                                save_path: str = None):
    """Histogram of per-image metric values across the test set."""
    metrics_to_plot = ["dice", "iou", "accuracy", "precision", "recall"]
    n = len(metrics_to_plot)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    colors = ["steelblue", "coral", "seagreen", "mediumpurple", "orange"]

    for ax, metric, color in zip(axes, metrics_to_plot, colors):
        values = all_metrics[metric]
        ax.hist(values, bins=20, color=color, edgecolor="white",
                alpha=0.85)
        ax.axvline(np.mean(values), color="black", linestyle="--",
                   linewidth=1.5, label=f"Mean={np.mean(values):.3f}")
        ax.set_title(metric.capitalize(), fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Image Metric Distributions (Test Set)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[Evaluate] Distribution plot saved → {save_path}")

    plt.show()
    plt.close()


# ── Threshold sensitivity analysis ───────────────────────────

@torch.no_grad()
def threshold_analysis(model, val_loader, device,
                        thresholds=None,
                        save_path: str = None):
    """
    Sweep over different binarisation thresholds and report
    Dice + IoU at each. Helps choose the optimal threshold.

    Args:
        thresholds : list of floats (default: 0.1 to 0.9 in steps of 0.05)
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05).tolist()

    model.eval()

    # Collect all logits and masks first (one pass)
    all_probs  = []
    all_masks  = []

    print("[Threshold Analysis] Collecting predictions …")
    for images, masks in tqdm(val_loader, unit="batch"):
        images = images.to(device)
        logits = model(images)
        probs  = torch.sigmoid(logits).cpu()
        all_probs.append(probs)
        all_masks.append(masks)

    all_probs = torch.cat(all_probs, dim=0).numpy()   # [N, 1, H, W]
    all_masks = torch.cat(all_masks, dim=0).numpy()   # [N, 1, H, W]

    # Sweep thresholds
    records = []
    for t in thresholds:
        preds = (all_probs > t).astype(np.float32)
        dice  = dice_coefficient(preds, all_masks)
        iou   = iou_score(preds, all_masks)
        records.append({"threshold": t, "dice": dice, "iou": iou})
        print(f"  Threshold {t:.2f}  →  Dice={dice:.4f}  IoU={iou:.4f}")

    # Best threshold by Dice
    best = max(records, key=lambda r: r["dice"])
    print(f"\n  Best threshold: {best['threshold']:.2f}  "
          f"(Dice={best['dice']:.4f}  IoU={best['iou']:.4f})")

    # Plot
    ts    = [r["threshold"] for r in records]
    dices = [r["dice"]      for r in records]
    ious  = [r["iou"]       for r in records]

    plt.figure(figsize=(8, 4))
    plt.plot(ts, dices, marker="o", label="Dice", color="steelblue")
    plt.plot(ts, ious,  marker="s", label="IoU",  color="coral")
    plt.axvline(best["threshold"], color="gray", linestyle="--",
                label=f"Best={best['threshold']:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Dice & IoU vs Binarisation Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[ThresholdAnalysis] Plot saved → {save_path}")

    plt.show()
    plt.close()
    return records


# ── CLI entry point ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained U-Net on test set"
    )
    parser.add_argument(
        "--kaggle_root", type=str, default=None,
        help="Path to kaggle_3m folder (default: data/kaggle_3m)"
    )
    parser.add_argument(
        "--model_path", type=str, default=config.MODEL_PATH,
        help="Path to saved model weights (.pt)"
    )
    parser.add_argument(
        "--threshold_sweep", action="store_true",
        help="Run threshold sensitivity analysis on validation set"
    )
    args = parser.parse_args()

    device = get_device()

    # Load model
    model = UNet()
    model = load_model(model, path=args.model_path, device=device)

    # Load data
    kaggle_root = args.kaggle_root or os.path.join(
        config.DATA_DIR, "kaggle_3m"
    )
    train_loader, val_loader, test_loader = get_dataloaders(kaggle_root)

    # Evaluate on test set
    results = evaluate(model, test_loader, device)

    # Optional threshold sweep on validation set
    if args.threshold_sweep:
        threshold_analysis(
            model, val_loader, device,
            save_path=os.path.join(config.SAVED_MODEL_DIR,
                                   "threshold_analysis.png"),
        )