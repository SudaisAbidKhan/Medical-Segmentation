"""
predict.py — U-Net Inference + Flask API Server
LGG MRI Segmentation (Kaggle Dataset)

Standalone inference:
    !python predict.py --image_path scan.tif --model_path models/unet_best.pth

Start API server (Colab → Frontend):
    !python predict.py --serve --model_path models/unet_best.pth
"""

import os
import io
import argparse
import base64

import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import cv2

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


# ─────────────────────────────────────────────
#  Shared config — must match train.py exactly
# ─────────────────────────────────────────────

MODEL_CONFIG = {
    "encoder":      "resnet34",
    "in_channels":  3,
    "num_classes":  1,
    "image_size":   256,
    "threshold":    0.5,
}

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)


# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────

def build_model() -> torch.nn.Module:
    """Same architecture as train.py — must stay in sync."""
    return smp.Unet(
        encoder_name=MODEL_CONFIG["encoder"],
        encoder_weights=None,               # Loaded from checkpoint
        in_channels=MODEL_CONFIG["in_channels"],
        classes=MODEL_CONFIG["num_classes"],
        activation=None,
    )


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}\n"
            "Run train.py first to generate the model."
        )
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded → {model_path}")
    return model


# ─────────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────────

def get_inference_transform(image_size: int) -> A.Compose:
    """Identical to val transforms in train.py."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])


def preprocess_image(image_input, image_size: int) -> tuple:
    """
    Accept a file path (str) or raw bytes (from API).
    Handles .tif, .png, .jpg via PIL — consistent with train.py.

    Returns:
        tensor   → (1, C, H, W) ready for model
        original → (H, W, 3) RGB numpy array for visualization
    """
    if isinstance(image_input, (str, os.PathLike)):
        image = np.array(Image.open(str(image_input)).convert("RGB"))
    else:
        image = np.array(Image.open(io.BytesIO(image_input)).convert("RGB"))

    original  = image.copy()
    transform = get_inference_transform(image_size)
    tensor    = transform(image=image)["image"].unsqueeze(0)   # (1, C, H, W)
    return tensor, original


# ─────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────

@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    threshold: float = MODEL_CONFIG["threshold"],
) -> np.ndarray:
    """
    Run forward pass and return a binary mask.
    Returns:
        mask → (H, W) uint8, values 0 or 255
    """
    logits = model(image_tensor.to(device))                    # (1, 1, H, W)
    prob   = torch.sigmoid(logits).squeeze().cpu().numpy()     # (H, W)
    return (prob > threshold).astype(np.uint8) * 255


# ─────────────────────────────────────────────
#  Overlay & Visualization
# ─────────────────────────────────────────────

def overlay_mask(original: np.ndarray, mask: np.ndarray,
                 color=(255, 0, 0), alpha: float = 0.4) -> np.ndarray:
    """
    Blend a colored segmentation mask onto the original image.
    Args:
        original : (H, W, 3) RGB
        mask     : (H, W)    binary mask, values 0 or 255
    """
    h, w          = original.shape[:2]
    mask_resized  = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay       = original.copy().astype(np.float32)
    region        = mask_resized > 0

    for c, col_val in enumerate(color):
        overlay[region, c] = overlay[region, c] * (1 - alpha) + col_val * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_visualization(original: np.ndarray, mask: np.ndarray, output_path: str):
    """Save side-by-side figure: Original | Mask | Overlay."""
    h, w         = original.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_rgb     = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2RGB)
    overlay      = overlay_mask(original, mask_resized)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(
        axes,
        [original, mask_rgb, overlay],
        ["Original MRI", "Predicted Mask", "Overlay"],
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Visualization saved → {output_path}")


# ─────────────────────────────────────────────
#  Flask API
# ─────────────────────────────────────────────

def encode_to_base64(image_rgb: np.ndarray) -> str:
    """Convert an RGB numpy array to a base64 PNG string."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer  = cv2.imencode(".png", image_bgr)
    return base64.b64encode(buffer).decode("utf-8")


def create_app(model: torch.nn.Module, device: torch.device) -> "Flask":
    if not FLASK_AVAILABLE:
        raise ImportError("Run: pip install flask flask-cors")

    app = Flask(__name__)
    CORS(app)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "device": str(device)})

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        Accepts : multipart/form-data with field 'image'
                  Supports .tif, .png, .jpg

        Returns : JSON
            {
                "mask_b64":    "<base64 PNG of binary mask>",
                "overlay_b64": "<base64 PNG of overlay>",
                "width": int,
                "height": int
            }

        Frontend usage:
            const form = new FormData();
            form.append('image', file);
            const res = await axios.post(NGROK_URL + '/predict', form);
            // res.data.mask_b64    → display as <img src="data:image/png;base64,...">
            // res.data.overlay_b64 → same
        """
        if "image" not in request.files:
            return jsonify({"error": "Missing 'image' field"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        try:
            raw_bytes          = file.read()
            tensor, original   = preprocess_image(raw_bytes, MODEL_CONFIG["image_size"])
            mask               = predict_mask(model, tensor, device)

            h, w         = original.shape[:2]
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_rgb     = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2RGB)
            overlay      = overlay_mask(original, mask_resized)

            return jsonify({
                "mask_b64":    encode_to_base64(mask_rgb),
                "overlay_b64": encode_to_base64(overlay),
                "width":  w,
                "height": h,
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def start_server(model_path: str, port: int = 5000, use_ngrok: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(model_path, device)
    app    = create_app(model, device)

    if use_ngrok:
        try:
            from flask_ngrok import run_with_ngrok
            run_with_ngrok(app)
            print("[INFO] ngrok tunnel active — check output for public URL")
            app.run()
        except ImportError:
            print("[WARN] flask-ngrok not found — starting on localhost only.")
            app.run(host="0.0.0.0", port=port)
    else:
        app.run(host="0.0.0.0", port=port, debug=False)


# ─────────────────────────────────────────────
#  Standalone CLI Inference
# ─────────────────────────────────────────────

def run_single_inference(image_path: str, model_path: str, output_path: str):
    device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model            = load_model(model_path, device)
    tensor, original = preprocess_image(image_path, MODEL_CONFIG["image_size"])
    mask             = predict_mask(model, tensor, device)
    save_visualization(original, mask, output_path)
    print(f"[DONE] Segmentation complete → {output_path}")


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="U-Net inference — standalone or API server"
    )
    parser.add_argument("--model_path",  default="models/unet_best.pth")
    parser.add_argument("--image_path",  default=None,
                        help="Single .tif image for inference")
    parser.add_argument("--output_path", default="output/prediction.png")
    parser.add_argument("--serve",       action="store_true",
                        help="Start Flask API server")
    parser.add_argument("--port",        type=int, default=5000)
    parser.add_argument("--no_ngrok",    action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.serve:
        start_server(
            model_path=args.model_path,
            port=args.port,
            use_ngrok=not args.no_ngrok,
        )
    elif args.image_path:
        run_single_inference(args.image_path, args.model_path, args.output_path)
    else:
        print(
            "Usage:\n"
            "  Inference  : python predict.py --image_path scan.tif\n"
            "  API server : python predict.py --serve"
        )