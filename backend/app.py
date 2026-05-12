# =============================================================
#  app.py  –  Flask API Server for U-Net Medical Image Segmentation
#
#  Routes:
#    GET  /health          → server + model status check
#    POST /predict         → upload MRI image, get segmentation mask
#    POST /predict-with-mask → upload image + ground truth, get metrics too
#    GET  /model-info      → model architecture details
#
#  Usage:
#    pip install -r requirements.txt
#    python app.py
#    Server runs on http://localhost:5000
# =============================================================

import os
import sys
import uuid
import time
import logging
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── Make ai/ importable from backend/ ────────────────────────
AI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ai')
sys.path.insert(0, os.path.abspath(AI_DIR))

import config
from model   import UNet
from utils   import get_device, load_model
from inference import (
    run_prediction,
    run_prediction_with_mask,
    load_model_once,
)

# ── App setup ─────────────────────────────────────────────────
app = Flask(__name__)

# Allow requests from React dev server (port 3000) and production build
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:5173",   # Vite dev server
            "http://127.0.0.1:3000",
        ]
    }
})

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s  [%(levelname)s]  %(message)s',
    datefmt = '%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── Upload folder ─────────────────────────────────────────────
UPLOAD_FOLDER   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'uploads')
ALLOWED_EXTENSIONS = {'tif', 'tiff', 'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024          # 16 MB max upload

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER']    = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# ── Load model once at startup ────────────────────────────────
logger.info("Loading U-Net model …")
MODEL, DEVICE = load_model_once()
logger.info(f"Model ready on {DEVICE} ✓")


# ── Helpers ───────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def unique_filename(original: str) -> str:
    """Prepend a UUID so uploads never collide."""
    ext  = Path(original).suffix
    return f"{uuid.uuid4().hex}{ext}"


def error_response(message: str, status: int = 400):
    return jsonify({"success": False, "error": message}), status


# ── Routes ────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    React can poll this on startup to confirm the backend is alive.
    """
    model_loaded = MODEL is not None
    return jsonify({
        "success"      : True,
        "status"       : "ok",
        "model_loaded" : model_loaded,
        "device"       : str(DEVICE),
        "model_path"   : config.MODEL_PATH,
        "timestamp"    : time.time(),
    }), 200


@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Return U-Net architecture details — shown in the React About page.
    """
    if MODEL is None:
        return error_response("Model not loaded", 503)

    return jsonify({
        "success"           : True,
        "architecture"      : "U-Net",
        "encoder_features"  : config.FEATURES,
        "input_shape"       : [config.IMAGE_CHANNELS,
                               config.IMAGE_HEIGHT,
                               config.IMAGE_WIDTH],
        "output_shape"      : [config.MASK_CHANNELS,
                               config.IMAGE_HEIGHT,
                               config.IMAGE_WIDTH],
        "trainable_params"  : MODEL.count_parameters(),
        "mask_threshold"    : config.MASK_THRESHOLD,
        "dataset"           : "LGG Brain MRI Segmentation (Kaggle)",
        "paper"             : "Ronneberger et al., MICCAI 2015",
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts a single MRI image and returns the predicted segmentation mask.

    Request  (multipart/form-data):
        file  : image file  (.tif / .png / .jpg)

    Response (JSON):
        {
          "success"        : true,
          "pred_mask_b64"  : "<base64 PNG>",
          "overlay_b64"    : "<base64 PNG>",
          "original_b64"   : "<base64 PNG>",
          "tumor_coverage" : 12.34,      // % of pixels predicted as tumor
          "inference_ms"   : 45.2,       // model inference time in ms
        }
    """
    # ── Validate request ──────────────────────────────────────
    if 'file' not in request.files:
        return error_response("No file part in request. "
                              "Send image as 'file' field.")

    file = request.files['file']

    if file.filename == '':
        return error_response("No file selected.")

    if not allowed_file(file.filename):
        return error_response(
            f"File type not allowed. Supported: "
            f"{', '.join(ALLOWED_EXTENSIONS)}"
        )

    # ── Read image bytes ──────────────────────────────────────
    try:
        image_bytes = file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        return error_response("Failed to read uploaded file.", 500)

    # ── Save a copy to uploads/ for debugging ─────────────────
    try:
        save_name = unique_filename(secure_filename(file.filename))
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        with open(save_path, 'wb') as f:
            f.write(image_bytes)
        logger.info(f"Uploaded file saved → uploads/{save_name}")
    except Exception as e:
        logger.warning(f"Could not save upload to disk: {e}")
        # Non-fatal — continue with prediction

    # ── Run inference ─────────────────────────────────────────
    try:
        t_start = time.perf_counter()
        result  = run_prediction(
            image_bytes = image_bytes,
            model       = MODEL,
            device      = DEVICE,
        )
        inference_ms = (time.perf_counter() - t_start) * 1000
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return error_response(f"Inference failed: {str(e)}", 500)

    logger.info(f"Prediction done in {inference_ms:.1f} ms  |  "
                f"Coverage: {result['tumor_coverage']:.2f}%")

    return jsonify({
        "success"        : True,
        "pred_mask_b64"  : result["pred_mask_b64"],
        "overlay_b64"    : result["overlay_b64"],
        "original_b64"   : result["original_b64"],
        "prob_map_b64"   : result["prob_map_b64"],
        "tumor_coverage" : round(result["tumor_coverage"], 4),
        "inference_ms"   : round(inference_ms, 2),
    }), 200


@app.route('/predict-with-mask', methods=['POST'])
def predict_with_mask():
    """
    Accepts an MRI image AND a ground-truth mask.
    Returns prediction + Dice / IoU / Accuracy metrics.
    Useful for demo / evaluation mode in the React UI.

    Request (multipart/form-data):
        image : MRI image file
        mask  : ground-truth mask file

    Response (JSON):
        { ...same as /predict..., "metrics": { dice, iou, accuracy } }
    """
    # ── Validate both files ───────────────────────────────────
    if 'image' not in request.files:
        return error_response("Missing 'image' field in request.")
    if 'mask' not in request.files:
        return error_response("Missing 'mask' field in request.")

    image_file = request.files['image']
    mask_file  = request.files['mask']

    for f, label in [(image_file, 'image'), (mask_file, 'mask')]:
        if f.filename == '':
            return error_response(f"No {label} file selected.")
        if not allowed_file(f.filename):
            return error_response(
                f"{label} file type not allowed. "
                f"Supported: {', '.join(ALLOWED_EXTENSIONS)}"
            )

    try:
        image_bytes = image_file.read()
        mask_bytes  = mask_file.read()
    except Exception as e:
        return error_response(f"Failed to read files: {e}", 500)

    # ── Run inference with ground-truth comparison ────────────
    try:
        t_start = time.perf_counter()
        result  = run_prediction_with_mask(
            image_bytes = image_bytes,
            mask_bytes  = mask_bytes,
            model       = MODEL,
            device      = DEVICE,
        )
        inference_ms = (time.perf_counter() - t_start) * 1000
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return error_response(f"Inference failed: {str(e)}", 500)

    logger.info(
        f"Prediction+metrics done in {inference_ms:.1f} ms  |  "
        f"Dice={result['metrics']['dice']:.4f}  "
        f"IoU={result['metrics']['iou']:.4f}"
    )

    return jsonify({
        "success"        : True,
        "pred_mask_b64"  : result["pred_mask_b64"],
        "overlay_b64"    : result["overlay_b64"],
        "original_b64"   : result["original_b64"],
        "prob_map_b64"   : result["prob_map_b64"],
        "gt_mask_b64"    : result["gt_mask_b64"],
        "tumor_coverage" : round(result["tumor_coverage"], 4),
        "inference_ms"   : round(inference_ms, 2),
        "metrics"        : result["metrics"],
    }), 200


# ── Error handlers ────────────────────────────────────────────

@app.errorhandler(413)
def file_too_large(e):
    return error_response(
        f"File too large. Maximum size is "
        f"{MAX_CONTENT_LENGTH // (1024*1024)} MB.", 413
    )


@app.errorhandler(404)
def not_found(e):
    return error_response("Endpoint not found.", 404)


@app.errorhandler(500)
def server_error(e):
    return error_response("Internal server error.", 500)


# ── Entry point ───────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  U-Net Medical Segmentation — Flask API")
    print("="*55)
    print(f"  Model  : {config.MODEL_PATH}")
    print(f"  Device : {DEVICE}")
    print(f"  Upload : {UPLOAD_FOLDER}")
    print(f"  URL    : http://localhost:5000")
    print("="*55 + "\n")

    app.run(
        host  = '0.0.0.0',
        port  = 5000,
        debug = False,      # set True during development
    )