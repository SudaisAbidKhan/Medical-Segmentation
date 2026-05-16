# =============================================================
#  config.py  –  Shared configuration for training & inference
# =============================================================

import os

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'unet_best.pth')

# ── Image dimensions ───────────────────────────────────────────
IMAGE_HEIGHT   = 256
IMAGE_WIDTH    = 256
IMAGE_CHANNELS = 3   # RGB input
MASK_CHANNELS  = 1   # Binary mask output

# ── Model architecture ─────────────────────────────────────────
# Number of feature maps at each encoder level
FEATURES = [64, 128, 256, 512]
DROPOUT  = 0.1

# ── Inference ──────────────────────────────────────────────────
# Sigmoid threshold — pixel is "tumour" if prob > this value
MASK_THRESHOLD = 0.5
