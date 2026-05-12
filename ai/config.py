# =============================================================
#  config.py  –  Central configuration for the entire project
# =============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
IMAGE_DIR       = os.path.join(DATA_DIR, "images")
MASK_DIR        = os.path.join(DATA_DIR, "masks")
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_PATH      = os.path.join(SAVED_MODEL_DIR, "unet_model.pt")
CHECKPOINT_PATH = os.path.join(SAVED_MODEL_DIR, "checkpoint.pt")

# ── Image settings ───────────────────────────────────────────
IMAGE_HEIGHT = 256          # Resize all images to this height
IMAGE_WIDTH  = 256          # Resize all images to this width
IMAGE_CHANNELS = 3          # RGB (LGG dataset has 3-channel .tif files)
MASK_CHANNELS  = 1          # Binary segmentation mask

# ── Dataset split ────────────────────────────────────────────
TRAIN_SPLIT = 0.80          # 80 % training
VAL_SPLIT   = 0.10          # 10 % validation
TEST_SPLIT  = 0.10          # 10 % testing
RANDOM_SEED = 42

# ── Training hyperparameters ─────────────────────────────────
BATCH_SIZE    = 16
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-4        # Adam optimizer LR
WEIGHT_DECAY  = 1e-5        # L2 regularisation
NUM_WORKERS   = 4           # DataLoader worker threads (set 0 on Windows)
PIN_MEMORY    = True        # Faster GPU transfer

# ── U-Net architecture ───────────────────────────────────────
FEATURES = [64, 128, 256, 512]   # Filters at each encoder level
DROPOUT  = 0.3                   # Dropout rate in bottleneck

# ── Loss function weights ────────────────────────────────────
# Combined loss = BCE_weight * BCE + DICE_weight * DiceLoss
BCE_WEIGHT  = 0.5
DICE_WEIGHT = 0.5

# ── Early stopping ───────────────────────────────────────────
PATIENCE = 10               # Stop if val loss doesn't improve for N epochs

# ── Evaluation thresholds ────────────────────────────────────
MASK_THRESHOLD = 0.5        # Sigmoid output > threshold → foreground pixel

# ── Logging / visualisation ──────────────────────────────────
LOG_INTERVAL   = 5          # Print metrics every N batches
SAVE_BEST_ONLY = True       # Only save when val Dice improves

# ── Device (auto-detected in train.py) ───────────────────────
# DEVICE = "cuda" | "mps" | "cpu"  – set automatically at runtime