"""
Generate a placeholder U-Net model (untrained weights).
Useful for testing the backend/frontend before training.
"""

import os
import torch
import config
from model import UNet

os.makedirs(config.SAVED_MODEL_DIR, exist_ok=True)

model = UNet(
    in_channels=config.IMAGE_CHANNELS,
    out_channels=config.MASK_CHANNELS,
    features=config.FEATURES,
    dropout=config.DROPOUT,
)

torch.save(model.state_dict(), config.MODEL_PATH)
print(f"✓ Placeholder model saved to {config.MODEL_PATH}")
