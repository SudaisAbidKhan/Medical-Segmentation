# =============================================================
#  utils.py  –  Shared utility functions
# =============================================================

import torch


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[utils] Using device: {device}')
    return device


def load_model(model, path: str, device: torch.device):
    """
    Load saved state dict into a model instance.

    Args:
        model  : instantiated UNet (weights not yet loaded)
        path   : path to the .pth file
        device : torch.device

    Returns:
        model in eval mode on the given device
    """
    state_dict = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f'[utils] Model loaded from {path}')
    return model
