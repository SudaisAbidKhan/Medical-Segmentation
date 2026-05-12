# =============================================================
#  dataset.py  –  Custom PyTorch Dataset for LGG MRI Segmentation
#
#  Kaggle dataset: mateuszbuda/lgg-mri-segmentation
#  Structure expected:
#    data/images/  →  *.tif  (3-channel MRI slices)
#    data/masks/   →  *_mask.tif  (binary segmentation masks)
# =============================================================

import os
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


# ── Augmentation pipelines ────────────────────────────────────

def get_train_transforms():
    """Strong augmentations for training to prevent overfitting."""
    return A.Compose([
        A.Resize(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                           rotate_limit=30, p=0.5),
        A.ElasticTransform(p=0.3),
        A.GridDistortion(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.4),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),   # ImageNet stats work well
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Minimal transforms for validation / test (no random augmentation)."""
    return A.Compose([
        A.Resize(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ── Dataset class ─────────────────────────────────────────────

class BrainMRIDataset(Dataset):
    """
    Loads paired (image, mask) .tif files from the LGG Kaggle dataset.

    The Kaggle dataset is organised per-patient:
        kaggle_3m/
            TCGA_CS_4941_19960909/
                TCGA_CS_4941_19960909_1.tif        ← MRI slice
                TCGA_CS_4941_19960909_1_mask.tif   ← binary mask
                ...

    We flatten all patient folders into two lists: images & masks.
    """

    def __init__(self, image_paths: list, mask_paths: list,
                 transform=None):
        """
        Args:
            image_paths : list of absolute paths to MRI .tif images
            mask_paths  : list of absolute paths to mask .tif images
            transform   : albumentations Compose pipeline
        """
        assert len(image_paths) == len(mask_paths), (
            "Number of images and masks must match!"
        )
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        """Open a .tif file and return a uint8 RGB numpy array (H, W, 3)."""
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.uint8)

    @staticmethod
    def _load_mask(path: str) -> np.ndarray:
        """Open a mask .tif and return a float32 array (H, W) in {0, 1}."""
        mask = Image.open(path).convert("L")        # grayscale
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 0).astype(np.float32)        # binarise
        return mask

    # ── Dataset interface ────────────────────────────────────

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = self._load_image(self.image_paths[idx])
        mask  = self._load_mask(self.mask_paths[idx])

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]              # Tensor [3, H, W] float32
            mask  = augmented["mask"]               # Tensor [H, W]    float32

        # Add channel dim to mask → [1, H, W]
        mask = mask.unsqueeze(0)

        return image, mask


# ── Dataset builder ───────────────────────────────────────────

def build_datasets(kaggle_root: str = None):
    """
    Scans the dataset directory, pairs images with masks, then
    splits into train / val / test PyTorch Datasets.

    Args:
        kaggle_root : path to the 'kaggle_3m' folder inside config.DATA_DIR.
                      If None, falls back to config.IMAGE_DIR / config.MASK_DIR
                      (flat layout).
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    if kaggle_root and os.path.isdir(kaggle_root):
        # ── Kaggle nested layout ──────────────────────────────
        all_images = sorted(glob.glob(
            os.path.join(kaggle_root, "**", "*.tif"), recursive=True
        ))
        # Keep only non-mask files, then derive mask path
        image_paths = [p for p in all_images if "_mask" not in p]
        mask_paths  = [p.replace(".tif", "_mask.tif") for p in image_paths]

        # Remove pairs where mask file is missing
        valid_pairs = [
            (img, msk) for img, msk in zip(image_paths, mask_paths)
            if os.path.exists(msk)
        ]
        image_paths, mask_paths = zip(*valid_pairs)
        image_paths = list(image_paths)
        mask_paths  = list(mask_paths)

    else:
        # ── Flat layout (data/images/ & data/masks/) ──────────
        image_paths = sorted(glob.glob(
            os.path.join(config.IMAGE_DIR, "*.tif")
        ))
        mask_paths = sorted(glob.glob(
            os.path.join(config.MASK_DIR, "*.tif")
        ))

    total = len(image_paths)
    assert total > 0, (
        f"No images found! Check your data directory: {config.DATA_DIR}"
    )
    print(f"[Dataset] Found {total} image-mask pairs.")

    # ── Deterministic shuffle then split ─────────────────────
    rng = np.random.default_rng(config.RANDOM_SEED)
    indices = rng.permutation(total).tolist()

    n_train = int(total * config.TRAIN_SPLIT)
    n_val   = int(total * config.VAL_SPLIT)
    # test gets the remainder

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val :]

    def _subset(idx_list, transform):
        imgs  = [image_paths[i] for i in idx_list]
        masks = [mask_paths[i]  for i in idx_list]
        return BrainMRIDataset(imgs, masks, transform=transform)

    train_ds = _subset(train_idx, get_train_transforms())
    val_ds   = _subset(val_idx,   get_val_transforms())
    test_ds  = _subset(test_idx,  get_val_transforms())

    print(f"[Dataset] Split → train: {len(train_ds)} | "
          f"val: {len(val_ds)} | test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


def get_dataloaders(kaggle_root: str = None):
    """
    Convenience wrapper: returns (train_loader, val_loader, test_loader).
    """
    train_ds, val_ds, test_ds = build_datasets(kaggle_root)

    train_loader = DataLoader(
        train_ds,
        batch_size  = config.BATCH_SIZE,
        shuffle     = True,
        num_workers = config.NUM_WORKERS,
        pin_memory  = config.PIN_MEMORY,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = config.NUM_WORKERS,
        pin_memory  = config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = 1,        # one at a time for evaluation
        shuffle     = False,
        num_workers = config.NUM_WORKERS,
        pin_memory  = config.PIN_MEMORY,
    )
    return train_loader, val_loader, test_loader


# ── Quick sanity check ────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Point this at your unzipped Kaggle folder
    KAGGLE_ROOT = os.path.join(config.DATA_DIR, "kaggle_3m")
    train_loader, val_loader, test_loader = get_dataloaders(KAGGLE_ROOT)

    images, masks = next(iter(train_loader))
    print(f"Image batch shape : {images.shape}")   # [B, 3, 256, 256]
    print(f"Mask  batch shape : {masks.shape}")    # [B, 1, 256, 256]
    print(f"Image dtype / range: {images.dtype} "
          f"[{images.min():.2f}, {images.max():.2f}]")
    print(f"Mask  dtype / range: {masks.dtype}  "
          f"[{masks.min():.2f}, {masks.max():.2f}]")

    # Visualise first sample
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    img_np = images[0].permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # rescale
    axes[0].imshow(img_np)
    axes[0].set_title("MRI Slice")
    axes[1].imshow(masks[0, 0].numpy(), cmap="gray")
    axes[1].set_title("Segmentation Mask")
    plt.tight_layout()
    plt.savefig("sample_batch.png")
    print("Saved sample_batch.png")