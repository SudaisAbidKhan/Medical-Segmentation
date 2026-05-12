# =============================================================
#  model.py  –  U-Net architecture for binary segmentation
#
#  Paper: "U-Net: Convolutional Networks for Biomedical Image
#          Segmentation" – Ronneberger et al., MICCAI 2015
#
#  Architecture:
#    Encoder  → 4 × DoubleConv + MaxPool  (downsampling path)
#    Bottleneck → DoubleConv + Dropout
#    Decoder  → 4 × Upsample + DoubleConv (upsampling path)
#    Skip connections bridge encoder ↔ decoder at each level
#    Output   → 1×1 Conv → Sigmoid (binary mask probability map)
# =============================================================

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import config


# ── Building blocks ───────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    Two consecutive:  Conv2d → BatchNorm → ReLU
    This is the basic repeating unit in both encoder and decoder.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """DoubleConv followed by MaxPool2d — one step down."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)     # kept for skip connection
        down = self.pool(skip)
        return skip, down


class DecoderBlock(nn.Module):
    """
    Upsample (transpose conv) → concatenate skip → DoubleConv.
    ConvTranspose2d learns the upsampling weights end-to-end.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle odd spatial dimensions (e.g. 255 → 128 mismatch)
        if x.shape != skip.shape:
            x = TF.resize(x, size=skip.shape[2:])

        x = torch.cat([skip, x], dim=1)   # channel-wise concat
        return self.conv(x)


# ── U-Net ─────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Full U-Net for binary semantic segmentation.

    Args:
        in_channels  : number of input image channels (3 for RGB)
        out_channels : number of output mask channels (1 for binary)
        features     : list of filter counts at each encoder level
                       default [64, 128, 256, 512]
        dropout      : dropout probability in bottleneck
    """

    def __init__(
        self,
        in_channels : int  = config.IMAGE_CHANNELS,
        out_channels: int  = config.MASK_CHANNELS,
        features    : list = None,
        dropout     : float = config.DROPOUT,
    ):
        super().__init__()

        if features is None:
            features = config.FEATURES   # [64, 128, 256, 512]

        # ── Encoder ──────────────────────────────────────────
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(EncoderBlock(ch, f))
            ch = f

        # ── Bottleneck ────────────────────────────────────────
        bottleneck_ch = features[-1] * 2
        self.bottleneck = nn.Sequential(
            DoubleConv(features[-1], bottleneck_ch),
            nn.Dropout2d(p=dropout),
        )

        # ── Decoder ───────────────────────────────────────────
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.decoders.append(DecoderBlock(bottleneck_ch, f))
            bottleneck_ch = f

        # ── Output head ───────────────────────────────────────
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encode
        for encoder in self.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode (reverse skip order)
        for decoder, skip in zip(self.decoders,
                                  reversed(skip_connections)):
            x = decoder(x, skip)

        # Final 1×1 conv → raw logits (no sigmoid here;
        # BCEWithLogitsLoss handles it for numerical stability)
        return self.output_conv(x)

    # ── Utilities ─────────────────────────────────────────────

    def predict_mask(self, x: torch.Tensor,
                     threshold: float = config.MASK_THRESHOLD) -> torch.Tensor:
        """
        Run a forward pass and return a binary mask tensor.
        Applies sigmoid then thresholds.

        Args:
            x         : input image tensor [1, C, H, W]
            threshold : pixel is foreground if sigmoid(logit) > threshold
        Returns:
            Binary mask tensor [1, 1, H, W] with values 0 or 1.
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs  = torch.sigmoid(logits)
            return (probs > threshold).float()

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick architecture test ───────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet().to(device)

    # Dummy batch: 2 images, 3 channels, 256×256
    dummy = torch.randn(2, config.IMAGE_CHANNELS,
                        config.IMAGE_HEIGHT, config.IMAGE_WIDTH).to(device)
    out   = model(dummy)

    print(f"Input  shape : {dummy.shape}")   # [2, 3, 256, 256]
    print(f"Output shape : {out.shape}")     # [2, 1, 256, 256]
    print(f"Trainable params : {model.count_parameters():,}")