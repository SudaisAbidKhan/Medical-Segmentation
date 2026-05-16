# =============================================================
#  model.py  –  U-Net using segmentation-models-pytorch
#  Accepts the same constructor arguments as inference.py uses
# =============================================================

import segmentation_models_pytorch as smp


class UNet(smp.Unet):
    def __init__(
        self,
        in_channels:  int   = 3,
        out_channels: int   = 1,   # mapped to smp's 'classes'
        features:     list  = None,  # ignored (smp handles internally)
        dropout:      float = 0.1,   # ignored (smp handles internally)
        encoder_name: str   = 'resnet34',
    ):
        super().__init__(
            encoder_name    = encoder_name,
            encoder_weights = None,   # weights come from .pth file
            in_channels     = in_channels,
            classes         = out_channels,
            activation      = None,   # raw logits; sigmoid applied in inference.py
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)