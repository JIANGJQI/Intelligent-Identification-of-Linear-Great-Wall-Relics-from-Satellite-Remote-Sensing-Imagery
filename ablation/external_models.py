"""External model wrappers — use official implementations."""
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'GeoSeg'))


class UNetFormerWrapper(nn.Module):
    """Wrapper: GeoSeg UNetFormer returns (x, ah) in training; we need single tensor."""
    def __init__(self, num_classes=1):
        super().__init__()
        from geoseg.models.UNetFormer import UNetFormer as _UF
        self.model = _UF(num_classes=num_classes, pretrained=False)

    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, tuple) else out


def create_unetformer(num_classes=1):
    return UNetFormerWrapper(num_classes=num_classes)


def create_deeplabv3plus(num_classes=1):
    """smp DeepLabV3+: ResNet18 encoder, no pretrained weights."""
    import segmentation_models_pytorch as smp
    return smp.DeepLabV3Plus(encoder_name='resnet18', encoder_weights=None, classes=num_classes)
