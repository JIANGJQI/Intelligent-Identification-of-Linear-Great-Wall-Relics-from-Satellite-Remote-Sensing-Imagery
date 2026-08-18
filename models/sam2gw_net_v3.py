"""
SAM2GWNet v3 — Pure Thick-DSConv bridge (no ASPP).
Replaces the 7-branch ASPP_GreatWall with a lightweight 5-branch SnakeBridge:
  - 1x pointwise (baseline)
  - 2x horizontal snake (fine d=1, coarse d=4)
  - 2x vertical snake   (fine d=1, coarse d=4)
Bridge params: ~1.1M (v2 ASPP was 7.0M, 6.4x reduction)
Total trainable: ~14M (v2 was 20.3M, 31% reduction)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
import torchvision

# ── Reuse all non-bridge components from v2 ──
from models.sam2gw_net import (
    DoubleConv, Up, Adapter, BasicConv2d, DRB_modified,
    CoordAtt, TileContextFusion, DenseDecoderBlock,
    LossNet, LossNetresnet50,
)
from utils.dsconv import ThickDSConv


# ==========================================
# ThickSkipBlock — x3 width-aware skip
# ==========================================
class ThickSkipBlock(nn.Module):
    """Replace CoordAtt at x3 with dual-morph Thick-DSConv.

    At H/16 (32×32), wall median width ≈ 1.4 fm px → thickness=3 is meaningful.
    morph=0 captures near-horizontal wall width, morph=1 covers near-vertical.
    Residual connection preserves original encoder features.
    """
    def __init__(self, in_ch=576, mid_ch=64):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(True),
        )
        self.s_h = ThickDSConv(mid_ch, mid_ch, kernel_size=5, morph=0, dilation=1, thickness=3)
        self.s_v = ThickDSConv(mid_ch, mid_ch, kernel_size=5, morph=1, dilation=1, thickness=3)
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch * 2, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(True),
        )

    def forward(self, x):
        r = self.reduce(x)
        f_h = self.s_h(r)
        f_v = self.s_v(r)
        gate = torch.sigmoid(self.fuse(torch.cat([f_h, f_v], dim=1)))
        return x * gate


# ==========================================
# SnakeBridge — Pure Thick-DSConv multi-scale fusion
# ==========================================
class SnakeBridge(nn.Module):
    """Lightweight bridge: 5 branches based on Thick-DSConv.

    4 Thick-DSConv branches (fine/coarse × horizontal/vertical) naturally
    cover multi-scale context + directional sampling, replacing ASPP's
    dilated convs + strip pooling + 45/135 rotation hacks.

    Args:
        in_channels: 1152 (from Hiera-L encoder x4 output)
        out_channels: 256 (d4 bridge output)
        mid_channels: inner channel for ThickDSConv (default 64)
    """
    def __init__(self, in_channels=1152, out_channels=256, mid_channels=96):
        super().__init__()

        # Shared 1x1 reduction: all branches operate at out_channels dim
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        # Pointwise baseline
        self.pt = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(True),
        )

        # Direction snakes
        self.s_h = self._snake(out_channels, mid_channels, out_channels, morph=0, dilation=1)
        self.s_v = self._snake(out_channels, mid_channels, out_channels, morph=1, dilation=1)

        # Multi-scale dilated convs
        self.d6 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(True),
        )
        self.d12 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(True),
        )

        # Project: 5 branches → out_channels
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(True),
        )

    @staticmethod
    def _snake(in_ch, mid_ch, out_ch, morph, dilation, thickness=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(True),
            ThickDSConv(mid_ch, mid_ch, kernel_size=5, morph=morph,
                        dilation=dilation, thickness=thickness),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )

    def forward(self, x):
        x = self.reduce(x)          # [B, 1152, H/32, W/32] → [B, 256, H/32, W/32]
        f_pt = self.pt(x)
        f_h = self.s_h(x)
        f_v = self.s_v(x)
        f_d6 = self.d6(x)
        f_d12 = self.d12(x)
        return self.project(torch.cat([f_pt, f_h, f_v, f_d6, f_d12], dim=1))


# ==========================================
# SAM2GWNet v3
# ==========================================
class SAM2GWNet_v3(nn.Module):
    def __init__(self, checkpoint_path=None, num_classes=1) -> None:
        super().__init__()
        model_cfg = "sam2_hiera_l.yaml"
        model = build_sam2(model_cfg, checkpoint_path) if checkpoint_path else build_sam2(model_cfg)

        self.encoder = model.image_encoder.trunk
        for param in self.encoder.parameters():
            param.requires_grad = False

        blocks = [Adapter(block) for block in self.encoder.blocks]
        self.encoder.blocks = nn.Sequential(*blocks)

        # ── v3: SnakeBridge replaces ASPP_GreatWall ──
        self.bridge = SnakeBridge(in_channels=1152, out_channels=256, mid_channels=96)

        self.thick_x3 = ThickSkipBlock(in_ch=576, mid_ch=64)  # width-aware snake skip
        self.ca2 = CoordAtt(288, 288)
        self.ca1 = CoordAtt(144, 144)

        self.dec3 = DenseDecoderBlock(256, 576, [288, 144], 256, use_strip=False)
        self.dec2 = DenseDecoderBlock(256, 288, [576, 144, 256], 192, use_strip=True)
        self.dec1 = DenseDecoderBlock(192, 144, [576, 288, 256], 96, use_strip=True)

        self.final_conv = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

        self.context_fusion = TileContextFusion(feature_dim=256, key_dim=32, pos_dim=32)

    def forward(self, x, neighbor_d4=None, neighbor_dirs=None, return_bridge_feat=False):
        h, w = x.size()[-2:]

        x1, x2, x3, x4 = self.encoder(x)
        x_red = self.bridge.reduce(x4)  # [B, 256, H/32, W/32], before snake branches
        d4 = self.bridge(x4)

        if neighbor_d4 is not None and len(neighbor_d4) > 0:
            d4 = self.context_fusion(d4, neighbor_d4, neighbor_dirs)

        f3 = self.thick_x3(x3)
        f2 = self.ca2(x2)
        f1 = self.ca1(x1)

        d3 = self.dec3(d4, f3, [f2, f1])
        d2 = self.dec2(d3, f2, [f3, f1, d4])
        d1 = self.dec1(d2, f1, [f3, f2, d4])

        out = self.final_conv(d1)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if return_bridge_feat:
            return out, x_red, x3
        return out


# ========== 测试 ==========
if __name__ == "__main__":
    print("=== v3 SnakeBridge 参数量 ===")
    bridge = SnakeBridge()
    bp = sum(p.numel() for p in bridge.parameters())
    print(f"SnakeBridge: {bp:,} params")

    print("\n=== v3 SAM2GWNet 推理测试 ===")
    model = SAM2GWNet_v3().cuda()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total:,}  |  Trainable: {trainable:,}")

    adapter = sum(p.numel() for p in model.encoder.blocks.parameters() if p.requires_grad)
    decoder = trainable - adapter
    print(f"Adapter: {adapter:,} ({adapter/trainable*100:.1f}%)")
    print(f"Decoder: {decoder:,} ({decoder/trainable*100:.1f}%)")

    with torch.no_grad():
        x = torch.randn(1, 3, 512, 512).cuda()
        out = model(x)
        print(f"Input: {x.shape}  →  Output: {out.shape}")
