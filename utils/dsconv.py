"""Thick Dynamic Snake Convolution (DSConv with thickness + dilation).
Core operators for ASPP_GreatWall directional sampling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_coordinate_map_2d(offset, morph, kernel_size, dilation=1, device=None,
                           thickness=1, width_param=None):
    """Build deformed coordinate grid for DSConv / Thick-DSConv.

    thickness=1: standard 1xK snake along the free axis
    thickness=3: 3 parallel snake rows (centerline +/- width edges)

    Args:
        offset: [B, 2K, H, W] predicted centerline offsets (dx,dy interleaved)
        morph: 0=snake along x (perpendicular offsets in y)
               1=snake along y (perpendicular offsets in x)
        kernel_size: K, sampling points along snake
        dilation: base grid spacing (1=standard, >1=dilated)
        device: torch device
        thickness: number of parallel rows (odd, >=1)
        width_param: [B, T//2, H, W] per-position width for thick rows

    Returns:
        coords: [B, K*T, H, W, 2] sampling coords in [-1,1] range
    """
    B, _, H, W = offset.shape
    K = kernel_size
    d = dilation
    center = K // 2
    T = thickness
    half_T = T // 2

    y_range = torch.arange(-center * d, (center + 1) * d, d,
                          device=device, dtype=torch.float32)
    x_range = torch.arange(-center * d, (center + 1) * d, d,
                          device=device, dtype=torch.float32)

    if morph == 0:
        y_spread = torch.zeros(K, device=device)
        x_spread = x_range
        y_offset = offset[:, 1::2, :, :]
        y_offset_new = torch.zeros_like(y_offset)
        y_offset_new[:, center, :, :] = 0
        for i in range(1, center + 1):
            idx_r, idx_l = center + i, center - i
            y_offset_new[:, idx_r] = y_offset_new[:, idx_r - 1] + torch.tanh(y_offset[:, idx_r])
            y_offset_new[:, idx_l] = y_offset_new[:, idx_l + 1] + torch.tanh(y_offset[:, idx_l])
        y_center = y_spread.view(1, K, 1, 1) + y_offset_new
        x_center = x_spread.view(1, K, 1, 1).expand(B, -1, H, W)
        all_y, all_x = [y_center], [x_center]
        if T > 1:
            w = width_param if width_param is not None else \
                torch.ones(B, half_T, H, W, device=device)
            for side in range(1, half_T + 1):
                dy = side * w[:, side - 1:side]
                all_y.insert(0, y_center - dy)
                all_y.append(y_center + dy)
                all_x.insert(0, x_center)
                all_x.append(x_center)
    else:
        y_spread = y_range
        x_spread = torch.zeros(K, device=device)
        x_offset = offset[:, 0::2, :, :]
        x_offset_new = torch.zeros_like(x_offset)
        x_offset_new[:, center, :, :] = 0
        for i in range(1, center + 1):
            idx_r, idx_l = center + i, center - i
            x_offset_new[:, idx_r] = x_offset_new[:, idx_r - 1] + torch.tanh(x_offset[:, idx_r])
            x_offset_new[:, idx_l] = x_offset_new[:, idx_l + 1] + torch.tanh(x_offset[:, idx_l])
        y_center = y_spread.view(1, K, 1, 1).expand(B, -1, H, W)
        x_center = x_spread.view(1, K, 1, 1) + x_offset_new
        all_y, all_x = [y_center], [x_center]
        if T > 1:
            w = width_param if width_param is not None else \
                torch.ones(B, half_T, H, W, device=device)
            for side in range(1, half_T + 1):
                dx = side * w[:, side - 1:side]
                all_x.insert(0, x_center - dx)
                all_x.append(x_center + dx)
                all_y.insert(0, y_center)
                all_y.append(y_center)

    y_all = torch.cat(all_y, dim=1)
    x_all = torch.cat(all_x, dim=1)
    y_all = y_all / (H - 1) * 2.0
    x_all = x_all / (W - 1) * 2.0

    # Add per-position base coordinate (BUGFIX: snake was always centered at (0,0))
    base_y = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1)
    base_x = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W)
    x_all = x_all + base_x
    y_all = y_all + base_y

    return torch.stack([x_all, y_all], dim=-1)


class ThickDSConv(nn.Module):
    """Thick Dynamic Snake Convolution.

    Standard DSConv (thickness=1): 1xK snake along the structure centerline.
    Thick mode (thickness>=3): multiple parallel snake rows forming a strip,
    with learnable width between rows.

    Args:
        in_ch, out_ch: channel dimensions
        kernel_size: K, sampling points along the snake
        morph: 0=snake along x (for near-horizontal structures)
               1=snake along y (for near-vertical structures)
        dilation: base grid spacing
        thickness: parallel rows (odd, 1=standard DSConv)
    """
    def __init__(self, in_ch, out_ch, kernel_size=5, morph=0, dilation=1, thickness=3):
        super().__init__()
        assert kernel_size % 2 == 1 and thickness % 2 == 1
        self.kernel_size = kernel_size
        self.morph = morph
        self.dilation = dilation
        self.thickness = thickness
        self.total_points = kernel_size * thickness
        half_T = thickness // 2

        # Offset prediction for centerline snake
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

        # Width prediction for thick rows
        self.learnable_width = thickness > 1
        if self.learnable_width:
            self.width_conv = nn.Conv2d(in_ch, half_T, 3, padding=1, bias=True)
            nn.init.constant_(self.width_conv.weight, 0.0)
            nn.init.constant_(self.width_conv.bias, 0.0)  # softplus(0)=0.693 fm px baseline
            self.width_scale = nn.Parameter(torch.tensor(0.5))  # → 0.347 fm px ≈ 11 img px per side ≈ median wall width

        # Conv weight: applied to sampled features
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, self.total_points, 1))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        B, C, H, W = x.shape
        K, TP = self.kernel_size, self.total_points

        offset = self.offset_conv(x)
        width_param = None
        if self.learnable_width:
            width_param = self.width_scale * F.softplus(self.width_conv(x))

        grid = get_coordinate_map_2d(offset, self.morph, K, self.dilation,
                                     x.device, self.thickness, width_param)
        # grid: [B, TP, H, W, 2] — coords normalized for original input H×W

        # Sample each kernel point independently (avoiding tiling bugs)
        samples = []
        for p in range(TP):
            g = grid[:, p, :, :, :]  # [B, H, W, 2]
            s = F.grid_sample(x, g, mode='bilinear',
                              padding_mode='border', align_corners=True)  # [B, C, H, W]
            samples.append(s.unsqueeze(2))
        sampled = torch.cat(samples, dim=2)  # [B, C, TP, H, W]
        sampled = sampled.permute(0, 1, 3, 4, 2)  # [B, C, H, W, TP]

        out = torch.einsum('oip,bihwp->bohw', self.weight.squeeze(-1), sampled)
        return out
