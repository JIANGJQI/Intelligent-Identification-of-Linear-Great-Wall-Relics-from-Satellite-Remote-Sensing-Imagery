import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
import torchvision
    
class LossNet(torch.nn.Module):
    def __init__(self, resize=True):
        super(LossNet, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target

        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.mse_loss(x, y)
        return loss

class LossNetresnet50(torch.nn.Module):
    def __init__(self, resize=True):
        super(LossNetresnet50, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.blocks = torch.nn.ModuleList([
            resnet.conv1,
            resnet.bn1,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        ])
        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False  # 冻结所有参数

        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target):
        # 处理输入和目标图像
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        # 可能需要调整大小
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        loss = 0.0
        x = input
        y = target
        #all_features = []  # 保存每一层的特征

        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.mse_loss(x, y)

            # 保存特征图
            #all_features.append(x.detach().cpu())  # .detach() 防止梯度计算，.cpu() 方便在 CPU 中处理
        return loss
        #return loss, all_features  # 返回损失和特征图


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
       
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        return self.block(x + prompt)
    
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
class DRB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DRB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return x * a_w * a_h

# ==========================================
# 2. 跨切片上下文融合 (Neighborhood Context Fusion)
# ==========================================
class TileContextFusion(nn.Module):
    """轻量跨切片注意力：扩大感受野从 512×512 到 ~768×768（3×3 邻域）

    中心切片 bridge 特征做 Query，邻居 bridge 特征+方向编码做 Key/Value，
    逐通道门控调制。无邻居时退化为恒等映射。
    """
    def __init__(self, feature_dim=256, key_dim=32, pos_dim=32):
        super().__init__()
        self.W_q = nn.Linear(feature_dim, key_dim, bias=False)
        self.W_k = nn.Linear(feature_dim + pos_dim, key_dim, bias=False)
        self.W_v = nn.Linear(feature_dim + pos_dim, feature_dim, bias=False)

        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, feature_dim)
        )

        # 8 方向可学习位置编码: [top-left, top, top-right, left, right, bottom-left, bottom, bottom-right]
        self.pos_emb = nn.Parameter(torch.randn(8, pos_dim) * 0.02)
        self.scale = key_dim ** 0.5

    def forward(self, d4, neighbor_d4_list, neighbor_dirs):
        """
        Args:
            d4: 中心 tile bridge 特征 [B, C, H, W]
            neighbor_d4_list: 邻居 d4 列表，每个 [B, C, H, W]，长度 0..8
            neighbor_dirs: [B, N] 方向索引 (0-7)，-1 表示 padding
        Returns:
            d4_fused: [B, C, H, W]
        """
        B, C, H, W = d4.shape

        if len(neighbor_d4_list) == 0:
            return d4

        # 全局平均池化 → 空间描述向量
        center_vec = d4.mean(dim=[-2, -1])          # [B, C]
        Q = self.W_q(center_vec).unsqueeze(1)        # [B, 1, key_dim]

        # Stack + pool 邻居
        N = len(neighbor_d4_list)
        neighbor_stack = torch.stack(neighbor_d4_list, dim=0)  # [N, B, C, H, W]
        neighbor_stack = neighbor_stack.permute(1, 0, 2, 3, 4) # [B, N, C, H, W]
        neighbor_vecs = neighbor_stack.mean(dim=[-2, -1])      # [B, N, C]

        # 方向编码
        dirs_clamped = neighbor_dirs.clamp(min=0)
        pos_encoding = self.pos_emb[dirs_clamped]              # [B, N, pos_dim]
        pad_mask = (neighbor_dirs >= 0).float().unsqueeze(-1)  # [B, N, 1]
        pos_encoding = pos_encoding * pad_mask

        nei_with_pos = torch.cat([neighbor_vecs, pos_encoding], dim=-1)  # [B, N, C+pos_dim]

        K = self.W_k(nei_with_pos)   # [B, N, key_dim]
        V = self.W_v(nei_with_pos)   # [B, N, C]

        # Scaled dot-product attention
        attn_logits = (Q @ K.transpose(-1, -2)) / self.scale   # [B, 1, N]
        attn_logits = attn_logits.masked_fill(
            (neighbor_dirs < 0).unsqueeze(1), float('-inf')
        )
        attn_weights = F.softmax(attn_logits, dim=-1)          # [B, 1, N]
        fused = (attn_weights @ V).squeeze(1)                   # [B, C]

        gate = torch.sigmoid(self.gate_mlp(fused))              # [B, C]
        gate = gate.view(B, C, 1, 1)
        return d4 * gate + d4


# ==========================================
# 3. 改进型 ASPP (集成条带池化)
# ==========================================
class ASPP_GreatWall(nn.Module):
    def __init__(self, in_channels, out_channels, dsconv_mid=64):
        super(ASPP_GreatWall, self).__init__()

        # Standard ASPP branches
        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
                                nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
                                nn.BatchNorm2d(out_channels), nn.ReLU(True))

        # 0°/90° Strip pooling (keep)
        self.pool_0 = nn.Sequential(nn.AdaptiveAvgPool2d((None, 1)),
                                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                    nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.pool_90 = nn.Sequential(nn.AdaptiveAvgPool2d((1, None)),
                                     nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels), nn.ReLU(True))

        # Thick-DSConv: replaces 45°/135° rotation hack
        # morph=0: snake along x (covers near-horizontal walls)
        # morph=1: snake along y (covers near-vertical walls)
        from utils.dsconv import ThickDSConv
        self.dsconv_h = nn.Sequential(
            nn.Conv2d(in_channels, dsconv_mid, 1, bias=False),
            nn.BatchNorm2d(dsconv_mid), nn.ReLU(True),
            ThickDSConv(dsconv_mid, dsconv_mid, kernel_size=5, morph=0, dilation=1, thickness=3),
            nn.Conv2d(dsconv_mid, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(True),
        )
        self.dsconv_v = nn.Sequential(
            nn.Conv2d(in_channels, dsconv_mid, 1, bias=False),
            nn.BatchNorm2d(dsconv_mid), nn.ReLU(True),
            ThickDSConv(dsconv_mid, dsconv_mid, kernel_size=5, morph=1, dilation=1, thickness=3),
            nn.Conv2d(dsconv_mid, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(True),
        )

        # 7 branches: b0, b1, b2, pool_0, pool_90, dsconv_h, dsconv_v
        self.project = nn.Sequential(nn.Conv2d(out_channels * 7, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels), nn.ReLU(True))

    def forward(self, x):
        size = x.shape[-2:]

        f0 = self.b0(x)
        f1 = self.b1(x)
        f2 = self.b2(x)

        # H + W strip pooling
        fh = F.interpolate(self.pool_0(x), size=size, mode='bilinear', align_corners=True)
        fw = F.interpolate(self.pool_90(x), size=size, mode='bilinear', align_corners=True)

        # Thick-DSConv: adaptive directional snake sampling (replaces 45°/135° rotation)
        fd_h = self.dsconv_h(x)
        fd_v = self.dsconv_v(x)

        return self.project(torch.cat([f0, f1, f2, fh, fw, fd_h, fd_v], dim=1))

# ==========================================
# 4. 密集解码块 (Dense Decoder)
# ==========================================
class DenseDecoderBlock(nn.Module):
    def __init__(self, prev_channels, skip_channels, extras_channels, out_channels,
                 adapt_channels=32, use_strip=True):
        super().__init__()
        total_in = prev_channels + skip_channels + len(extras_channels) * adapt_channels
        self.extras_adapt = nn.ModuleList([
            nn.Conv2d(ch, adapt_channels, 1, bias=False) for ch in extras_channels
        ])
        self.conv = nn.Sequential(
            nn.Conv2d(total_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # H+V 条带池化: 扩大感受野到整行/整列
        self.use_strip = use_strip
        if use_strip:
            self.strip_h = nn.Conv2d(total_in, out_channels // 4, 1, bias=False)
            self.strip_w = nn.Conv2d(total_in, out_channels // 4, 1, bias=False)
            self.strip_fuse = nn.Conv2d(out_channels + out_channels // 2, out_channels, 1, bias=False)

    def forward(self, x, skip, extras):
        target_size = skip.shape[-2:]
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        resized = [adapt(F.interpolate(f, size=target_size, mode='bilinear', align_corners=True))
                   for f, adapt in zip(extras, self.extras_adapt)]
        feat = torch.cat([x, skip] + resized, dim=1)
        conv_out = self.conv(feat)

        if self.use_strip:
            h, w = target_size
            # 水平条带: 沿宽度方向压缩, "这一行有墙吗"
            sh = F.adaptive_avg_pool2d(feat, (None, 1))
            sh = self.strip_h(sh)
            sh = F.interpolate(sh, size=(h, w), mode='bilinear', align_corners=True)
            # 垂直条带: 沿高度方向压缩, "这一列有墙吗"
            sw = F.adaptive_avg_pool2d(feat, (1, None))
            sw = self.strip_w(sw)
            sw = F.interpolate(sw, size=(h, w), mode='bilinear', align_corners=True)
            return self.strip_fuse(torch.cat([conv_out, sh, sw], dim=1))

        return conv_out

# ==========================================
# 5. SAM2GWNet (长城提取网络)
# ==========================================
class SAM2GWNet(nn.Module):
    def __init__(self, checkpoint_path=None, num_classes=1) -> None:
        super(SAM2GWNet, self).__init__()
        model_cfg = "sam2_hiera_l.yaml"
        model = build_sam2(model_cfg, checkpoint_path) if checkpoint_path else build_sam2(model_cfg)

        self.encoder = model.image_encoder.trunk
        for param in self.encoder.parameters():
            param.requires_grad = False

        blocks = [Adapter(block) for block in self.encoder.blocks]
        self.encoder.blocks = nn.Sequential(*blocks)

        self.bridge = ASPP_GreatWall(1152, 256)

        self.ca3 = CoordAtt(576, 576)
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

    def forward(self, x, neighbor_d4=None, neighbor_dirs=None):
        h, w = x.size()[-2:]

        x1, x2, x3, x4 = self.encoder(x)

        d4 = self.bridge(x4)

        if neighbor_d4 is not None and len(neighbor_d4) > 0:
            d4 = self.context_fusion(d4, neighbor_d4, neighbor_dirs)

        f3 = self.ca3(x3)
        f2 = self.ca2(x2)
        f1 = self.ca1(x1)

        d3 = self.dec3(d4, f3, [f2, f1])
        d2 = self.dec2(d3, f2, [f3, f1, d4])
        d1 = self.dec1(d2, f1, [f3, f2, d4])

        out = self.final_conv(d1)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        return out


# ========== 测试代码 ==========
if __name__ == "__main__":
    from utils.dsconv import ThickDSConv
    with torch.no_grad():
        model = SAM2GWNet().cuda()
        x = torch.randn(4, 3, 1024, 1024).cuda()
        out = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")