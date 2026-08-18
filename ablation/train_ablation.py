"""Ablation: incremental contribution of Thick-DSConv components.

Variant       Bridge             x3 Skip        SnakeLoss
baseline      ASPP-like (no蛇)    CoordAtt       无
plus_x4       SnakeBridge         CoordAtt       无
plus_x3       ASPP-like           ThickSkipBlock 无
full          SnakeBridge         ThickSkipBlock 有 (w=0.3)
"""
import os, sys, argparse, numpy as np, torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.dataset import FullDataset
from core.logger import create_logger
from utils.cldice_loss import soft_cl_dice
from utils.dsconv import ThickDSConv, get_coordinate_map_2d
from models.sam2gw_net import CoordAtt, DenseDecoderBlock, Adapter, DoubleConv, Up
from sam2.build_sam import build_sam2
import ablation.config_ablation as config

torch.cuda.empty_cache()

use_fp16 = config.TRAIN_CONFIG['use_fp16']
DEVICE = torch.device("cuda")


# ── Shared encoder ──
def build_encoder(checkpoint_path):
    model_cfg = "sam2_hiera_l.yaml"
    m = build_sam2(model_cfg, checkpoint_path) if checkpoint_path else build_sam2(model_cfg)
    encoder = m.image_encoder.trunk
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.blocks = nn.Sequential(*[Adapter(b) for b in encoder.blocks])
    return encoder


# ── ASPP-like bridge (no snakes) ──
class ASPPBridge(nn.Module):
    def __init__(self, in_ch=1152, out_ch=256):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.b0 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 1, bias=False),
                                nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.b1 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=3, dilation=3, bias=False),
                                nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.pool_h = nn.Sequential(nn.AdaptiveAvgPool2d((None, 1)),
                                    nn.Conv2d(out_ch, out_ch, 1, bias=False),
                                    nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.pool_w = nn.Sequential(nn.AdaptiveAvgPool2d((1, None)),
                                    nn.Conv2d(out_ch, out_ch, 1, bias=False),
                                    nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(out_ch * 4, out_ch, 1, bias=False),
                                     nn.BatchNorm2d(out_ch), nn.ReLU(True))

    def forward(self, x):
        x = self.reduce(x)
        sz = x.shape[-2:]
        f0 = self.b0(x)
        f1 = self.b1(x)
        fh = F.interpolate(self.pool_h(x), size=sz, mode='bilinear', align_corners=True)
        fw = F.interpolate(self.pool_w(x), size=sz, mode='bilinear', align_corners=True)
        return self.project(torch.cat([f0, f1, fh, fw], dim=1))


# ── Models ──
class DeepLabV3PlusASPP(nn.Module):
    """Standard DeepLabV3+ ASPP: 5 branches at bridge."""
    def __init__(self, in_ch=1152, out_ch=256):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.b0 = nn.Conv2d(out_ch, out_ch, 1)
        self.b6 = nn.Conv2d(out_ch, out_ch, 3, padding=6, dilation=6)
        self.b12 = nn.Conv2d(out_ch, out_ch, 3, padding=12, dilation=12)
        self.b18 = nn.Conv2d(out_ch, out_ch, 3, padding=18, dilation=18)
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                         nn.Conv2d(out_ch, out_ch, 1),
                                         nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
                                     nn.BatchNorm2d(out_ch), nn.ReLU(True))
        # BN for each
        self.bn0 = nn.BatchNorm2d(out_ch); self.bn6 = nn.BatchNorm2d(out_ch)
        self.bn12 = nn.BatchNorm2d(out_ch); self.bn18 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.reduce(x); sz = x.shape[-2:]
        f0 = F.relu(self.bn0(self.b0(x)))
        f6 = F.relu(self.bn6(self.b6(x)))
        f12 = F.relu(self.bn12(self.b12(x)))
        f18 = F.relu(self.bn18(self.b18(x)))
        fg = F.interpolate(self.global_pool(x), size=sz, mode='bilinear', align_corners=True)
        return self.project(torch.cat([f0, f6, f12, f18, fg], dim=1))


class LiteDecoder(nn.Module):
    """DeepLabV3+-style simple decoder: 1 low-level skip + 2 convs."""
    def __init__(self, low_ch=288, in_ch=256, out_ch=256):
        super().__init__()
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_ch, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch + 48, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, 1, 1))

    def forward(self, d4, x2, orig_size):
        low = self.low_conv(x2)
        d4_up = F.interpolate(d4, size=low.shape[2:], mode='bilinear', align_corners=True)
        out = self.decoder(torch.cat([d4_up, low], dim=1))
        return F.interpolate(out, size=orig_size, mode='bilinear', align_corners=True)


class UNetDecoder(nn.Module):
    """Simple U-Net decoder with channel-matched skips."""
    def __init__(self):
        super().__init__()
        self.skip3 = nn.Conv2d(576, 256, 1)
        self.skip2 = nn.Conv2d(288, 128, 1)
        self.skip1 = nn.Conv2d(144, 64, 1)
        # d4(256)↑+skip3(256)=512 → 256
        self.up3 = DoubleConv(512, 256)
        # d3(256)↑+skip2(128)=384 → 128
        self.up2 = DoubleConv(384, 128)
        # d2(128)↑+skip1(64)=192 → 64
        self.up1 = DoubleConv(192, 64)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 1, 1))

    def forward(self, d4, x3, x2, x1, orig_size):
        d3 = self.up3(torch.cat([F.interpolate(d4, size=x3.shape[2:], mode='bilinear', align_corners=True),
                                 self.skip3(x3)], dim=1))
        d2 = self.up2(torch.cat([F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=True),
                                 self.skip2(x2)], dim=1))
        d1 = self.up1(torch.cat([F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=True),
                                 self.skip1(x1)], dim=1))
        out = self.final(d1)
        return F.interpolate(out, size=orig_size, mode='bilinear', align_corners=True)


class AblationModel(nn.Module):
    def __init__(self, hiers_path, variant):
        super().__init__()
        self.variant = variant

        is_former = variant == "unetformer"
        is_dlv3 = variant == "deeplabv3plus"
        if is_former:
            from ablation.external_models import create_unetformer
            self.model = create_unetformer()
            return
        if is_dlv3:
            from ablation.external_models import create_deeplabv3plus
            self.model = create_deeplabv3plus()
            return

        self.encoder = build_encoder(hiers_path)
        use_snake = variant in ("plus_x4", "full")
        use_thick = variant in ("plus_x3", "full", "plus_x3_dsconv")
        is_unet = variant == "unet"
        is_dlv3p = variant == "deeplabv3plus"
        is_lite = variant in ("x4_lite", "baseline_lite", "x3_lite", "full_lite")

        if variant in ("x4_lite", "full_lite"):
            from models.sam2gw_net_v3 import SnakeBridge
            self.bridge = SnakeBridge(1152, 256, 96)
            if variant == "full_lite":
                from models.sam2gw_net_v3 import ThickSkipBlock
                self.skip_x3 = ThickSkipBlock(576, 64)
        elif variant == "x3_lite":
            self.bridge = ASPPBridge(1152, 256)
            from models.sam2gw_net_v3 import ThickSkipBlock
            self.skip_x3 = ThickSkipBlock(576, 64)
        elif variant == "baseline_lite":
            self.bridge = ASPPBridge(1152, 256)
        elif is_unet:
            self.bridge = nn.Sequential(
                nn.Conv2d(1152, 256, 1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(True))
        elif is_dlv3p:
            self.bridge = DeepLabV3PlusASPP(1152, 256)
        elif use_snake:
            from models.sam2gw_net_v3 import SnakeBridge
            self.bridge = SnakeBridge(1152, 256, 96)
        else:
            self.bridge = ASPPBridge(1152, 256)

        self.is_simple = is_unet or is_dlv3p or is_lite
        if is_lite:
            low_ch = 576 if variant in ("x3_lite", "full_lite") else 288
            self.decoder = LiteDecoder(low_ch, 256, 256)
        elif is_unet:
            self.decoder = UNetDecoder()
        elif is_dlv3p:
            # DeepLabV3+: simple decoder with 1 low-level skip (x2)
            self.dlv3p_low = nn.Sequential(
                nn.Conv2d(288, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(True))
            self.dlv3p_dec = nn.Sequential(
                nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(True),
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(True),
                nn.Conv2d(256, 1, 1))
        else:
            if use_thick:
                from models.sam2gw_net_v3 import ThickSkipBlock
                self.skip_x3 = ThickSkipBlock(576, 64)
                if variant == "plus_x3_dsconv":
                    from utils.dsconv import ThickDSConv
                    self.skip_x3.s_h = ThickDSConv(64, 64, kernel_size=5, morph=0, dilation=1, thickness=1)
                    self.skip_x3.s_v = ThickDSConv(64, 64, kernel_size=5, morph=1, dilation=1, thickness=1)
            else:
                self.skip_x3 = CoordAtt(576, 576)
            self.ca2 = CoordAtt(288, 288)
            self.ca1 = CoordAtt(144, 144)
            self.dec3 = DenseDecoderBlock(256, 576, [288, 144], 256, use_strip=False)
            self.dec2 = DenseDecoderBlock(256, 288, [576, 144, 256], 192, use_strip=True)
            self.dec1 = DenseDecoderBlock(192, 144, [576, 288, 256], 96, use_strip=True)
            self.final_conv = nn.Sequential(
                nn.Conv2d(96, 32, 3, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(True),
                nn.Conv2d(32, 1, 1))

    def forward(self, x, return_aux=False):
        if hasattr(self, 'model'):
            out = self.model(x)
            return (out, x, x) if return_aux else out  # dummy aux for compat
        h, w = x.shape[-2:]
        x1, x2, x3, x4 = self.encoder(x)

        d4 = self.bridge(x4)
        x_red = self.bridge.reduce(x4) if hasattr(self.bridge, 'reduce') else x4

        if self.is_simple:
            if self.variant == "unet":
                out = self.decoder(d4, x3, x2, x1, (h, w))
            elif self.variant == "x3_lite":
                f_x3 = self.skip_x3(x3)
                out = self.decoder(d4, f_x3, (h, w))
            elif self.variant == "full_lite":
                f_x3 = self.skip_x3(x3)
                out = self.decoder(d4, f_x3, (h, w))
                if return_aux:
                    return out, x_red, x3
                return out
            elif self.variant in ("x4_lite", "baseline_lite"):
                out = self.decoder(d4, x2, (h, w))
            else:  # deeplabv3plus
                low_feat = self.dlv3p_low(x2)
                d4_up = F.interpolate(d4, size=low_feat.shape[2:], mode='bilinear', align_corners=True)
                fused = torch.cat([d4_up, low_feat], dim=1)
                out = self.dlv3p_dec(fused)
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            if return_aux:
                return out, x_red, x3
            return out

        f3 = self.skip_x3(x3)
        f2 = self.ca2(x2)
        f1 = self.ca1(x1)
        d3 = self.dec3(d4, f3, [f2, f1])
        d2 = self.dec2(d3, f2, [f3, f1, d4])
        d1 = self.dec1(d2, f1, [f3, f2, d4])
        out = self.final_conv(d1)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        if return_aux:
            return out, x_red, x3
        return out


# ── Loss helpers ──
def dice_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    return (1 - (2 * inter + 1e-6) / (union + 1e-6)).mean()


def snake_align_loss(model, x_red, x3, target):
    total, n = 0.0, 0
    with torch.no_grad():
        gt_fm = F.adaptive_avg_pool2d(target.float(), x_red.shape[2:])
        wm = (gt_fm > 0.3).float()
    if wm.sum() >= 1 and hasattr(model.bridge, 's_h'):
        for seq in [model.bridge.s_h, model.bridge.s_v]:
            dsconv = seq[3]
            x_mid = seq[2](seq[1](seq[0](x_red)))
            o = dsconv.offset_conv(x_mid)
            wp = dsconv.width_scale * F.softplus(dsconv.width_conv(x_mid)) if dsconv.thickness > 1 else None
            g = get_coordinate_map_2d(o, dsconv.morph, dsconv.kernel_size, dsconv.dilation, DEVICE,
                                      dsconv.thickness, wp)
            K = dsconv.kernel_size; c0 = (dsconv.thickness // 2) * K
            for k in range(K):
                gt_s = F.grid_sample(gt_fm, g[:, c0 + k], mode='bilinear', padding_mode='border',
                                     align_corners=True)
                total += (wm * (1.0 - gt_s)).sum() / wm.sum().clamp(min=1); n += 1
    if hasattr(model, 'skip_x3') and hasattr(model.skip_x3, 's_h'):
        with torch.no_grad():
            gt_fm3 = F.adaptive_avg_pool2d(target.float(), x3.shape[2:])
            wm3 = (gt_fm3 > 0.3).float()
        if wm3.sum() >= 1:
            x3_mid = model.skip_x3.reduce(x3)
            for dsconv in [model.skip_x3.s_h, model.skip_x3.s_v]:
                o = dsconv.offset_conv(x3_mid)
                wp = dsconv.width_scale * F.softplus(dsconv.width_conv(x3_mid))
                g = get_coordinate_map_2d(o, dsconv.morph, dsconv.kernel_size, dsconv.dilation, DEVICE,
                                          dsconv.thickness, wp)
                K = dsconv.kernel_size; c0 = (dsconv.thickness // 2) * K
                for k in range(K):
                    gt_s = F.grid_sample(gt_fm3, g[:, c0 + k], mode='bilinear', padding_mode='border',
                                         align_corners=True)
                    total += (wm3 * (1.0 - gt_s)).sum() / wm3.sum().clamp(min=1); n += 1
    return torch.tensor(total / max(n, 1), device=DEVICE)


# ── Main ──
def main():
    parser = argparse.ArgumentParser("Ablation")
    parser.add_argument("--variant", type=str, required=True,
                        choices=["deeplabv3plus", "unetformer", "baseline", "plus_x3", "plus_x3_dsconv", "plus_x4", "full"])
    parser.add_argument("--epoch", type=int, default=config.TRAIN_CONFIG['epoch'])
    parser.add_argument("--resume", type=str, default=None, help="resume from checkpoint path")
    args = parser.parse_args()
    variant = args.variant
    scaler = amp.GradScaler(enabled=use_fp16, growth_interval=999999)
    ckpt_dir = os.path.join(config.CKPT_DIR, variant)
    log_dir = os.path.join(config.LOG_DIR, variant)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*50}\nABLATION: {variant}\n{'='*50}")

    dataset = FullDataset([config.TRAIN_IMAGE_DIR], [config.TRAIN_GT_DIR],
                          config.TRAIN_CONFIG['crop_size'], mode='train')
    dataloader = DataLoader(dataset, batch_size=config.TRAIN_CONFIG['batch_size'],
                            shuffle=True, num_workers=0, drop_last=True)
    print(f"Train: {len(dataset)} samples")

    # Validation set
    val_dataset = FullDataset([config.VAL_IMAGE_DIR], [config.VAL_GT_DIR],
                              config.TRAIN_CONFIG['crop_size'], mode='train')
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    print(f"Val: {len(val_dataset)} samples")

    model = AblationModel(config.HIERA_PATH, variant).to(DEVICE)
    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {tp:,}")

    optim = opt.AdamW([p for p in model.parameters() if p.requires_grad],
                       lr=config.TRAIN_CONFIG['lr'], weight_decay=config.TRAIN_CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(optim, T_max=args.epoch, eta_min=5e-6)

    start_epoch = 0
    best_val_iou = 0.0
    best_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        try:
            optim.load_state_dict(ckpt['optimizer_state_dict'])
            for pg in optim.param_groups: pg['lr'] = config.TRAIN_CONFIG['lr']
        except Exception as e:
            print(f"[WARN] Optimizer not restored: {e}")
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        except Exception as e:
            print(f"[WARN] Scheduler not restored: {e}")
        start_epoch = ckpt['epoch']
        best_val_iou = ckpt.get('val_iou', 0.0)
        print(f"Resumed from epoch {start_epoch}, best_val_iou={best_val_iou:.4f}")
    logger = create_logger(log_dir, resume_from=start_epoch > 0)

    use_sloss = (variant == "full_lite")  # full_lite only; full uses BCE+Dice to isolate architecture
    use_cldice = variant == "full_lite"  # Only full_lite uses clDice for now
    sw = 0.3 if use_sloss else 0.0

    def validate():
        model.eval()
        total_inter = total_union = 0
        with torch.no_grad():
            for batch in val_loader:
                vx, vtgt = batch['image'].to(DEVICE), batch['label'].to(DEVICE)
                vpred = (torch.sigmoid(model(vx)) > 0.5).float()
                total_inter += (vpred * vtgt).sum().item()
                total_union += ((vpred + vtgt) > 0).float().sum().item()
        model.train()
        val_iou = total_inter / max(total_union, 1)
        return val_iou

    for epoch in range(start_epoch, args.epoch):
        eloss = 0; ebce = 0; edice = 0; ecld = 0; esnk = 0; nb = 0
        for i, batch in enumerate(dataloader):
            x = batch['image'].to(DEVICE); tgt = batch['label'].to(DEVICE)
            optim.zero_grad()

            with amp.autocast(enabled=use_fp16):
                if use_sloss:
                    pred, x_red, x3 = model(x, return_aux=True)
                else:
                    pred = model(x)
                    x_red, x3 = None, None
                loss_bce = F.binary_cross_entropy_with_logits(pred, tgt)
                loss_dice = dice_loss(pred, tgt)
                loss = config.LOSS_WEIGHTS['bce_weight'] * loss_bce + \
                       config.LOSS_WEIGHTS['dice_weight'] * loss_dice

            cw = config.LOSS_WEIGHTS['cldice_weight'] if use_cldice else 0
            loss_cld = torch.tensor(0.0, device=DEVICE)
            if cw > 0:
                loss_cld = soft_cl_dice(torch.sigmoid(pred.float()), tgt, k=5)
                loss = loss + cw * loss_cld

            loss_snk = torch.tensor(0.0, device=DEVICE)
            if use_sloss:
                loss_snk = snake_align_loss(model, x_red.float(), x3.float(), tgt)
                loss = loss + sw * loss_snk

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()

            eloss += loss.item(); ebce += loss_bce.item(); edice += loss_dice.item()
            ecld += loss_cld.item(); esnk += loss_snk.item(); nb += 1

            if i % 50 == 0:
                print(f"[{variant}] E{epoch+1}/{args.epoch} S{i} "
                      f"L={loss.item():.4f} BCE={loss_bce.item():.4f} DICE={loss_dice.item():.4f} "
                      f"CLD={loss_cld.item():.4f} SNK={loss_snk.item():.4f}")

        al = eloss/nb; abl = ebce/nb; adl = edice/nb
        acl = ecld/nb; asl = esnk/nb; lr = optim.param_groups[0]['lr']
        scheduler.step()
        print(f"[{variant}] E{epoch+1}: L={al:.4f} BCE={abl:.4f} DICE={adl:.4f} "
              f"CLD={acl:.4f} SNK={asl:.4f} lr={lr:.2e}")
        logger.log_epoch(epoch + 1, al, abl, adl, lr)

        # Validate every epoch and save best by IoU
        val_iou = validate()
        print(f"[{variant}] E{epoch+1} VAL IoU={val_iou:.4f}  "
              f"(best={best_val_iou:.4f} @ E{best_epoch})")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch + 1
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optim.state_dict(),
                         'scheduler_state_dict': scheduler.state_dict(),
                         'val_iou': val_iou, 'variant': variant},
                        os.path.join(ckpt_dir, f'{variant}-best.pth'))

        # Also save latest for crash recovery
        torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optim.state_dict(),
                     'scheduler_state_dict': scheduler.state_dict(),
                     'variant': variant},
                    os.path.join(ckpt_dir, f'{variant}-latest.pth'))

    logger.close()
    # Save final checkpoint
    torch.save({'epoch': args.epoch, 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optim.state_dict(), 'variant': variant},
                os.path.join(ckpt_dir, f'{variant}-final.pth'))
    print(f"[{variant}] Best val IoU={best_val_iou:.4f} @ epoch {best_epoch}")
    print(f"[{variant}] Done.")


if __name__ == "__main__":
    main()
