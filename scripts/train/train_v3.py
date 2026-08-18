"""Train v3 (SnakeBridge) from scratch or resume."""
import os
import argparse
import glob
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from core.dataset import FullDataset
from models.sam2gw_net_v3 import SAM2GWNet_v3
from core.logger import create_logger
from utils.cldice_loss import soft_cl_dice
import config

torch.cuda.empty_cache()

# Auto-discover latest v3 checkpoint (numeric sort by epoch number)
_v3_ckpts = sorted(glob.glob(os.path.join(config.SAVE_PATH, "SAM2FPNv3-*.pth")),
                   key=lambda p: int(p.split('-')[-1].split('.')[0]))
_latest_v3 = _v3_ckpts[-1] if _v3_ckpts else None

parser = argparse.ArgumentParser("SAM2-SnakeBridge-v3")
parser.add_argument("--hiera_path", type=str, default=config.HIERA_PATH)
parser.add_argument('--save_path', type=str, default=config.TRAIN_OUTPUT_DIR)
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--lr", type=float, default=config.TRAIN_CONFIG['lr'])
parser.add_argument("--batch_size", default=config.TRAIN_CONFIG['batch_size'], type=int)
parser.add_argument("--weight_decay", default=config.TRAIN_CONFIG['weight_decay'], type=float)
parser.add_argument("--resume", type=str, default=_latest_v3, help="resume from v3 checkpoint")
parser.add_argument("--no_resume", action='store_true', help="force train from scratch")
args = parser.parse_args()

use_fp16 = config.TRAIN_CONFIG['use_fp16']
scaler = amp.GradScaler(enabled=use_fp16, growth_interval=999999)  # old API: compatible with dev torch


def dice_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    return (1 - (2 * inter + 1e-6) / (union + 1e-6)).mean()


def snake_align_loss(model, x_red, x3, target, device):
    """Encourage snake centerlines to sample from wall regions.
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


    Penalizes centerline sampling points that land outside the wall mask
    (downsampled to bridge/x3 resolution). Only active in cells containing wall.
    """
    from utils.dsconv import get_coordinate_map_2d

    total = 0.0
    n = 0

    # ── x4 bridge snakes (H/32, 16×16) ──
    with torch.no_grad():
        gt_fm = F.adaptive_avg_pool2d(target.float(), x_red.shape[2:])
        wall_mask = (gt_fm > 0.3).float()
    if wall_mask.sum() >= 1:
        for seq in [model.bridge.s_h, model.bridge.s_v]:
            dsconv = seq[3]
            x_mid = seq[0](x_red); x_mid = seq[1](x_mid); x_mid = seq[2](x_mid)
            offset = dsconv.offset_conv(x_mid)
            wp = dsconv.width_scale * F.softplus(dsconv.width_conv(x_mid)) if dsconv.thickness > 1 else None
            grid = get_coordinate_map_2d(offset, dsconv.morph, dsconv.kernel_size,
                                         dsconv.dilation, device, dsconv.thickness, wp)
            K = dsconv.kernel_size; T = dsconv.thickness
            c_start = T // 2 * K
            centerline = grid[:, c_start:c_start + K]
            for k in range(K):
                g = centerline[:, k]
                gt_s = F.grid_sample(gt_fm, g, mode='bilinear', padding_mode='border', align_corners=True)
                penalty = wall_mask * (1.0 - gt_s)
                total += penalty.sum() / wall_mask.sum().clamp(min=1)
                n += 1

    # ── x3 thick skip snakes (H/16, 32×32) ──
    with torch.no_grad():
        gt_fm3 = F.adaptive_avg_pool2d(target.float(), x3.shape[2:])
        wall_mask3 = (gt_fm3 > 0.3).float()
    if wall_mask3.sum() >= 1:
        x3_mid = model.thick_x3.reduce(x3)  # Sequential: Conv → BN → ReLU (correct)
        for seq in [model.thick_x3.s_h, model.thick_x3.s_v]:
            dsconv = seq
            offset = dsconv.offset_conv(x3_mid)
            wp = dsconv.width_scale * F.softplus(dsconv.width_conv(x3_mid))
            grid = get_coordinate_map_2d(offset, dsconv.morph, dsconv.kernel_size,
                                         dsconv.dilation, device, dsconv.thickness, wp)
            K = dsconv.kernel_size; T = dsconv.thickness
            c_start = T // 2 * K
            centerline = grid[:, c_start:c_start + K]
            for k in range(K):
                g = centerline[:, k]
                gt_s = F.grid_sample(gt_fm3, g, mode='bilinear', padding_mode='border', align_corners=True)
                penalty = wall_mask3 * (1.0 - gt_s)
                total += penalty.sum() / wall_mask3.sum().clamp(min=1)
                n += 1

    return torch.tensor(total / max(n, 1), device=device)


def main(args):
    image_dirs, gt_dirs = config.discover_train_dirs()
    print(f"\nFound {len(image_dirs)} training sets")
    for i, (img_d, gt_d) in enumerate(zip(image_dirs, gt_dirs)):
        print(f"  {i+1}. images: {img_d}  |  gt: {gt_d}")

    dataset = FullDataset(image_dirs, gt_dirs,
                          config.TRAIN_CONFIG['crop_size'], mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=config.TRAIN_CONFIG['num_workers'])

    device = torch.device("cuda")
    model = SAM2GWNet_v3(args.hiera_path)
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = opt.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3, min_lr=5e-6)

    start_epoch = 0
    resume_from = False
    if args.resume and os.path.exists(args.resume) and not args.no_resume:
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        try:
            optim.load_state_dict(ckpt['optimizer_state_dict'])
            for pg in optim.param_groups:
                pg['lr'] = args.lr
            print("[OK] Optimizer state restored, LR reset")
        except Exception as e:
            print(f"[WARN] Optimizer not restored: {e}")
        start_epoch = ckpt['epoch']
        resume_from = True
        print(f"[OK] Resuming from epoch {start_epoch}")
    else:
        print("Training from scratch (v3 SnakeBridge)")

    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    os.makedirs(args.save_path, exist_ok=True)
    train_logger = create_logger(args.save_path, resume_from=resume_from)

    for epoch in range(start_epoch, args.epoch):
        epoch_loss = 0
        epoch_loss_bce = 0
        epoch_loss_dice = 0
        epoch_loss_cldice = 0
        num_batches = 0

        for i, batch in enumerate(dataloader):
            x = batch['image']
            target = batch['label']
            x = x.to(device)
            target = target.to(device)

            optim.zero_grad()

            with amp.autocast(enabled=use_fp16):
                pred1, x_red, x3 = model(x, return_bridge_feat=True)
                loss_bce = F.binary_cross_entropy_with_logits(pred1, target)
                loss_dice = dice_loss(pred1, target)
                loss_seg = (config.LOSS_WEIGHTS['bce_weight'] * loss_bce +
                           config.LOSS_WEIGHTS['dice_weight'] * loss_dice)
                # Snake alignment inside autocast for FP16 consistency
                loss_snake = snake_align_loss(model, x_red.float(), x3.float(), target, device)

            cls_weight = config.LOSS_WEIGHTS.get('cldice_weight', 0.1)
            if cls_weight > 0:
                pred_prob = torch.sigmoid(pred1.float())
                loss_cldice = soft_cl_dice(pred_prob, target, k=5)
                loss = loss_seg + cls_weight * loss_cldice
            else:
                loss = loss_seg

            loss = loss + 0.3 * loss_snake

            if i % 50 == 0:
                cld_str = f", cldice={loss_cldice.item():.4f}" if cls_weight > 0 else ""
                snake_str = f", snake={loss_snake.item():.4f}"
                print(f"Epoch {epoch + 1}/{args.epoch}, Step {i}, Loss: {loss.item():.4f} "
                      f"(bce={loss_bce.item():.4f}, dice={loss_dice.item():.4f}{cld_str}{snake_str})")

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()

            epoch_loss += loss.item()
            epoch_loss_bce += loss_bce.item()
            epoch_loss_dice += loss_dice.item()
            if cls_weight > 0:
                epoch_loss_cldice += loss_cldice.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_bce = epoch_loss_bce / num_batches
        avg_dice = epoch_loss_dice / num_batches
        avg_cldice = epoch_loss_cldice / num_batches if epoch_loss_cldice > 0 else 0
        current_lr = optim.param_groups[0]['lr']
        scheduler.step(avg_loss)

        cl_info = f", clDice={avg_cldice:.4f}" if avg_cldice > 0 else ""
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f} (bce={avg_bce:.4f}, dice={avg_dice:.4f}{cl_info}), lr={current_lr:.2e}")
        train_logger.log_epoch(epoch + 1, avg_loss, avg_bce, avg_dice, current_lr)

        save_path = os.path.join(args.save_path, f'SAM2FPNv3-{epoch + 1}.pth')
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'args': vars(args)
        }
        torch.save(checkpoint, save_path)
        print(f'[Saved] {save_path}')

    train_logger.close()


if __name__ == "__main__":
    main(args)
