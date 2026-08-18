"""
微调 TileContextFusion 模块
冻结 encoder + decoder，只训练 context_fusion (~157K 参数)
用 3×3 tile 网格训练，让模型学习哪些邻居方向对长城预测有帮助
"""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import os
import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from collections import defaultdict
import glob
import random

sys.path.insert(0, os.path.dirname(__file__))
from models.sam2gw_net import SAM2GWNet
import config

# ========== 配置 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRIDE = 128
TILE_SIZE = 512
BATCH_SIZE = 4
NUM_EPOCHS = 3
LR = 1e-4
MAX_SAMPLES = 200  # 只用 200 个样本快速微调
SAVE_DIR = config.SAVE_PATH

MODEL_PATH = os.path.join(SAVE_DIR, "SAM2FPN-60.pth")

NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),           ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

# ========== 工具函数 ==========

def parse_tile_position(filename):
    """从训练 tile 文件名解析像素坐标"""
    basename = os.path.basename(filename)
    m = re.search(r'_(\d{6})?_(\d+)_(\d+)\.png$', basename)
    if m:
        return int(m.group(2)), int(m.group(3))
    raise ValueError(f"Cannot parse: {basename}")

def pixel_to_grid(sr, sc):
    return sr // STRIDE, sc // STRIDE

def load_image(path):
    img = Image.open(path).convert('RGB')
    arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(arr)

def load_mask(path):
    img = Image.open(path).convert('L')
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]


# ========== GridTileDataset ==========

class GridTileDataset(Dataset):
    """返回 3×3 tile 组：中心 tile + 邻居 tiles + 中心 mask"""

    def __init__(self, image_dir, gt_dir):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # 构建网格
        img_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.full_grid = {}
        for p in img_paths:
            sr, sc = parse_tile_position(p)
            gr, gc = pixel_to_grid(sr, sc)
            self.full_grid[(gr, gc)] = p

        # 构建 gt 网格
        gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
        self.gt_grid = {}
        for p in gt_paths:
            sr, sc = parse_tile_position(p)
            gr, gc = pixel_to_grid(sr, sc)
            self.gt_grid[(gr, gc)] = p

        # 找到有 ≥1 邻居且有 gt 的 tile
        self.samples = []
        for pos, img_path in self.full_grid.items():
            if pos not in self.gt_grid:
                continue
            nei_positions = []
            nei_dirs = []
            for d, (dr, dc) in enumerate(NEIGHBOR_OFFSETS):
                npos = (pos[0] + dr, pos[1] + dc)
                if npos in self.full_grid:
                    nei_positions.append(npos)
                    nei_dirs.append(d)
            if len(nei_positions) > 0:
                self.samples.append((pos, nei_positions, nei_dirs))

        print(f"  可选训练样本: {len(self.samples)}/{len(self.full_grid)} "
              f"(有邻居且有gt)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        center_pos, nei_positions, nei_dirs = self.samples[idx]

        # 加载中心 tile 图像和 mask
        center_img = load_image(self.full_grid[center_pos])
        center_mask = load_mask(self.gt_grid[center_pos])

        # 加载邻居 tile 图像
        neighbor_imgs = []
        for npos in nei_positions:
            neighbor_imgs.append(load_image(self.full_grid[npos]))

        return {
            'center_img': center_img,
            'center_mask': center_mask,
            'neighbor_imgs': neighbor_imgs,
            'neighbor_dirs': torch.tensor(nei_dirs, dtype=torch.long),
            'center_pos': center_pos,
        }


def collate_fn(batch):
    """自定义 batch 整理：处理变长邻居列表"""
    B = len(batch)

    # 中心图像和 mask
    center_imgs = torch.stack([item['center_img'] for item in batch])
    center_masks = torch.stack([item['center_mask'] for item in batch])

    # 收集所有邻居信息
    all_nei_imgs = []  # 展平的邻居图像列表
    nei_counts = []    # 每个样本的邻居数
    nei_dirs_list = [] # 每个样本的邻居方向

    for item in batch:
        all_nei_imgs.extend(item['neighbor_imgs'])
        nei_counts.append(len(item['neighbor_imgs']))
        nei_dirs_list.append(item['neighbor_dirs'])

    return {
        'center_img': center_imgs,
        'center_mask': center_masks,
        'neighbor_imgs': all_nei_imgs,
        'nei_counts': nei_counts,
        'nei_dirs_list': nei_dirs_list,
    }


# ========== 训练函数 ==========

def dice_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    return (1 - (2 * inter + 1e-6) / (union + 1e-6)).mean()


def main():
    print("=" * 60)
    print("微调 TileContextFusion 模块")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}, LR: {LR}, Batch: {BATCH_SIZE}")

    # ========== 数据集 ==========
    print("\n加载训练数据...")
    img_dir = os.path.join(config.TRAIN_DATASET_ROOT, "train_old", "images")
    gt_dir = os.path.join(config.TRAIN_DATASET_ROOT, "train_old", "gt")
    dataset = GridTileDataset(img_dir, gt_dir)

    # 随机子采样加速微调
    if MAX_SAMPLES and len(dataset) > MAX_SAMPLES:
        indices = random.sample(range(len(dataset)), MAX_SAMPLES)
        dataset = Subset(dataset, indices)
        print(f"  子采样: {MAX_SAMPLES}/{len(dataset)} 样本")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_fn, num_workers=0)
    print(f"  总样本: {len(dataset)}, Batches: {len(dataloader)}")

    # ========== 模型 ==========
    print("\n加载模型...")
    model = SAM2GWNet(config.HIERA_PATH)
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)

    # 冻结除 context_fusion 外的所有参数
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name, param in model.named_parameters():
        if 'context_fusion' not in name:
            param.requires_grad = False
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数: {trainable_before:,} → {trainable_after:,}")

    # ========== 优化器 ==========
    optimizer = opt.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-4
    )
    scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS, eta_min=1e-6)

    # ========== 训练循环 ==========
    model.train()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_loss_bce = 0.0
        epoch_loss_dice = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            center_img = batch['center_img'].to(DEVICE)
            center_mask = batch['center_mask'].to(DEVICE)
            neighbor_imgs = [ni.to(DEVICE) for ni in batch['neighbor_imgs']]
            nei_counts = batch['nei_counts']
            nei_dirs_list = batch['nei_dirs_list']

            B = center_img.size(0)
            center_img_norm = normalize(center_img)

            # 预计算所有邻居的 bridge features（encoder 冻结，无梯度）
            neighbor_d4_all = []
            if len(neighbor_imgs) > 0:
                nei_imgs_stacked = torch.stack(neighbor_imgs)
                nei_imgs_norm = normalize(nei_imgs_stacked)
                with torch.no_grad():
                    _, _, _, nei_x4 = model.encoder(nei_imgs_norm)
                    nei_d4_stacked = model.bridge(nei_x4)  # [total_nei, 256, 16, 16]

            # 将邻居特征按样本分组
            nei_idx = 0
            for i in range(B):
                n = nei_counts[i]
                sample_nei_d4 = []
                for j in range(n):
                    sample_nei_d4.append(nei_d4_stacked[nei_idx:nei_idx+1])
                    nei_idx += 1
                neighbor_d4_all.append(sample_nei_d4)

            # 构建 batch 的 neighbor_d4_list 和 neighbor_dirs
            # 所有样本的邻居方向求并集
            all_dirs_set = set()
            for dirs in nei_dirs_list:
                all_dirs_set.update(dirs.tolist())
            all_dirs = sorted(all_dirs_set)

            if len(all_dirs) > 0:
                dummy = torch.zeros(1, 256, 16, 16, device=DEVICE)
                neighbor_d4_list = []
                neighbor_dirs_t = torch.full((B, len(all_dirs)), -1,
                                             dtype=torch.long, device=DEVICE)

                for dir_idx, d in enumerate(all_dirs):
                    dir_stack = []
                    for i in range(B):
                        dirs_i = nei_dirs_list[i]
                        if d in dirs_i:
                            j = (dirs_i == d).nonzero(as_tuple=True)[0].item()
                            dir_stack.append(neighbor_d4_all[i][j])
                            neighbor_dirs_t[i, dir_idx] = d
                        else:
                            dir_stack.append(dummy)
                    neighbor_d4_list.append(torch.cat(dir_stack, dim=0))
            else:
                neighbor_d4_list = []
                neighbor_dirs_t = None

            # 前向：中心 tile + 上下文融合
            outputs = model(center_img_norm,
                           neighbor_d4_list if len(all_dirs) > 0 else None,
                           neighbor_dirs_t if len(all_dirs) > 0 else None)

            loss_bce = F.binary_cross_entropy_with_logits(outputs, center_mask)
            loss_dice = dice_loss(outputs, center_mask)
            loss = loss_bce + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_bce += loss_bce.item()
            epoch_loss_dice += loss_dice.item()
            n_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1}, "
                      f"Loss={loss.item():.4f} (bce={loss_bce.item():.4f}, "
                      f"dice={loss_dice.item():.4f})")

        avg_loss = epoch_loss / n_batches
        avg_bce = epoch_loss_bce / n_batches
        avg_dice = epoch_loss_dice / n_batches
        scheduler.step()

        print(f"\n  Epoch {epoch+1} 完成: Loss={avg_loss:.4f} "
              f"(BCE={avg_bce:.4f}, Dice={avg_dice:.4f}), "
              f"LR={scheduler.get_last_lr()[0]:.2e}")

        # 每 2 个 epoch 保存
        if (epoch + 1) % 2 == 0:
            save_path = os.path.join(SAVE_DIR, f"SAM2FPN-60_ctxFT-{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'context_fusion_only': True,
            }, save_path)
            print(f"  已保存: {save_path}")

    # ========== 保存最终模型 ==========
    final_path = os.path.join(SAVE_DIR, "SAM2FPN-60_ctxFT.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'context_fusion_only': True,
    }, final_path)
    print(f"\n最终模型已保存: {final_path}")
    print("微调完成!")


if __name__ == "__main__":
    main()
