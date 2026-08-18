"""
快速对比实验：原始推理 vs 上下文融合推理
在小规模子集上运行，输出关键差异指标
"""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import glob
import random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from models.sam2gw_net import SAM2GWNet
from context_inference import (build_grid_map, get_neighbors,
                                parse_start_row_col, pixel_to_grid,
                                NEIGHBOR_OFFSETS)
import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = config.CONTEXT_INFERENCE_CONFIG
MODEL_PATH = os.path.join(config.SAVE_PATH, "SAM2FPN-60_ctxFT.pth")
THRESHOLD = config.INFERENCE_CONFIG['threshold']

# ========== 加载模型 ==========
print("加载模型...")
model = SAM2GWNet(config.HIERA_PATH)
ckpt = torch.load(MODEL_PATH, map_location='cpu')
state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
model.load_state_dict(state_dict, strict=False)  # context_fusion 是新增的
model.to(DEVICE)
model.eval()
print(f"Model loaded (epoch {ckpt.get('epoch', '?')})")

# ========== 数据准备 ==========
tile_dir = os.path.join(config.DATASET_CHANGE_DIR, "2011-09-19")
all_tiles = sorted(glob.glob(os.path.join(tile_dir, "*.png")))

# 构建完整网格，然后取中间一块连续区域
full_grid = {}
for path in all_tiles:
    sr, sc = parse_start_row_col(path)
    gr, gc = pixel_to_grid(sr, sc, cfg['stride'])
    full_grid[(gr, gc)] = path

# 取最密集的连续区块（rows 2-16, cols 222-238）确保大部分有 8 邻居
grid_map = {}
for r in range(2, 18):
    for c in range(222, 240):
        path = full_grid.get((r, c))
        if path:
            grid_map[(r, c)] = {'path': path, 'bridge_feat': None}

print(f"\n连续区块: {len(grid_map)} 个 tile (rows 2-17, cols 222-239)")

# 统计邻居分布
nei_counts = defaultdict(int)
for pos in grid_map:
    nei_feats, _ = get_neighbors(pos, grid_map)
    nei_counts[len(nei_feats)] += 1
print("邻居数分布:", dict(sorted(nei_counts.items())))

# ========== 原始推理（无context）==========
print("\n" + "=" * 60)
print("[1/3] 原始推理（无上下文）...")
print("=" * 60)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

original_preds = {}
original_probs = {}

with torch.no_grad():
    for idx, (pos, info) in enumerate(grid_map.items()):
        img = Image.open(info['path']).convert('RGB')
        arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
        t = normalize(t)

        out = model(t)  # 无 context
        prob = torch.sigmoid(out)
        pred = (prob > THRESHOLD).float()

        original_preds[pos] = pred.squeeze().cpu().numpy()
        original_probs[pos] = prob.squeeze().cpu().numpy()

        if (idx + 1) % 20 == 0:
            print(f"  进度: {idx + 1}/{len(grid_map)}")

print("原始推理完成")

# ========== Pass 1: 编码全部 tile（缓存 bridge features）==========
print("\n" + "=" * 60)
print("[2/3] 上下文推理 — Pass 1 编码...")
print("=" * 60)

with torch.no_grad():
    entries = list(grid_map.items())
    for start in range(0, len(entries), cfg['batch_size_encode']):
        batch = entries[start:start + cfg['batch_size_encode']]

        imgs = []
        for pos, info in batch:
            img = Image.open(info['path']).convert('RGB')
            arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
            imgs.append(torch.from_numpy(arr))

        imgs = torch.stack(imgs).to(DEVICE)
        imgs = normalize(imgs)

        x1, x2, x3, x4 = model.encoder(imgs)
        d4 = model.bridge(x4)

        for i, (pos, info) in enumerate(batch):
            feat = d4[i].cpu()
            if cfg['store_bridge_fp16']:
                feat = feat.half()
            info['bridge_feat'] = feat

        if (start // cfg['batch_size_encode'] + 1) % 10 == 0:
            print(f"  Pass 1: {min(start + cfg['batch_size_encode'], len(entries))}/{len(entries)}")

print("Pass 1 完成")

# ========== Pass 2: 上下文解码 ==========
print("\n" + "=" * 60)
print("[3/3] 上下文推理 — Pass 2 解码...")
print("=" * 60)

context_preds = {}
context_probs = {}

# 按邻居数分组
groups = defaultdict(list)
for pos, info in grid_map.items():
    nei_feats, _ = get_neighbors(pos, grid_map)
    groups[len(nei_feats)].append((pos, info))

with torch.no_grad():
    for n_neighbors, group_entries in sorted(groups.items()):
        print(f"  解码 {len(group_entries)} tiles ({n_neighbors} 邻居)...")

        for start in range(0, len(group_entries), cfg['batch_size_decode']):
            batch = group_entries[start:start + cfg['batch_size_decode']]
            B = len(batch)

            imgs = []
            nei_lists = []
            for pos, info in batch:
                img = Image.open(info['path']).convert('RGB')
                arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
                imgs.append(torch.from_numpy(arr))
                nei_lists.append(get_neighbors(pos, grid_map))

            imgs = torch.stack(imgs).to(DEVICE)
            imgs = normalize(imgs)

            if n_neighbors > 0:
                all_dirs_set = set()
                for nei_feats, nei_dirs in nei_lists:
                    all_dirs_set.update(nei_dirs)
                all_dirs = sorted(all_dirs_set)

                dummy = torch.zeros(1, 256, 16, 16, device=DEVICE,
                                    dtype=torch.float16 if cfg['store_bridge_fp16'] else torch.float32)
                neighbor_d4_list = []
                neighbor_dirs_t = torch.full((B, len(all_dirs)), -1,
                                             dtype=torch.long, device=DEVICE)

                for dir_idx, d in enumerate(all_dirs):
                    dir_stack = []
                    for i, (nei_feats, nei_dirs) in enumerate(nei_lists):
                        if d in nei_dirs:
                            j = nei_dirs.index(d)
                            feat = nei_feats[j].to(device=DEVICE)
                            if feat.dim() == 3:
                                feat = feat.unsqueeze(0)
                            dir_stack.append(feat)
                            neighbor_dirs_t[i, dir_idx] = d
                        else:
                            dir_stack.append(dummy)
                    dir_tensor = torch.cat(dir_stack, dim=0)
                    if cfg['store_bridge_fp16']:
                        dir_tensor = dir_tensor.float()
                    neighbor_d4_list.append(dir_tensor)
            else:
                neighbor_d4_list = []
                neighbor_dirs_t = None

            out = model(imgs,
                       neighbor_d4_list if n_neighbors > 0 else None,
                       neighbor_dirs_t if n_neighbors > 0 else None)

            prob = torch.sigmoid(out)
            pred = (prob > THRESHOLD).float()

            for i, (pos, info) in enumerate(batch):
                context_preds[pos] = pred[i].squeeze().cpu().numpy()
                context_probs[pos] = prob[i].squeeze().cpu().numpy()

print("Pass 2 完成")

# ========== 对比分析 ==========
print("\n" + "=" * 70)
print("                    对 比 分 析 结 果")
print("=" * 70)

# 1. 像素级一致性
all_keys = list(original_preds.keys())
total_pixels = 0
agreed_pixels = 0
pos_original = 0
pos_context = 0
abs_diff_sum = 0.0

for pos in all_keys:
    op = original_preds[pos]
    cp = context_preds[pos]
    oprob = original_probs[pos]
    cprob = context_probs[pos]

    agreed_pixels += (op == cp).sum()
    total_pixels += op.size
    pos_original += op.sum()
    pos_context += cp.sum()
    abs_diff_sum += np.abs(oprob - cprob).sum()

agreement = agreed_pixels / total_pixels
mean_abs_prob_diff = abs_diff_sum / total_pixels

print(f"\n样本数: {len(all_keys)}")
print(f"每个 tile 尺寸: {op.shape[0]}×{op.shape[1]} = {op.size} 像素")
print(f"总像素: {total_pixels:,}")

print(f"\n--- 二值预测 ---")
print(f"像素一致率: {agreement * 100:.2f}%")
print(f"不一致像素: {(1 - agreement) * 100:.2f}%")
print(f"原始预测长城像素数: {int(pos_original):,} ({pos_original/total_pixels*100:.2f}%)")
print(f"上下文预测长城像素数: {int(pos_context):,} ({pos_context/total_pixels*100:.2f}%)")
print(f"长城像素变化: {int(pos_context - pos_original):,} ({(pos_context/pos_original - 1)*100:+.1f}%)")

print(f"\n--- 概率图 ---")
print(f"平均绝对概率差: {mean_abs_prob_diff:.4f}")

# 2. 按邻居数的分组分析
print(f"\n--- 按邻居数分组 ---")
for n in sorted(set(len(get_neighbors(pos, grid_map)[0]) for pos in grid_map)):
    keys_n = [pos for pos in all_keys if len(get_neighbors(pos, grid_map)[0]) == n]
    if not keys_n:
        continue

    agreed = 0
    total = 0
    abs_diff = 0.0
    for pos in keys_n:
        op = original_preds[pos]
        cp = context_preds[pos]
        oprob = original_probs[pos]
        cprob = context_probs[pos]
        agreed += (op == cp).sum()
        total += op.size
        abs_diff += np.abs(oprob - cprob).sum()

    print(f"  {n} 邻居 ({len(keys_n)} tiles): "
          f"一致率={agreed/total*100:.2f}%, "
          f"概率差={abs_diff/total:.4f}")

# 3. 找差异最大的 tile
print(f"\n--- 差异最大的 5 个 tile ---")
tile_diffs = []
for pos in all_keys:
    diff = np.abs(original_probs[pos] - context_probs[pos]).mean()
    tile_diffs.append((pos, diff))
tile_diffs.sort(key=lambda x: x[1], reverse=True)

for pos, diff in tile_diffs[:5]:
    n_nei = len(get_neighbors(pos, grid_map)[0])
    print(f"  ({pos[0]}, {pos[1]}) | {n_nei}邻居 | 概率差={diff:.6f}")

print("\n" + "=" * 70)
print("对比实验完成")
print("=" * 70)
