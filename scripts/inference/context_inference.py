"""
上下文感知推理 — Neighborhood Context Fusion
两阶段管线：
  Pass 1: 编码全部切片 → 缓存 bridge features
  Pass 2: 逐切片融合邻居 bridge 特征 → 解码 → 保存
  Phase 3: 拼接为全图
"""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import os
import re
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import glob
from datetime import datetime
from collections import defaultdict

from models.sam2gw_net import SAM2GWNet
from scripts.inference.predict import mosaic_tiles, mosaic_prob_tiles
import config


# ========== Tile position helpers ==========

def parse_start_row_col(filename):
    """从 tile 文件名解析像素坐标"""
    basename = os.path.basename(filename)
    m = re.search(r'tile_\d+_(\d+)_(\d+)\.png$', basename)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f"Cannot parse position: {basename}")


def pixel_to_grid(start_row, start_col, stride=128):
    """像素坐标转为网格坐标"""
    return start_row // stride, start_col // stride


# ========== Neighbor offsets (8 directions) ==========

NEIGHBOR_OFFSETS = [
    (-1, -1),  # 0: top-left
    (-1,  0),  # 1: top
    (-1,  1),  # 2: top-right
    ( 0, -1),  # 3: left
    ( 0,  1),  # 4: right
    ( 1, -1),  # 5: bottom-left
    ( 1,  0),  # 6: bottom
    ( 1,  1),  # 7: bottom-right
]


# ========== Grid map ==========

def build_grid_map(tile_dir):
    """从 tile 文件构建稀疏网格字典 {(grid_row, grid_col): {path, bridge_feat}}"""
    tile_paths = sorted(glob.glob(os.path.join(tile_dir, "*.png")))
    if not tile_paths:
        raise FileNotFoundError(f"No tiles found in {tile_dir}")

    grid_map = {}
    max_r, max_c = 0, 0

    for path in tile_paths:
        sr, sc = parse_start_row_col(path)
        gr, gc = pixel_to_grid(sr, sc, config.CONTEXT_INFERENCE_CONFIG['stride'])
        grid_map[(gr, gc)] = {'path': path, 'bridge_feat': None}
        max_r = max(max_r, gr)
        max_c = max(max_c, gc)

    print(f"  网格: {max_r + 1} 行 × {max_c + 1} 列, {len(grid_map)} 个有效切片")
    return grid_map, max_r + 1, max_c + 1


def get_neighbors(grid_pos, grid_map):
    """返回 grid_pos 的所有邻居 bridge features 和方向索引"""
    r, c = grid_pos
    neighbor_feats = []
    neighbor_dirs = []

    for d, (dr, dc) in enumerate(NEIGHBOR_OFFSETS):
        entry = grid_map.get((r + dr, c + dc))
        if entry is not None and entry['bridge_feat'] is not None:
            neighbor_feats.append(entry['bridge_feat'])
            neighbor_dirs.append(d)

    return neighbor_feats, neighbor_dirs


# ========== Image loading ==========

def load_image_tensor(path, device='cpu'):
    """加载并归一化图像为 tensor [3, H, W]"""
    img = Image.open(path).convert('RGB')
    arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(arr).to(device)


# ========== Pass 1: Encode all tiles ==========

@torch.no_grad()
def encode_pass(model, grid_map, device):
    """编码全部切片，缓存 bridge features 到 grid_map"""
    cfg = config.CONTEXT_INFERENCE_CONFIG
    entries = [(pos, info) for pos, info in grid_map.items()]
    n_total = len(entries)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    for start in range(0, n_total, cfg['batch_size_encode']):
        batch_entries = entries[start:start + cfg['batch_size_encode']]

        imgs = [load_image_tensor(info['path'], device)
                for _, info in batch_entries]
        imgs = torch.stack(imgs)
        imgs = normalize(imgs)

        # 仅编码器 + bridge，不解码
        x1, x2, x3, x4 = model.encoder(imgs)
        d4 = model.bridge(x4)  # [B, 256, 16, 16]

        for i, (pos, info) in enumerate(batch_entries):
            feat = d4[i].cpu()
            if cfg['store_bridge_fp16']:
                feat = feat.half()
            info['bridge_feat'] = feat

        if (start // cfg['batch_size_encode'] + 1) % 20 == 0:
            print(f"  Pass 1: {min(start + cfg['batch_size_encode'], n_total)}/{n_total}")

    print(f"  Pass 1 完成: {n_total} 个切片已编码")


# ========== Pass 2: Decode with context ==========

@torch.no_grad()
def decode_pass(model, grid_map, device, infer_dir, prob_dir):
    """逐切片融合邻居特征，解码并保存"""
    cfg = config.CONTEXT_INFERENCE_CONFIG
    threshold = config.INFERENCE_CONFIG['threshold']
    entries = [(pos, info) for pos, info in grid_map.items()]
    n_total = len(entries)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # 按邻居数分组以支持批量处理
    groups = defaultdict(list)
    for pos, info in entries:
        nei_feats, _ = get_neighbors(pos, grid_map)
        groups[len(nei_feats)].append((pos, info))

    for n_neighbors, group_entries in sorted(groups.items()):
        print(f"  解码 {len(group_entries)} 个切片 ({n_neighbors} 邻居)...")

        batch_size = cfg['batch_size_decode']
        for start in range(0, len(group_entries), batch_size):
            batch_entries = group_entries[start:start + batch_size]
            B = len(batch_entries)

            # 加载中心切片图像 + 收集邻居特征
            imgs = []
            nei_lists = []
            for pos, info in batch_entries:
                imgs.append(load_image_tensor(info['path'], device))
                nei_lists.append(get_neighbors(pos, grid_map))

            imgs = torch.stack(imgs)
            imgs = normalize(imgs)

            if n_neighbors > 0:
                # 求 batch 内所有方向并集
                all_dirs_set = set()
                for nei_feats, nei_dirs in nei_lists:
                    all_dirs_set.update(nei_dirs)
                all_dirs = sorted(all_dirs_set)

                # 构建 per-direction tensors [B, C, H, W]
                dummy = torch.zeros(1, 256, 16, 16, device=device,
                                    dtype=torch.float16 if cfg['store_bridge_fp16'] else torch.float32)

                neighbor_d4_list = []
                neighbor_dirs_t = torch.full((B, len(all_dirs)), -1,
                                             dtype=torch.long, device=device)

                for dir_idx, d in enumerate(all_dirs):
                    dir_stack = []
                    for i, (nei_feats, nei_dirs) in enumerate(nei_lists):
                        if d in nei_dirs:
                            j = nei_dirs.index(d)
                            feat = nei_feats[j].to(device=device)
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

            # 完整推理：encode + context fusion + decode
            outputs = model(imgs,
                            neighbor_d4_list if n_neighbors > 0 else None,
                            neighbor_dirs_t if n_neighbors > 0 else None)

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            # 保存
            for i, (pos, info) in enumerate(batch_entries):
                base_name = os.path.splitext(os.path.basename(info['path']))[0]

                pred_np = preds[i].squeeze().cpu().numpy()
                pred_uint8 = (pred_np * 255).astype(np.uint8)
                Image.fromarray(pred_uint8).save(
                    os.path.join(infer_dir, f"{base_name}.png"))

                prob_np = probs[i].squeeze().cpu().numpy().astype(np.float32)
                np.save(os.path.join(prob_dir, f"{base_name}.npy"), prob_np)

            if (start // batch_size + 1) % 50 == 0:
                print(f"    {min(start + batch_size, len(group_entries))}/{len(group_entries)}")

    print(f"  Pass 2 完成: {n_total} 个切片已解码")


# ========== Main ==========

def main():
    MODEL_PATH = os.path.join(config.SAVE_PATH, "SAM2FPN-60.pth")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config.CONTEXT_INFERENCE_CONFIG

    print("=" * 70)
    print("SAM2 长城提取 — 上下文融合推理系统")
    print("=" * 70)
    print(f"设备: {DEVICE}")
    print(f"大图尺寸: {config.IMAGE_WIDTH} x {config.IMAGE_HEIGHT}")
    print(f"切片尺寸: {cfg['tile_size']}, 步长: {cfg['stride']}")
    print(f"邻域大小: {cfg['neighbor_grid_size']}x{cfg['neighbor_grid_size']}")
    print(f"二值化阈值: {config.INFERENCE_CONFIG['threshold']}")
    print("-" * 70)

    # 加载模型
    print("\n加载模型...")
    model = SAM2GWNet(config.HIERA_PATH)
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"加载检查点 (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        state_dict = checkpoint
        print("加载模型权重")

    # 处理旧 checkpoint 中不存在的 context_fusion 键
    model_dict = model.state_dict()
    ctx_keys = [k for k in model_dict if 'context_fusion' in k]
    loaded_ctx = [k for k in state_dict if 'context_fusion' in k]
    if ctx_keys and not loaded_ctx:
        print("[INFO] context_fusion 参数未在 checkpoint 中找到，使用随机初始化")
        for k in ctx_keys:
            if k in model_dict and k not in state_dict:
                state_dict[k] = model_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    # 处理各时相
    time_folders = sorted([f for f in os.listdir(config.DATASET_CHANGE_DIR)
                          if os.path.isdir(os.path.join(config.DATASET_CHANGE_DIR, f))])

    for time_folder in time_folders:
        print("\n" + "=" * 70)
        print(f"处理时相: {time_folder}")
        print("=" * 70)

        tile_dir = os.path.join(config.DATASET_CHANGE_DIR, time_folder)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(config.INFERENCE_OUTPUT_ROOT, time_folder,
                               f"context_{timestamp}")
        infer_dir = os.path.join(run_dir, "inference_tiles")
        prob_dir = os.path.join(run_dir, "prob_tiles")
        os.makedirs(infer_dir, exist_ok=True)
        os.makedirs(prob_dir, exist_ok=True)

        mosaic_path = os.path.join(config.INFERENCE_OUTPUT_ROOT, time_folder,
                                   f"{time_folder}_context_mosaic.tif")
        prob_mosaic_path = os.path.join(config.INFERENCE_OUTPUT_ROOT, time_folder,
                                        f"{time_folder}_context_prob.tif")

        # 构建网格
        print(f"\n[1/4] 构建网格...")
        grid_map, n_rows, n_cols = build_grid_map(tile_dir)

        # Pass 1
        print(f"\n[2/4] Pass 1: 编码 {len(grid_map)} 个切片...")
        encode_pass(model, grid_map, DEVICE)

        # Pass 2
        print(f"\n[3/4] Pass 2: 上下文融合解码...")
        decode_pass(model, grid_map, DEVICE, infer_dir, prob_dir)

        # 拼接
        print(f"\n[4/4] 拼接...")
        mosaic_tiles(infer_dir, mosaic_path,
                     config.IMAGE_HEIGHT, config.IMAGE_WIDTH,
                     config.INFERENCE_CONFIG['vote_threshold'])

        mosaic_prob_tiles(prob_dir, prob_mosaic_path,
                          config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

        # 释放内存
        for info in grid_map.values():
            info['bridge_feat'] = None

        print(f"\n✅ {time_folder} 处理完成!")
        print(f"   二值图: {mosaic_path}")
        print(f"   概率图: {prob_mosaic_path}")

    print("\n" + "=" * 70)
    print("✅ 全部完成！")
    print(f"输出根目录: {config.INFERENCE_OUTPUT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
