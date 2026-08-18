"""
SAM2 长城提取 - 批量推理与拼接系统
同时输出二值预测图 + 概率图（用于后处理行走引擎）
"""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import re
import rasterio
from datetime import datetime
from models.sam2gw_net_v3 import SAM2GWNet_v3
from core.dataset import SAM2TestDataset
import config


# ==================== 文件名解析 ====================

def parse_filename_for_position(filename):
    basename = os.path.basename(filename)
    pattern = r'(?:test|tile)_\d+_(\d+)_(\d+)\.(?:png|npy)$'
    match = re.search(pattern, basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"无法解析: {basename}")


# ==================== 拼接函数 ====================

def mosaic_tiles(tile_dir, output_mosaic_path, height, width, vote_threshold):
    vote_count = np.zeros((height, width), dtype=np.int32)
    total_coverage = np.zeros((height, width), dtype=np.int32)

    tile_files = glob.glob(os.path.join(tile_dir, "*.png"))
    if len(tile_files) == 0:
        print(f"警告: 在 {tile_dir} 中未找到切片文件")
        return np.zeros((height, width), dtype=np.uint8)

    print(f"找到 {len(tile_files)} 个切片，开始拼接（投票机制，阈值={vote_threshold}）...")

    success_count = 0
    for tile_file in tile_files:
        try:
            tile = np.array(Image.open(tile_file))
            tile_binary = (tile > 0).astype(np.uint8) if tile.max() > 1 else tile
            start_row, start_col = parse_filename_for_position(tile_file)
            h, w = tile_binary.shape
            end_row = min(start_row + h, height)
            end_col = min(start_col + w, width)

            if start_row < height and start_col < width:
                paste_h = end_row - start_row
                paste_w = end_col - start_col
                vote_count[start_row:end_row, start_col:end_col] += tile_binary[:paste_h, :paste_w]
                total_coverage[start_row:end_row, start_col:end_col] += 1
                success_count += 1
        except Exception as e:
            print(f"跳过文件 {os.path.basename(tile_file)}: {e}")
            continue

    print(f"成功拼接 {success_count}/{len(tile_files)} 个切片")

    total_coverage = np.maximum(total_coverage, 1)
    vote_ratio = vote_count / total_coverage
    mosaic = (vote_ratio >= vote_threshold).astype(np.uint8)

    print(f"\n📊 投票统计:")
    print(f"  被覆盖的像素: {np.sum(total_coverage > 0)}/{height * width} ({np.sum(total_coverage > 0) / (height * width) * 100:.2f}%)")
    print(f"  平均投票比例: {vote_ratio.mean():.3f}")
    print(f"  阈值 {vote_threshold} 下长城像素: {np.sum(mosaic)}")

    os.makedirs(os.path.dirname(output_mosaic_path), exist_ok=True)

    if output_mosaic_path.endswith('.tif'):
        try:
            with rasterio.open(
                output_mosaic_path, 'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=rasterio.uint8,
                compress='lzw'
            ) as dst:
                dst.write(mosaic * 255, 1)
        except:
            Image.fromarray(mosaic * 255).save(output_mosaic_path.replace('.tif', '.png'))
    else:
        if not output_mosaic_path.endswith('.png'):
            output_mosaic_path = output_mosaic_path.replace('.tif', '.png')
        Image.fromarray(mosaic * 255).save(output_mosaic_path)

    print(f"拼接完成: {output_mosaic_path}")
    return mosaic


# ==================== 概率图拼接 ====================

def mosaic_prob_tiles(tile_dir, output_mosaic_path, height, width):
    """将概率切片拼接为全图概率场（重叠区取最大值）"""
    prob_max = np.zeros((height, width), dtype=np.float32)
    prob_count = np.zeros((height, width), dtype=np.int32)

    tile_files = glob.glob(os.path.join(tile_dir, "*.npy"))
    if len(tile_files) == 0:
        print(f"警告: 在 {tile_dir} 中未找到概率切片文件")
        return np.zeros((height, width), dtype=np.float32)

    print(f"找到 {len(tile_files)} 个概率切片，开始拼接（重叠区平均）...")

    success_count = 0
    for tile_file in tile_files:
        try:
            prob = np.load(tile_file)
            start_row, start_col = parse_filename_for_position(tile_file)
            h, w = prob.shape
            end_row = min(start_row + h, height)
            end_col = min(start_col + w, width)

            if start_row < height and start_col < width:
                paste_h = end_row - start_row
                paste_w = end_col - start_col
                region = prob_max[start_row:end_row, start_col:end_col]
                np.maximum(region, prob[:paste_h, :paste_w], out=region)
                prob_count[start_row:end_row, start_col:end_col] += 1
                success_count += 1
        except Exception as e:
            print(f"跳过文件 {os.path.basename(tile_file)}: {e}")
            continue

    print(f"成功拼接 {success_count}/{len(tile_files)} 个概率切片")

    print(f"\n📊 概率场统计:")
    print(f"  覆盖像素: {np.sum(prob_count > 0)}/{height * width}")
    print(f"  概率范围: [{prob_max.min():.4f}, {prob_max.max():.4f}]")
    print(f"  均值: {prob_max.mean():.4f}")
    print(f"  概率>0.5 像素: {np.sum(prob_max > 0.5):,}")
    print(f"  概率>0.7 像素: {np.sum(prob_max > 0.7):,}")

    os.makedirs(os.path.dirname(output_mosaic_path), exist_ok=True)

    with rasterio.open(
        output_mosaic_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=rasterio.float32,
        compress='lzw'
    ) as dst:
        dst.write(prob_max, 1)

    print(f"概率图拼接完成: {output_mosaic_path}")
    return prob_max


# ==================== 主函数 ====================

def main():
    # 从配置文件读取参数
    MODEL_PATH = os.path.join(config.SAVE_PATH, "SAM2FPNv3-74.pth")  # v3 SnakeBridge
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    INFERENCE_CFG = config.INFERENCE_CONFIG

    print("=" * 70)
    print("SAM2 长城提取 - 批量推理与拼接系统")
    print("=" * 70)
    print(f"设备: {DEVICE}")
    print(f"大图尺寸: {config.IMAGE_WIDTH} x {config.IMAGE_HEIGHT}")
    print(f"二值化阈值: {INFERENCE_CFG['threshold']}")
    print(f"拼接投票阈值: {INFERENCE_CFG['vote_threshold']}")
    print(f"找到时间序列: {os.listdir(config.DATASET_CHANGE_DIR)}")
    print("-" * 70)

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\n加载模型...")
    model = SAM2GWNet_v3(config.HIERA_PATH)
    checkpoint = torch.load(MODEL_PATH)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 加载检查点 (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ 加载检查点")

    model.to(DEVICE)
    model.eval()

    time_folders = sorted([f for f in os.listdir(config.DATASET_CHANGE_DIR)
                          if os.path.isdir(os.path.join(config.DATASET_CHANGE_DIR, f))])

    for time_folder in time_folders:
        print("\n" + "=" * 70)
        print(f"处理时间序列: {time_folder}")
        print("=" * 70)

        TEST_IMAGE_DIR = os.path.join(config.DATASET_CHANGE_DIR, time_folder)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        RUN_OUTPUT_DIR = os.path.join(config.INFERENCE_OUTPUT_ROOT, time_folder, timestamp)
        INFER_OUTPUT_DIR = os.path.join(RUN_OUTPUT_DIR, "inference_tiles")
        PROB_OUTPUT_DIR = os.path.join(RUN_OUTPUT_DIR, "prob_tiles")
        os.makedirs(INFER_OUTPUT_DIR, exist_ok=True)
        os.makedirs(PROB_OUTPUT_DIR, exist_ok=True)

        FULL_PRED_PATH = os.path.join(config.INFERENCE_OUTPUT_ROOT, time_folder, f"{time_folder}_mosaic.tif")
        FULL_PROB_PATH = os.path.join(config.INFERENCE_OUTPUT_ROOT, time_folder, f"{time_folder}_prob.tif")

        dataset = SAM2TestDataset(image_dir=TEST_IMAGE_DIR, transform=transform)
        if len(dataset) == 0:
            print(f"⚠️ 跳过 {time_folder}: 没有找到图像文件")
            continue

        dataloader = DataLoader(dataset, batch_size=INFERENCE_CFG['batch_size'],
                                shuffle=False, num_workers=0)

        print("\n开始推理...")
        with torch.no_grad():
            for batch_idx, (images, img_paths) in enumerate(dataloader):
                images = images.to(DEVICE)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs > INFERENCE_CFG['threshold']).float()

                for i in range(len(img_paths)):
                    base_name = os.path.splitext(os.path.basename(img_paths[i]))[0]

                    # 保存二值预测
                    pred_np = preds[i].squeeze().cpu().numpy()
                    pred_uint8 = (pred_np * 255).astype(np.uint8)
                    output_path = os.path.join(INFER_OUTPUT_DIR, f"{base_name}.png")
                    Image.fromarray(pred_uint8).save(output_path)

                    # 保存概率图 (float32)
                    prob_np = probs[i].squeeze().cpu().numpy().astype(np.float32)
                    prob_path = os.path.join(PROB_OUTPUT_DIR, f"{base_name}.npy")
                    np.save(prob_path, prob_np)

                if (batch_idx + 1) % 10 == 0:
                    print(f"  进度: {batch_idx + 1}/{len(dataloader)} 批次")

        print(f"推理完成，处理了 {len(dataset)} 张图像")

        print("\n开始拼接...")
        mosaic_tiles(INFER_OUTPUT_DIR, FULL_PRED_PATH,
                    config.IMAGE_HEIGHT, config.IMAGE_WIDTH,
                    INFERENCE_CFG['vote_threshold'])

        print("\n开始拼接概率图...")
        mosaic_prob_tiles(PROB_OUTPUT_DIR, FULL_PROB_PATH,
                          config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

        print(f"✅ {time_folder} 处理完成!")

    print("\n" + "=" * 70)
    print("✅ 批量处理完成！")
    print(f"输出根目录: {config.INFERENCE_OUTPUT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
