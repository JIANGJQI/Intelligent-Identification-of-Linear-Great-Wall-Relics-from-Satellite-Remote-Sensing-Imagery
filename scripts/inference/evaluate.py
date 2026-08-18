"""
SAM2 长城提取 - 测试与评估系统 (无后处理版)
功能：
1. 对切片图像进行推理
2. 拼接成全图
3. 全局评估指标计算与可视化
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
import rasterio
import re
import matplotlib.pyplot as plt
import json
from datetime import datetime
from models.sam2gw_net import SAM2GWNet
from core.dataset import SAM2TestDataset
import config

# ==================== 设置中文字体 ====================
import matplotlib
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False
    USE_CHINESE = True
except:
    USE_CHINESE = False
    print("警告: 中文字体设置失败，将使用英文显示")


# ==================== 文件名解析 ====================

def parse_filename_for_position(filename):
    """
    从文件名解析切片位置
    适配格式: test_编号_行_列.png
    """
    basename = os.path.basename(filename)
    pattern = r'test_\d+_(\d+)_(\d+)\.png$'
    match = re.search(pattern, basename)

    if match:
        start_row = int(match.group(1))
        start_col = int(match.group(2))
        return start_row, start_col
    else:
        # 兼容旧格式
        pattern2 = r'(\d+)\.png$'
        match2 = re.search(pattern2, basename)
        if match2:
            print(f"警告: 文件名 {basename} 不包含位置信息")
            return 0, 0
        raise ValueError(f"无法从文件名 {basename} 中解析出坐标")


# ==================== 拼接函数 ====================

# ==================== 拼接函数（投票机制） ====================

def mosaic_tiles(tile_dir, output_mosaic_path, original_image_path=None, target_shape=None, vote_threshold=0.5):
    """
    将切片拼接成全图 - 投票机制
    记录每个像素被预测为长城的次数，超过阈值才记为长城

    Args:
        tile_dir: 切片目录
        output_mosaic_path: 输出路径
        original_image_path: 原始大图路径（用于获取尺寸）
        target_shape: 目标尺寸（当没有原始大图时使用）
        vote_threshold: 投票阈值，默认0.5（超过50%的投票才接受）
    """
    # 确定画布尺寸
    if original_image_path:
        with rasterio.open(original_image_path) as src:
            height, width = src.height, src.width
            profile = src.profile
        print(f"原始大图尺寸: {width} x {height}")
    elif target_shape:
        height, width = target_shape
        profile = None
    else:
        raise ValueError("必须提供 original_image_path 或 target_shape")

    # 创建计数器和总覆盖次数
    vote_count = np.zeros((height, width), dtype=np.int32)  # 预测为长城的次数
    total_coverage = np.zeros((height, width), dtype=np.int32)  # 总覆盖次数

    # 获取所有切片
    tile_files = glob.glob(os.path.join(tile_dir, "*.png"))
    if len(tile_files) == 0:
        print(f"警告: 在 {tile_dir} 中未找到切片文件")
        return np.zeros((height, width), dtype=np.uint8)

    print(f"找到 {len(tile_files)} 个切片，开始拼接（投票机制，阈值={vote_threshold}）...")

    success_count = 0
    for tile_file in tile_files:
        try:
            # 读取切片
            tile_slice = np.array(Image.open(tile_file))

            # 确保二值化
            if tile_slice.max() > 1:
                tile_binary = (tile_slice > 0).astype(np.uint8)
            else:
                tile_binary = tile_slice

            # 解析位置
            start_row, start_col = parse_filename_for_position(tile_file)

            h, w = tile_binary.shape
            end_row = min(start_row + h, height)
            end_col = min(start_col + w, width)

            if start_row < height and start_col < width:
                # 实际粘贴的区域大小
                paste_h = end_row - start_row
                paste_w = end_col - start_col

                # 更新计数器
                vote_count[start_row:end_row, start_col:end_col] += tile_binary[:paste_h, :paste_w]
                total_coverage[start_row:end_row, start_col:end_col] += 1
                success_count += 1

        except Exception as e:
            print(f"跳过文件 {os.path.basename(tile_file)}: {e}")
            continue

    print(f"成功拼接 {success_count}/{len(tile_files)} 个切片")

    # 避免除零
    total_coverage = np.maximum(total_coverage, 1)

    # 计算投票比例
    vote_ratio = vote_count / total_coverage

    # 根据阈值决定最终结果
    mosaic = (vote_ratio >= vote_threshold).astype(np.uint8)

    # 统计信息
    print(f"\n📊 投票统计:")
    print(
        f"  被覆盖的像素: {np.sum(total_coverage > 0)}/{height * width} ({np.sum(total_coverage > 0) / (height * width) * 100:.2f}%)")
    print(f"  平均投票比例: {vote_ratio.mean():.3f}")
    print(f"  阈值 {vote_threshold} 下长城像素: {np.sum(mosaic)}")

    # 保存结果
    if original_image_path and profile:
        profile.update({'count': 1, 'dtype': 'uint8', 'nodata': 0})
        with rasterio.open(output_mosaic_path, 'w', **profile) as dst:
            dst.write(mosaic * 255, 1)  # 转回0-255
        print(f"拼接完成 (带地理信息): {output_mosaic_path}")
    else:
        # 保存为PNG
        Image.fromarray(mosaic * 255).save(output_mosaic_path.replace('.tif', '.png'))
        print(f"拼接完成 (PNG): {output_mosaic_path}")

    return mosaic


# ==================== 指标计算 ====================

def calculate_metrics(pred, gt):
    """计算所有评估指标 (参考 testtotal.py)"""
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    TP = np.logical_and(pred_bool, gt_bool).sum()
    FP = np.logical_and(pred_bool, np.logical_not(gt_bool)).sum()
    FN = np.logical_and(np.logical_not(pred_bool), gt_bool).sum()
    TN = np.logical_and(np.logical_not(pred_bool), np.logical_not(gt_bool)).sum()

    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    return {
        'iou': iou,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'accuracy': accuracy,
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'TN': int(TN)
    }


# ==================== 可视化 ====================

def visualize_evaluation(pred, gt, metrics, save_path):
    """可视化评估结果 (参考 testtotal.py)"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    if USE_CHINESE:
        title_pred = '预测结果'
        title_gt = '真实标签'
        title_diff = '差异图 (红:误检, 蓝:漏检)'
        title_overlay = '重叠图 (绿:正确, 红:误检, 蓝:漏检)'
        title_metrics = '评估指标'
        title_cm = '混淆矩阵'
        bg_label = '背景'
        wall_label = '长城'
    else:
        title_pred = 'Prediction'
        title_gt = 'Ground Truth'
        title_diff = 'Difference (Red:FP, Blue:FN)'
        title_overlay = 'Overlay (Green:TP, Red:FP, Blue:FN)'
        title_metrics = 'Metrics'
        title_cm = 'Confusion Matrix'
        bg_label = 'BG'
        wall_label = 'Wall'

    # 1. 预测图
    axes[0, 0].imshow(pred, cmap='gray')
    axes[0, 0].set_title(title_pred)
    axes[0, 0].axis('off')

    # 2. 真值图
    axes[0, 1].imshow(gt, cmap='gray')
    axes[0, 1].set_title(title_gt)
    axes[0, 1].axis('off')

    # 3. 差异图
    axes[0, 2].set_title(title_diff)
    diff = np.zeros((*pred.shape, 3), dtype=np.uint8)
    diff[(pred == 1) & (gt == 0)] = [255, 0, 0]   # FP: 红色
    diff[(pred == 0) & (gt == 1)] = [0, 0, 255]   # FN: 蓝色
    axes[0, 2].imshow(diff)
    axes[0, 2].axis('off')

    # 4. 重叠图
    axes[1, 0].set_title(title_overlay)
    overlay = np.zeros((*pred.shape, 3), dtype=np.uint8)
    overlay[..., 1] = ((pred == 1) & (gt == 1)) * 255  # TP: 绿色
    overlay[..., 0] = ((pred == 1) & (gt == 0)) * 255  # FP: 红色
    overlay[..., 2] = ((pred == 0) & (gt == 1)) * 255  # FN: 蓝色
    axes[1, 0].imshow(overlay)
    axes[1, 0].axis('off')

    # 5. 指标文本
    axes[1, 1].axis('off')
    metrics_text = "\n".join([
        f"IoU: {metrics['iou']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"F1: {metrics['f1']:.4f}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        "",
        f"TP: {metrics['TP']}",
        f"FP: {metrics['FP']}",
        f"FN: {metrics['FN']}",
        f"TN: {metrics['TN']}"
    ])
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                    transform=axes[1, 1].transAxes)

    # 6. 混淆矩阵
    axes[1, 2].set_title(title_cm)
    cm = np.array([[metrics['TN'], metrics['FP']],
                   [metrics['FN'], metrics['TP']]])

    cm_norm = cm / cm.max() if cm.max() > 0 else cm
    im = axes[1, 2].imshow(cm_norm, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, ax=axes[1, 2])

    for i in range(2):
        for j in range(2):
            axes[1, 2].text(j, i, str(cm[i, j]),
                           ha='center', va='center',
                           fontsize=12, color='black')

    axes[1, 2].set_xticks([0, 1])
    axes[1, 2].set_yticks([0, 1])
    axes[1, 2].set_xticklabels([bg_label, wall_label])
    axes[1, 2].set_yticklabels([bg_label, wall_label])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"评估可视化已保存: {save_path}")


# ==================== 主函数 ====================

def main():
    """主函数：推理 + 拼接 + 评估 (无后处理)"""

    # =================== 基础路径配置 ===================
    PROJECT_ROOT = config.PROJECT_ROOT

    MODEL_PATH = str(config.SAVE_PATH / "SAM2FPN-26.pth")
    TEST_IMAGE_DIR = str(PROJECT_ROOT / "data" / "dataset" / "test" / "images")
    MASK_DIR = str(PROJECT_ROOT / "data" / "dataset" / "test" / "gt")
    OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "evaluation_results")
    ORIGINAL_TIF_PATH = str(PROJECT_ROOT / "data" / "original" / "test.tif")

    # =================== 推理参数 ===================
    BATCH_SIZE = 4
    THRESHOLD = 0.1# 二值化阈值
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =================== 创建输出目录 ===================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sam2_eval_{timestamp}"
    RUN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, run_name)
    INFER_OUTPUT_DIR = os.path.join(RUN_OUTPUT_DIR, "inference_tiles")

    os.makedirs(INFER_OUTPUT_DIR, exist_ok=True)

    # 拼接输出路径
    FULL_PRED_PATH = os.path.join(RUN_OUTPUT_DIR, "full_prediction.tif")
    FULL_GT_PATH = os.path.join(RUN_OUTPUT_DIR, "full_gt.tif")

    print("=" * 70)
    print("SAM2 长城提取 - 测试与评估系统 (无后处理版)")
    print("=" * 70)
    print(f"设备: {DEVICE}")
    print(f"输出目录: {RUN_OUTPUT_DIR}")
    print(f"二值化阈值: {THRESHOLD}")
    print(f"中文显示: {'支持' if USE_CHINESE else '不支持'}")
    print("-" * 70)

    # =================== 数据预处理 ===================
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # =================== 数据集和数据加载器 ===================
    dataset = SAM2TestDataset(image_dir=TEST_IMAGE_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # =================== 初始化模型 ===================
    # 加载模型
    print("\n加载模型...")
    model = SAM2GWNet(str(config.HIERA_PATH))

    # 加载训练好的权重
    # 直接使用硬编码的路径
    checkpoint = torch.load(MODEL_PATH)  # 用 MODEL_PATH 而不是 args.checkpoint

    # 兼容新旧两种格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 新格式（包含完整信息）
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            print(f"✅ 加载新格式检查点 (epoch {checkpoint['epoch']})")
        else:
            print("✅ 加载新格式检查点")
    else:
        # 旧格式（直接是state_dict）
        model.load_state_dict(checkpoint)
        print("✅ 加载旧格式检查点")

    model.to(DEVICE)
    model.eval()

    # =================== 步骤1: 推理 ===================
    print("\n" + "=" * 70)
    print("步骤1: 开始推理...")

    inference_stats = {
        'total_images': len(dataset),
        'processed': 0,
        'avg_prob_min': 0,
        'avg_prob_max': 0
    }

    with torch.no_grad():
        for batch_idx, (images, img_paths) in enumerate(dataloader):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            inference_stats['avg_prob_min'] += probs.min().item()
            inference_stats['avg_prob_max'] += probs.max().item()

            preds = (probs > THRESHOLD).float()

            for i in range(len(img_paths)):
                img_path = img_paths[i]
                base_name = os.path.splitext(os.path.basename(img_path))[0]

                pred_np = preds[i].squeeze().cpu().numpy()

                # 【无后处理】直接保存
                pred_uint8 = (pred_np * 255).astype(np.uint8)

                # 保存为PNG
                output_path = os.path.join(INFER_OUTPUT_DIR, f"{base_name}.png")
                Image.fromarray(pred_uint8).save(output_path)

                inference_stats['processed'] += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  进度: {batch_idx + 1}/{len(dataloader)} 批次")

    inference_stats['avg_prob_min'] /= len(dataloader)
    inference_stats['avg_prob_max'] /= len(dataloader)

    print(f"推理完成，处理了 {inference_stats['processed']} 张图像")
    print(f"概率范围: [{inference_stats['avg_prob_min']:.4f}, {inference_stats['avg_prob_max']:.4f}]")

    # =================== 步骤2: 拼接预测结果 ===================
    print("\n" + "=" * 70)
    print("步骤2: 拼接预测结果...")
    mosaic_tiles(INFER_OUTPUT_DIR, FULL_PRED_PATH, original_image_path=ORIGINAL_TIF_PATH)

    # =================== 步骤3: 拼接真实标签 ===================
    print("\n" + "=" * 70)
    print("步骤3: 拼接真实标签...")
    mosaic_tiles(MASK_DIR, FULL_GT_PATH, original_image_path=ORIGINAL_TIF_PATH)

    # =================== 步骤4: 全局评估 ===================
    print("\n" + "=" * 70)
    print("步骤4: 全局评估...")

    with rasterio.open(FULL_PRED_PATH) as src:
        pred_full = src.read(1)
    with rasterio.open(FULL_GT_PATH) as src:
        gt_full = src.read(1)

    # 确保尺寸一致
    if pred_full.shape != gt_full.shape:
        print(f"形状不匹配，进行裁剪: pred {pred_full.shape}, gt {gt_full.shape}")
        min_h = min(pred_full.shape[0], gt_full.shape[0])
        min_w = min(pred_full.shape[1], gt_full.shape[1])
        pred_full = pred_full[:min_h, :min_w]
        gt_full = gt_full[:min_h, :min_w]

    # 确保二值化
    pred_full = (pred_full > 0).astype(np.uint8)
    gt_full = (gt_full > 0).astype(np.uint8)

    metrics = calculate_metrics(pred_full, gt_full)

    print("\n" + "=" * 70)
    print("全局评估结果")
    print("=" * 70)
    print(f"IoU:           {metrics['iou']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"F1-Score:      {metrics['f1']:.4f}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print("-" * 70)
    print(f"True Positives:  {metrics['TP']:>10,}")
    print(f"False Positives: {metrics['FP']:>10,}")
    print(f"False Negatives: {metrics['FN']:>10,}")
    print(f"True Negatives:  {metrics['TN']:>10,}")
    print("=" * 70)

    total_pixels = pred_full.size
    wall_pixels_gt = np.sum(gt_full)
    wall_pixels_pred = np.sum(pred_full)

    print(f"\n统计信息:")
    print(f"  总像素数: {total_pixels:,}")
    print(f"  真实长城像素: {wall_pixels_gt:,} ({wall_pixels_gt / total_pixels * 100:.2f}%)")
    print(f"  预测长城像素: {wall_pixels_pred:,} ({wall_pixels_pred / total_pixels * 100:.2f}%)")

    # =================== 计算切片平均指标 ===================
    print("\n" + "=" * 70)
    print("步骤5: 计算切片平均指标...")

    tile_files = glob.glob(os.path.join(INFER_OUTPUT_DIR, "*.png"))
    mask_files = glob.glob(os.path.join(MASK_DIR, "*.png"))

    print(f"找到 {len(tile_files)} 个预测切片")
    print(f"找到 {len(mask_files)} 个真实切片")

    slice_ious = []
    slice_recalls = []
    matched_count = 0

    for tile_file in tile_files:
        tile_basename = os.path.basename(tile_file)
        mask_file = os.path.join(MASK_DIR, tile_basename)

        if not os.path.exists(mask_file):
            continue

        matched_count += 1

        # 读取预测和真实切片
        pred_slice = np.array(Image.open(tile_file))
        gt_slice = np.array(Image.open(mask_file))

        # 确保二值化
        pred_slice = (pred_slice > 0).astype(np.uint8)
        gt_slice = (gt_slice > 0).astype(np.uint8)

        # 计算指标
        pred_bool = pred_slice.astype(bool)
        gt_bool = gt_slice.astype(bool)

        TP = np.logical_and(pred_bool, gt_bool).sum()
        FP = np.logical_and(pred_bool, np.logical_not(gt_bool)).sum()
        FN = np.logical_and(np.logical_not(pred_bool), gt_bool).sum()

        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        slice_ious.append(iou)
        slice_recalls.append(recall)

    if slice_ious:
        miou = np.mean(slice_ious)
        mrecall = np.mean(slice_recalls)
        std_iou = np.std(slice_ious)
        std_recall = np.std(slice_recalls)

        print("\n" + "=" * 50)
        print("切片平均指标")
        print("=" * 50)
        print(f"成功匹配的切片数量: {matched_count}/{len(tile_files)}")
        print(f"mIoU: {miou:.4f} ± {std_iou:.4f}")
        print(f"mRecall: {mrecall:.4f} ± {std_recall:.4f}")
        print("=" * 50)

        print("\n【关键指标】")
        print(f"mIoU = {miou:.4f}")
        print(f"mRecall = {mrecall:.4f}")
    else:
        print("\n错误: 没有找到匹配的切片对")

    # 保存可视化
    vis_path = os.path.join(RUN_OUTPUT_DIR, "evaluation_visualization.png")
    visualize_evaluation(pred_full, gt_full, metrics, vis_path)

    # 保存结果到JSON
    results = {
        'timestamp': timestamp,
        'threshold': THRESHOLD,
        'inference_stats': inference_stats,
        'metrics': {k: float(v) if isinstance(v, np.float64) else v for k, v in metrics.items()},
        'statistics': {
            'total_pixels': int(total_pixels),
            'wall_pixels_gt': int(wall_pixels_gt),
            'wall_pixels_pred': int(wall_pixels_pred),
            'wall_percentage_gt': float(wall_pixels_gt / total_pixels),
            'wall_percentage_pred': float(wall_pixels_pred / total_pixels)
        }
    }

    if slice_ious:
        results['slice_metrics'] = {
            'miou': float(miou),
            'mrecall': float(mrecall),
            'std_iou': float(std_iou),
            'std_recall': float(std_recall),
            'num_slices': len(slice_ious)
        }

    json_path = os.path.join(RUN_OUTPUT_DIR, "results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    txt_path = os.path.join(RUN_OUTPUT_DIR, "results.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("SAM2 长城提取评估结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"阈值: {THRESHOLD}\n")
        f.write("-" * 60 + "\n")
        f.write(f"IoU:        {metrics['iou']:.4f}\n")
        f.write(f"Recall:     {metrics['recall']:.4f}\n")
        f.write(f"Precision:  {metrics['precision']:.4f}\n")
        f.write(f"F1-Score:   {metrics['f1']:.4f}\n")
        f.write(f"Accuracy:   {metrics['accuracy']:.4f}\n")

        if slice_ious:
            f.write("-" * 60 + "\n")
            f.write("切片平均指标:\n")
            f.write(f"  mIoU:    {miou:.4f} ± {std_iou:.4f}\n")
            f.write(f"  mRecall: {mrecall:.4f} ± {std_recall:.4f}\n")
            f.write(f"  切片数量: {len(slice_ious)}\n")

        f.write("-" * 60 + "\n")
        f.write(f"TP: {metrics['TP']}\n")
        f.write(f"FP: {metrics['FP']}\n")
        f.write(f"FN: {metrics['FN']}\n")
        f.write(f"TN: {metrics['TN']}\n")
        f.write("=" * 60 + "\n")

    print(f"\n结果已保存至: {RUN_OUTPUT_DIR}")
    print(f"  - JSON: {json_path}")
    print(f"  - TXT: {txt_path}")
    print(f"  - 可视化: {vis_path}")
    print("\n✅ 全部完成！")


if __name__ == "__main__":
    main()
