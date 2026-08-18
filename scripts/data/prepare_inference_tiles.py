"""
多时相 TIF 影像切片工具 — 生成推理数据集
所有时相使用统一网格，保证切片位置完全对齐
"""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import re
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
import logging

from core.tile_engine import (compute_grid_params, get_tile_bounds,
                         is_black_tile, normalize_and_save_tile)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def split_raster_with_grid(image_path, output_dir, grid_params, filename_prefix=None):
    with rasterio.open(image_path) as src:
        rows, cols = grid_params['rows'], grid_params['cols']
        stride, tile_size = grid_params['stride'], grid_params['tile_size']
        width, height = grid_params['width'], grid_params['height']

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"处理影像: {width}x{height}, tile_size={tile_size}, stride={stride}")

        counter = 0
        for r in range(rows):
            for c in range(cols):
                start_row, start_col = get_tile_bounds(r, c, stride, height, width, tile_size)
                window = Window(start_col, start_row, tile_size, tile_size)
                tile_data = src.read(window=window)

                if is_black_tile(tile_data):
                    continue

                prefix = f"{filename_prefix}_" if filename_prefix else ""
                filename = f"{prefix}tile_{counter:06d}_{start_row}_{start_col}.png"
                normalize_and_save_tile(tile_data, os.path.join(output_dir, filename))
                counter += 1

                if counter % 100 == 0:
                    logger.info(f"已生成 {counter} 个切片")

        logger.info(f"完成！共生成 {counter} 个切片")
        return counter


def batch_process_phases(base_dir, output_base_dir, tile_size=352, overlap=264):
    pattern = os.path.join(base_dir, "**", "Level19", "*.tif")
    tif_files = glob.glob(pattern, recursive=True)

    if not tif_files:
        pattern = os.path.join(base_dir, "*", "*.tif")
        tif_files = glob.glob(pattern, recursive=True)

    logger.info(f"找到 {len(tif_files)} 个TIF文件")
    if not tif_files:
        logger.error("未找到TIF文件")
        return

    with rasterio.open(tif_files[0]) as src:
        grid_params = compute_grid_params(src.width, src.height, tile_size, overlap)

    total_tiles = 0
    for tif_path in tif_files:
        basename = os.path.basename(tif_path)
        match = re.search(r'(\d{4}-\d{2}-\d{2})', basename)
        if match:
            phase_name = match.group(1)
        else:
            phase_name = None
            for part in Path(tif_path).parts:
                if re.match(r'\d{4}-\d{2}-\d{2}', part):
                    phase_name = part
                    break
            if phase_name is None:
                phase_name = os.path.splitext(basename)[0]

        output_dir = os.path.join(output_base_dir, phase_name)
        logger.info(f"\n{'=' * 50}\n处理时相: {phase_name}\n输入: {tif_path}\n输出: {output_dir}")
        tiles = split_raster_with_grid(tif_path, output_dir, grid_params)
        total_tiles += tiles

    logger.info(f"\n{'=' * 50}\n全部完成！共处理 {len(tif_files)} 个时相，生成 {total_tiles} 个切片")


def main():
    base_dir = str(PROJECT_ROOT / "data" / "inference_raw")
    output_base_dir = str(PROJECT_ROOT / "data" / "dataset_change")
    batch_process_phases(base_dir, output_base_dir, tile_size=512, overlap=384)


if __name__ == "__main__":
    main()
