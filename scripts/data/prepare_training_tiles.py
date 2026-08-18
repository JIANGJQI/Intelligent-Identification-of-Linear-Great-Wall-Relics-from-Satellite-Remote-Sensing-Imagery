"""
TIF + Shapefile → 训练切片生成工具
流程：裁剪黑边 → 创建二值掩膜 → 分割成训练瓦片
"""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
from PIL import Image
import logging
import traceback

from core.tile_engine import (compute_grid_params, get_tile_bounds,
                         is_black_tile, normalize_and_save_tile,
                         normalize_and_save_mask, find_valid_bbox)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def crop_to_valid_region(input_path, output_path, nodata_value=0):
    try:
        with rasterio.open(input_path) as src:
            full_data = src.read()
            bbox = find_valid_bbox(full_data, nodata_value=nodata_value)
            if bbox is None:
                logger.warning(f"无法找到有效数据: {input_path}")
                return False, None, None
            row_min, row_max, col_min, col_max = bbox
            window = Window.from_slices((row_min, row_max + 1), (col_min, col_max + 1))

            cropped_data = src.read(window=window)
            out_meta = src.meta.copy()
            out_meta.update({
                "height": int(cropped_data.shape[1]),
                "width": int(cropped_data.shape[2]),
                "transform": src.window_transform(window),
            })
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with rasterio.open(output_path, 'w', **out_meta) as dst:
                dst.write(cropped_data)
            logger.info(f"裁剪成功: {os.path.basename(input_path)} -> {cropped_data.shape[2]}x{cropped_data.shape[1]}")
            return True, window, cropped_data.shape
    except Exception as e:
        logger.error(f"裁剪失败 {input_path}: {e}")
        return False, None, None


def create_binary_mask_from_shapefile(shapefile_path, reference_raster_path, output_mask_path):
    try:
        with rasterio.open(reference_raster_path) as ref_src:
            out_meta = ref_src.meta.copy()
            transform = ref_src.transform
            height, width = ref_src.height, ref_src.width
            crs = ref_src.crs

        gdf = gpd.read_file(shapefile_path)

        if gdf.empty:
            logger.warning(f"Shapefile 为空: {shapefile_path}")
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            if gdf.crs != crs:
                gdf = gdf.to_crs(crs)

            valid_geoms = [(geom, 1) for geom in gdf.geometry if geom.is_valid and not geom.is_empty]
            if not valid_geoms:
                logger.warning(f"无有效几何体: {shapefile_path}")
                mask = np.zeros((height, width), dtype=np.uint8)
            else:
                mask = rasterize(
                    valid_geoms, out_shape=(height, width), fill=0, default_value=1,
                    transform=transform, dtype=np.uint8, all_touched=True
                )

        out_meta.update({"count": 1, "dtype": rasterio.uint8, "nodata": 0})
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        with rasterio.open(output_mask_path, 'w', **out_meta) as dst:
            dst.write(mask, 1)

        logger.info(f"掩膜创建成功: {os.path.basename(output_mask_path)}")
        return True
    except Exception as e:
        logger.error(f"创建掩膜失败 {shapefile_path}: {e}")
        return False


def create_background_mask(reference_raster_path, output_mask_path):
    try:
        with rasterio.open(reference_raster_path) as ref_src:
            out_meta = ref_src.meta.copy()
            height, width = ref_src.height, ref_src.width

        mask = np.zeros((height, width), dtype=np.uint8)
        out_meta.update({"count": 1, "dtype": rasterio.uint8, "nodata": 0})
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        with rasterio.open(output_mask_path, 'w', **out_meta) as dst:
            dst.write(mask, 1)

        logger.info(f"背景掩膜创建成功: {os.path.basename(output_mask_path)}")
        return True
    except Exception as e:
        logger.error(f"创建背景掩膜失败 {reference_raster_path}: {e}")
        return False


def split_into_tiles(image_path, mask_path, output_base_dir, source_name, tile_size=352, overlap=264):
    try:
        with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as msk_src:
            width, height = img_src.width, img_src.height
            grid = compute_grid_params(width, height, tile_size, overlap)
            rows, cols, stride = grid['rows'], grid['cols'], grid['stride']

            img_out_dir = os.path.join(output_base_dir, "train", "images")
            mask_out_dir = os.path.join(output_base_dir, "train", "gt")
            os.makedirs(img_out_dir, exist_ok=True)
            os.makedirs(mask_out_dir, exist_ok=True)

            counter = 0
            fg_count, bg_count = 0, 0

            for r in range(rows):
                for c in range(cols):
                    start_row, start_col = get_tile_bounds(r, c, stride, height, width, tile_size)
                    window = Window(start_col, start_row, tile_size, tile_size)

                    tile_img = img_src.read(window=window)
                    tile_mask = msk_src.read(window=window)

                    if is_black_tile(tile_img):
                        continue

                    filename = f"{source_name}_{counter:06d}_{start_row}_{start_col}.png"
                    normalize_and_save_tile(tile_img, os.path.join(img_out_dir, filename))
                    normalize_and_save_mask(tile_mask, os.path.join(mask_out_dir, filename))

                    counter += 1
                    if np.sum(tile_mask > 0) > 0:
                        fg_count += 1
                    else:
                        bg_count += 1

            logger.info(f"{source_name}: 生成 {counter} 个切片（前景: {fg_count}, 背景: {bg_count}）")
            return True
    except Exception as e:
        logger.error(f"分割失败 {image_path}: {e}")
        return False


def main():
    DATA_DIR = str(PROJECT_ROOT / "data" / "training_raw" / "Level19")
    OUTPUT_DIR = str(PROJECT_ROOT / "data" / "dataset")
    TEMP_DIR = str(PROJECT_ROOT / "outputs" / "temp_crop")

    TILE_SIZE = 512
    OVERLAP = 384

    os.makedirs(TEMP_DIR, exist_ok=True)

    tif_files = glob.glob(os.path.join(DATA_DIR, "*.tif"))
    tif_files = [f for f in tif_files if not f.endswith('.aux.xml')]

    logger.info(f"找到 {len(tif_files)} 个 TIF 文件")

    for tif_path in tif_files:
        base_name = os.path.splitext(os.path.basename(tif_path))[0]
        shapefile_path = os.path.join(DATA_DIR, f"{base_name}.shp")

        logger.info(f"\n{'=' * 60}\n处理: {base_name}\n  影像: {tif_path}\n{'=' * 60}")

        has_shapefile = os.path.exists(shapefile_path)
        if has_shapefile:
            logger.info(f"  标注: {shapefile_path} (包含长城标注)")
        else:
            logger.info(f"  标注: 无对应Shapefile，此影像全为背景")

        temp_cropped = os.path.join(TEMP_DIR, f"{base_name}_cropped.tif")
        temp_mask = os.path.join(TEMP_DIR, f"{base_name}_mask.tif")

        success, _, _ = crop_to_valid_region(tif_path, temp_cropped)
        if not success:
            continue

        if has_shapefile:
            success = create_binary_mask_from_shapefile(shapefile_path, temp_cropped, temp_mask)
        else:
            logger.info(f"为 {base_name} 创建全背景掩膜")
            success = create_background_mask(temp_cropped, temp_mask)

        if not success:
            continue

        split_into_tiles(temp_cropped, temp_mask, OUTPUT_DIR, f"train_{base_name}", TILE_SIZE, OVERLAP)

    logger.info(f"\n{'=' * 60}\n✅ 批量处理完成！\n输出目录: {OUTPUT_DIR}\n{'=' * 60}")


if __name__ == "__main__":
    main()
