"""Generate tiles from left.tif/right.tif and split into train/val/test by spatial blocks.

70/15/15 split with 512px buffer gaps between regions.
Each TIF contributes tiles to all three sets proportionally.
"""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os, sys, logging
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from core.tile_engine import compute_grid_params, get_tile_bounds, is_black_tile, find_valid_bbox

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
DATA_DIR = str(PROJECT_ROOT / "data" / "raw_split")
OUTPUT_DIR = str(PROJECT_ROOT / "data" / "dataset_split")
TEMP_DIR = str(PROJECT_ROOT / "outputs" / "temp_crop_split")
TILE_SIZE = 512
OVERLAP = 384
STRIDE = TILE_SIZE - OVERLAP  # 128
GAP = 512  # px buffer between train/val/test regions

TIF_FILES = ["left.tif", "right.tif"]
SHAPEFILE = os.path.join(DATA_DIR, "shp.shp")

# Split proportions
TRAIN_PCT = 0.70
VAL_PCT = 0.10
TEST_PCT = 0.20

os.makedirs(TEMP_DIR, exist_ok=True)


def crop_to_valid_region(input_path, output_path):
    with rasterio.open(input_path) as src:
        full_data = src.read()
        bbox = find_valid_bbox(full_data, nodata_value=0)
        if bbox is None:
            return False, None, None
        row_min, row_max, col_min, col_max = bbox
        window = Window.from_slices((row_min, row_max + 1), (col_min, col_max + 1))
        cropped = src.read(window=window)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": cropped.shape[1], "width": cropped.shape[2],
            "transform": src.window_transform(window),
        })
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(cropped)
        return True, window, cropped.shape


def create_mask(shapefile_path, ref_raster_path, output_path):
    with rasterio.open(ref_raster_path) as ref:
        out_meta = ref.meta.copy()
        transform = ref.transform
        height, width = ref.height, ref.width
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs != out_meta['crs']:
        gdf = gdf.to_crs(out_meta['crs'])
    valid_geoms = [(g, 1) for g in gdf.geometry if g.is_valid and not g.is_empty]
    mask = rasterize(valid_geoms, out_shape=(height, width), fill=0, default_value=1,
                     transform=transform, dtype=np.uint8, all_touched=True)
    out_meta.update({"count": 1, "dtype": rasterio.uint8, "nodata": 0})
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(mask, 1)
    return True


def normalize_and_save_tile(data, path):
    img = data[:3].transpose(1, 2, 0)
    img = ((img - img.min()) / max(img.max() - img.min(), 1e-6) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def normalize_and_save_mask(data, path):
    mask = (data[0] > 0).astype(np.uint8) * 255
    Image.fromarray(mask).save(path)


def get_split_label(start_col, tif_width):
    """Assign train/val/test based on start_col with gaps."""
    train_end = int(TRAIN_PCT * tif_width)
    val_start = train_end + GAP
    val_end = int((TRAIN_PCT + VAL_PCT) * tif_width)
    test_start = val_end + GAP

    if start_col < train_end:
        return "train"
    elif val_start <= start_col < val_end:
        return "val"
    elif start_col >= test_start:
        return "test"
    else:
        return "gap"  # discard


def generate_tiles(cropped_path, mask_path, source_name):
    """Generate tiles and assign to train/val/test."""
    with rasterio.open(cropped_path) as img_src, rasterio.open(mask_path) as msk_src:
        width, height = img_src.width, img_src.height
        grid = compute_grid_params(width, height, TILE_SIZE, OVERLAP)
        rows, cols, stride = grid['rows'], grid['cols'], grid['stride']

    counts = {"train": 0, "val": 0, "test": 0, "gap": 0}
    fg_counts = {"train": 0, "val": 0, "test": 0}

    img_out = {s: os.path.join(OUTPUT_DIR, s, "images") for s in ["train", "val", "test"]}
    gt_out = {s: os.path.join(OUTPUT_DIR, s, "gt") for s in ["train", "val", "test"]}
    for d in list(img_out.values()) + list(gt_out.values()):
        os.makedirs(d, exist_ok=True)

    with rasterio.open(cropped_path) as img_src, rasterio.open(mask_path) as msk_src:
        for r in range(rows):
            for c in range(cols):
                sr, sc = get_tile_bounds(r, c, stride, height, width, TILE_SIZE)
                window = Window(sc, sr, TILE_SIZE, TILE_SIZE)
                tile_img = img_src.read(window=window)
                tile_mask = msk_src.read(window=window)
                if is_black_tile(tile_img):
                    continue

                split = get_split_label(sc, width)
                if split == "gap":
                    counts["gap"] += 1
                    continue

                fname = f"{source_name}_{counts[split]:06d}_{sr}_{sc}.png"
                normalize_and_save_tile(tile_img, os.path.join(img_out[split], fname))
                normalize_and_save_mask(tile_mask, os.path.join(gt_out[split], fname))
                counts[split] += 1
                if np.sum(tile_mask > 0) > 0:
                    fg_counts[split] += 1

    return counts, fg_counts


def main():
    for tif_name in TIF_FILES:
        tif_path = os.path.join(DATA_DIR, tif_name)
        base = os.path.splitext(tif_name)[0]
        logger.info(f"\n{'='*50}\nProcessing: {tif_name}\n{'='*50}")

        # Step 1: Crop black borders
        temp_cropped = os.path.join(TEMP_DIR, f"{base}_cropped.tif")
        ok, _, shape = crop_to_valid_region(tif_path, temp_cropped)
        if not ok:
            logger.error(f"Failed to crop {tif_name}")
            continue
        logger.info(f"Cropped: {shape[2]}x{shape[1]}")

        # Step 2: Create mask
        temp_mask = os.path.join(TEMP_DIR, f"{base}_mask.tif")
        create_mask(SHAPEFILE, temp_cropped, temp_mask)
        logger.info("Mask created")

        # Step 3: Generate tiles with split
        counts, fg = generate_tiles(temp_cropped, temp_mask, base)
        total = sum(counts.values())
        logger.info(f"Tiles: {total} (train={counts['train']}, val={counts['val']}, "
                    f"test={counts['test']}, gap={counts['gap']})")
        logger.info(f"Foreground: train={fg['train']}, val={fg['val']}, test={fg['test']}")

    # Final summary
    logger.info("\n" + "=" * 50)
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(OUTPUT_DIR, split, "images")
        gt_dir = os.path.join(OUTPUT_DIR, split, "gt")
        n_img = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
        n_gt = len(os.listdir(gt_dir)) if os.path.exists(gt_dir) else 0
        logger.info(f"{split}: {n_img} images, {n_gt} gt")
    logger.info("Done!")


if __name__ == "__main__":
    main()
