"""Generate cross-satellite test tiles from all periods in 时序 folder."""
import os, sys, glob, logging
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
from PIL import Image

from core.tile_engine import compute_grid_params, get_tile_bounds, is_black_tile, find_valid_bbox

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = str(PROJECT_ROOT / "data" / "crosssat_raw")
OUTPUT_DIR = str(PROJECT_ROOT / "data" / "dataset_crosssat_test")
TEMP_DIR = os.path.join(OUTPUT_DIR, "_temp")
TILE_SIZE = 512
OVERLAP = 384

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "gt"), exist_ok=True)


def crop_black(tif_path, out_path):
    with rasterio.open(tif_path) as src:
        data = src.read()
        bbox = find_valid_bbox(data, nodata_value=0)
        if bbox is None:
            return False, None, None
        r_min, r_max, c_min, c_max = bbox
        window = Window.from_slices((r_min, r_max + 1), (c_min, c_max + 1))
        cropped = src.read(window=window)
        meta = src.meta.copy()
        meta.update({"height": cropped.shape[1], "width": cropped.shape[2],
                      "transform": src.window_transform(window)})
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(cropped)
        return True, window, cropped.shape


def create_mask(shp_path, ref_path, out_path):
    with rasterio.open(ref_path) as ref:
        meta = ref.meta.copy(); t = ref.transform; h, w = ref.height, ref.width
    gdf = gpd.read_file(shp_path)
    if gdf.crs != meta['crs']:
        gdf = gdf.to_crs(meta['crs'])
    geoms = [(g, 1) for g in gdf.geometry if g.is_valid and not g.is_empty]
    mask = rasterize(geoms, out_shape=(h, w), fill=0, default_value=1,
                     transform=t, dtype=np.uint8, all_touched=True)
    meta.update({"count": 1, "dtype": rasterio.uint8, "nodata": 0})
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(mask, 1)
    return True


def save_tile(data, path):
    img = data[:3].transpose(1, 2, 0)
    img = ((img - img.min()) / max(img.max() - img.min(), 1e-6) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def save_mask(data, path):
    mask = (data[0] > 0).astype(np.uint8) * 255
    Image.fromarray(mask).save(path)


def main():
    # Find all unique period prefixes (e.g., 11, 15, 17, ...)
    shp_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.shp")))
    periods = []
    for shp in shp_files:
        base = os.path.splitext(os.path.basename(shp))[0]
        periods.append(base)

    logger.info(f"Found {len(periods)} periods: {periods}")

    total_fg = 0
    for period in periods:
        tif_path = os.path.join(DATA_DIR, f"{period}.tif")
        shp_path = os.path.join(DATA_DIR, f"{period}.shp")
        if not os.path.exists(tif_path):
            logger.warning(f"TIF not found for {period}, skipping")
            continue

        logger.info(f"\n{'='*50}\nProcessing: {period}")

        # Crop
        temp_crop = os.path.join(TEMP_DIR, f"{period}_crop.tif")
        ok, _, shape = crop_black(tif_path, temp_crop)
        if not ok:
            continue
        logger.info(f"  Cropped: {shape[2]}x{shape[1]}")

        # Mask
        temp_mask = os.path.join(TEMP_DIR, f"{period}_mask.tif")
        create_mask(shp_path, temp_crop, temp_mask)

        # Tile
        with rasterio.open(temp_crop) as img_src, rasterio.open(temp_mask) as msk_src:
            w, h = img_src.width, img_src.height
            grid = compute_grid_params(w, h, TILE_SIZE, OVERLAP)
            rows, cols, stride = grid['rows'], grid['cols'], grid['stride']
            counter = 0; fg = 0

            for r in range(rows):
                for c in range(cols):
                    sr, sc = get_tile_bounds(r, c, stride, h, w, TILE_SIZE)
                    window = Window(sc, sr, TILE_SIZE, TILE_SIZE)
                    tile_img = img_src.read(window=window)
                    tile_mask = msk_src.read(window=window)
                    if is_black_tile(tile_img):
                        continue

                    fname = f"{period}_{counter:06d}_{sr}_{sc}.png"
                    save_tile(tile_img, os.path.join(OUTPUT_DIR, "images", fname))
                    save_mask(tile_mask, os.path.join(OUTPUT_DIR, "gt", fname))
                    counter += 1
                    if np.sum(tile_mask > 0) > 0:
                        fg += 1

            logger.info(f"  {counter} tiles ({fg} with wall)")
            total_fg += fg

    n_img = len(os.listdir(os.path.join(OUTPUT_DIR, "images")))
    n_gt = len(os.listdir(os.path.join(OUTPUT_DIR, "gt")))
    logger.info(f"\nDone! {n_img} images, {n_gt} gt, {total_fg} with wall")
    logger.info(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

