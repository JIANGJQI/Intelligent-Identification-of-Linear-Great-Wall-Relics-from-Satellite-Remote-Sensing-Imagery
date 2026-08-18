import os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_grid_params(width, height, tile_size, overlap):
    stride = tile_size - overlap
    rows = max(1, (height - tile_size) // stride + 2)
    cols = max(1, (width - tile_size) // stride + 2)
    return {'rows': rows, 'cols': cols, 'stride': stride, 'tile_size': tile_size,
            'width': width, 'height': height}


def get_tile_bounds(r, c, stride, height, width, tile_size):
    start_row = min(r * stride, height - tile_size)
    start_col = min(c * stride, width - tile_size)
    start_row = max(0, start_row)
    start_col = max(0, start_col)
    return start_row, start_col


def is_black_tile(tile_data):
    return np.all(tile_data == 0)


def normalize_and_save_tile(tile_data, output_path):
    img_array = np.transpose(tile_data, (1, 2, 0))
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)
    Image.fromarray(img_array).save(output_path)


def normalize_and_save_mask(tile_data, output_path):
    mask_array = tile_data[0]
    if mask_array.max() <= 1.0:
        mask_array = (mask_array * 255).astype(np.uint8)
    else:
        mask_array = mask_array.astype(np.uint8)
    Image.fromarray(mask_array).save(output_path)


def find_valid_bbox(data_array, nodata_value=0):
    if np.all(data_array == nodata_value):
        return None
    valid_mask = np.any(data_array != nodata_value, axis=0)
    valid_coords = np.where(valid_mask)
    if valid_coords[0].size == 0:
        return None
    row_min, row_max = valid_coords[0].min(), valid_coords[0].max()
    col_min, col_max = valid_coords[1].min(), valid_coords[1].max()
    return row_min, row_max, col_min, col_max
