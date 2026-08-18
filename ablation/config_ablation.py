"""Ablation experiment configuration."""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "dataset_split"

TRAIN_IMAGE_DIR = str(DATASET_ROOT / "train" / "images") + os.sep
TRAIN_GT_DIR = str(DATASET_ROOT / "train" / "gt") + os.sep
VAL_IMAGE_DIR = str(DATASET_ROOT / "val" / "images") + os.sep
VAL_GT_DIR = str(DATASET_ROOT / "val" / "gt") + os.sep
TEST_IMAGE_DIR = str(DATASET_ROOT / "test" / "images") + os.sep
TEST_GT_DIR = str(DATASET_ROOT / "test" / "gt") + os.sep

HIERA_PATH = str(PROJECT_ROOT / "weights" / "sam2_hiera_large.pt")

TRAIN_CONFIG = {
    "epoch": 100,
    "lr": 1e-4,
    "batch_size": 2,
    "weight_decay": 5e-4,
    "crop_size": 512,
    "use_fp16": True,
    "num_workers": 0,
}

LOSS_WEIGHTS = {
    "bce_weight": 1.0,
    "dice_weight": 0.5,
    "cldice_weight": 0.1,
}

INFERENCE_CONFIG = {
    "batch_size": 8,
    "threshold": 0.5,
}

CROSSSAT_IMAGE_DIR = str(PROJECT_ROOT / "data" / "dataset_crosssat_test" / "images") + os.sep
CROSSSAT_GT_DIR = str(PROJECT_ROOT / "data" / "dataset_crosssat_test" / "gt") + os.sep

CKPT_DIR = str(PROJECT_ROOT / "outputs" / "ablation" / "checkpoints")
LOG_DIR = str(PROJECT_ROOT / "outputs" / "ablation" / "logs")
