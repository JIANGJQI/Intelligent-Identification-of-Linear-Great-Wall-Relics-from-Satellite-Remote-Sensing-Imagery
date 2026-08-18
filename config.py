"""Default configuration for SAM2-GreatWall.

Copy this file and adjust paths for your local data and pretrained weights.
Large datasets, checkpoints, and prediction outputs are intentionally excluded
from the public repository.
"""
from pathlib import Path
import glob

PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories expected by the training/inference scripts.
TRAIN_DATASET_ROOT = PROJECT_ROOT / "data" / "dataset_new"
DATASET_CHANGE_DIR = PROJECT_ROOT / "data" / "dataset_change"
CROSSSAT_DATASET_ROOT = PROJECT_ROOT / "data" / "dataset_crosssat_test"

# Pretrained SAM2 weights should be downloaded separately.
HIERA_PATH = PROJECT_ROOT / "weights" / "sam2_hiera_large.pt"

# Runtime outputs.
SAVE_PATH = PROJECT_ROOT / "outputs" / "checkpoints"
TRAIN_OUTPUT_DIR = SAVE_PATH
INFERENCE_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "predictions"
RESUME_CHECKPOINT = None
RESUME_EPOCH = 0

# Full-scene dimensions used by the original inference workflow.
IMAGE_HEIGHT = 10132
IMAGE_WIDTH = 43429

TRAIN_CONFIG = {
    "epoch": 100,
    "lr": 1e-4,
    "batch_size": 2,
    "weight_decay": 5e-4,
    "crop_size": 512,
    "use_fp16": True,
    "num_workers": 0,
}

INFERENCE_CONFIG = {
    "batch_size": 8,
    "threshold": 0.7,
    "vote_threshold": 0.5,
}

CONTEXT_INFERENCE_CONFIG = {
    "tile_size": 512,
    "stride": 128,
    "batch_size_encode": 8,
    "batch_size_decode": 4,
    "store_bridge_fp16": True,
    "neighbor_grid_size": 3,
}

LOSS_WEIGHTS = {
    "bce_weight": 1.0,
    "dice_weight": 0.5,
    "cldice_weight": 0.1,
    "perceptual_loss_weight": 0.0,
}

RESUME_CONFIG = {
    "resume": False,
    "warm_start": False,
    "resume_optimizer": True,
}


def discover_train_dirs():
    train_dirs = sorted(glob.glob(str(TRAIN_DATASET_ROOT / "train_*")))
    if not train_dirs:
        raise FileNotFoundError(f"No train_* folders found under {TRAIN_DATASET_ROOT}")
    image_dirs = [str(Path(d) / "images") + "/" for d in train_dirs]
    gt_dirs = [str(Path(d) / "gt") + "/" for d in train_dirs]
    return image_dirs, gt_dirs


def print_config():
    print("=" * 60)
    print("SAM2-GreatWall configuration")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Train dataset: {TRAIN_DATASET_ROOT}")
    print(f"Inference dataset: {DATASET_CHANGE_DIR}")
    print(f"SAM2 weights: {HIERA_PATH}")
    print(f"Checkpoint output: {SAVE_PATH}")
    print(f"Prediction output: {INFERENCE_OUTPUT_ROOT}")
    print("=" * 60)


def check_paths():
    missing = []
    if not HIERA_PATH.exists():
        missing.append(f"SAM2 pretrained weights: {HIERA_PATH}")
    if not TRAIN_DATASET_ROOT.exists():
        missing.append(f"training dataset root: {TRAIN_DATASET_ROOT}")
    if missing:
        print("Missing paths:")
        for item in missing:
            print(f"  - {item}")
        return False
    return True


if __name__ == "__main__":
    print_config()
    check_paths()
