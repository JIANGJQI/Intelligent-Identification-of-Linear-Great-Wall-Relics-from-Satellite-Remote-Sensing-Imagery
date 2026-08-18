# Intelligent Identification of Linear Great Wall Relics from Satellite Remote Sensing Imagery

This repository is a cleaned public code version of an undergraduate innovation project titled "Intelligent Identification of Linear Great Wall Relics from Satellite Remote Sensing Imagery". It is intended as a compact project portfolio repository for academic communication, showing the core model design, training workflow, inference pipeline, and ablation structure.

The project explores how a SAM2 Hiera image encoder can be combined with lightweight decoder modules for long, thin, width-varying heritage structures. The final version introduces Thick-DSConv-related modules and a SnakeBridge-style connection to improve boundary continuity and thin-structure representation.

> Public scope: code and configuration templates only. Private datasets, pretrained weights, trained checkpoints, raster imagery, and prediction outputs are intentionally excluded.

## Project Highlights

- Adapts SAM2-style visual features for remote-sensing segmentation of linear cultural heritage targets.
- Implements a final Thick-DSConv / SnakeBridge model variant for thin and discontinuous structures.
- Provides a cleaned training, inference, evaluation, and ablation workflow.
- Keeps local data paths, model weights, and generated outputs outside the public repository.

## Structure

```text
core/                  Dataset, logging, and tiling utilities
models/                SAM2-GreatWall base model and final v3 Thick-DSConv model
utils/                 Losses, Thick-DSConv, and helper utilities
scripts/train/         Final training entry point
scripts/inference/     Prediction, evaluation, context inference, and comparison tools
scripts/data/          Raster tiling and dataset preparation scripts
ablation/              Ablation training/evaluation code
sam2/                  Vendored SAM2 model code used by the project
sam2_configs/          SAM2 configuration files
configs/               Example configuration
config.py              Relative-path default configuration
requirements.txt       Python dependency snapshot
THIRD_PARTY_NOTICE.md  Third-party code/source notice
```

## Main Entry Points

- `scripts/train/train_v3.py`: final Thick-DSConv / SnakeBridge training script.
- `models/sam2gw_net_v3.py`: final model with `SnakeBridge` and `ThickSkipBlock`.
- `utils/dsconv.py`: Thick Dynamic Snake Convolution implementation.
- `scripts/inference/predict.py`: tiled inference and mosaic assembly.
- `scripts/inference/evaluate.py`: prediction evaluation and visualization.
- `ablation/train_ablation.py`: ablation models and training pipeline.

## Data and Weights

Place local-only assets in ignored folders:

```text
data/       training, validation, test, and cross-satellite tiles
weights/    SAM2 pretrained weights, e.g. sam2_hiera_large.pt
outputs/    trained checkpoints, logs, predictions, and visual results
```

The repository does not include original `.tif` imagery, generated tiles, `.pth/.pt/.ckpt` weights, or prediction outputs. This keeps the public version lightweight and avoids redistributing private or large research assets.

## Typical Workflow

```powershell
python scripts/data/prepare_split_dataset.py
python scripts/train/train_v3.py --hiera_path weights/sam2_hiera_large.pt
python scripts/inference/predict.py
python scripts/inference/evaluate.py
```

Adjust `config.py` for local paths before running. `configs/config.example.py` is kept as a clean reference copy.

## Notes

This repository is designed as a readable portfolio version of the project rather than a fully packaged release with public data. To reproduce training or inference, users need to provide their own compatible raster tiles, masks, and SAM2 pretrained weights.
