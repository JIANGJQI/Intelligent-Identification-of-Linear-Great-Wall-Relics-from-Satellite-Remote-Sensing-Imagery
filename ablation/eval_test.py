"""Evaluate on test set — stitch per-TIF sub-image, accumulate per-pixel metrics."""
import os, sys, re, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from PIL import Image
from ablation.train_ablation import AblationModel
import ablation.config_ablation as config

DEVICE = torch.device("cuda")

# ── Inference hyperparameters (tune here) ──
SIGMOID_THRESHOLD = 0.7   # binary threshold for prediction
VOTE_THRESHOLD = 0.3      # mosaic voting ratio
BATCH_SIZE = 8            # inference batch size

# ── Test set selection ──
# Set to True for cross-satellite generalization test
USE_CROSS_SATELLITE = False


def get_tile_position(filename):
    m = re.search(r'_(\d+)_(\d+)\.png$', filename)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def stitch_tiles(tile_dir, tile_names, height, width):
    """Stitch tiles with majority vote. tile_dir: directory containing PNG files."""
    vote = np.zeros((height, width), dtype=np.int32)
    count = np.zeros((height, width), dtype=np.int32)
    for name in tile_names:
        tile = np.array(Image.open(os.path.join(tile_dir, name))) > 128
        sr, sc = get_tile_position(name)
        h, w = tile.shape
        er, ec = min(sr + h, height), min(sc + w, width)
        ph, pw = er - sr, ec - sc
        vote[sr:er, sc:ec] += tile[:ph, :pw]
        count[sr:er, sc:ec] += 1
    count = np.maximum(count, 1)
    return (vote / count >= VOTE_THRESHOLD).astype(np.uint8)


def evaluate_variant(variant, checkpoint, groups, img_dir, gt_dir):
    """groups: list of (sub_image_name, tile_names, H, W)"""
    ckpt = torch.load(checkpoint)
    model = AblationModel(config.HIERA_PATH, variant).to(DEVICE)
    try:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    except RuntimeError as e:
        print(f"  [SKIP] checkpoint incompatible: {str(e)[:80]}...")
        return None
    model.eval()

    # Read tiles directly (no random crop from FullDataset)
    all_names = sorted(os.listdir(img_dir))
    tmp_dir = os.path.join(config.CKPT_DIR, variant, f"_tmp_{hash(img_dir)}")
    os.makedirs(tmp_dir, exist_ok=True)

    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    for i in range(0, len(all_names), BATCH_SIZE):
        batch_names = all_names[i:i + BATCH_SIZE]
        batch = []
        for name in batch_names:
            img = np.array(Image.open(os.path.join(img_dir, name)).convert('RGB'))
            arr = img.astype(np.float32).transpose(2, 0, 1) / 255.0
            batch.append(arr)
        x = torch.from_numpy(np.stack(batch)).to(DEVICE)
        x = (x - mean) / std
        with torch.no_grad():
            pred = (torch.sigmoid(model(x)) > SIGMOID_THRESHOLD).cpu().numpy().astype(np.uint8)
        for j, name in enumerate(batch_names):
            Image.fromarray(pred[j, 0] * 255).save(os.path.join(tmp_dir, name))

    # Accumulate per sub-image
    total_inter = total_union = 0
    total_tp = total_fp = total_fn = 0
    for sub_name, tile_names, H, W in groups:
        gt = stitch_tiles(gt_dir, tile_names, H, W)
        pred = stitch_tiles(tmp_dir, tile_names, H, W)
        inter = (pred & gt).sum()
        union = (pred | gt).sum()
        tp = inter
        fp = pred.sum() - tp
        fn = gt.sum() - tp
        total_inter += inter; total_union += union
        total_tp += tp; total_fp += fp; total_fn += fn

    import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)

    iou = total_inter / max(total_union, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {'iou': iou, 'recall': recall, 'precision': precision, 'f1': f1,
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn}


def main():
    # Select test set
    if USE_CROSS_SATELLITE:
        img_dir = config.CROSSSAT_IMAGE_DIR
        gt_dir = config.CROSSSAT_GT_DIR
        label = "CROSS-SATELLITE"
    else:
        img_dir = config.TEST_IMAGE_DIR
        gt_dir = config.TEST_GT_DIR
        label = "IN-DOMAIN"

    # Group tiles by prefix
    gt_files = sorted(os.listdir(gt_dir))
    groups = {}
    for f in gt_files:
        prefix = f.split('_')[0]
        groups.setdefault(prefix, []).append(f)

    # Compute sub-image dimensions
    group_info = []
    for prefix, files in groups.items():
        max_r, max_c = 0, 0
        for f in files:
            r, c = get_tile_position(f)
            max_r = max(max_r, r + 512); max_c = max(max_c, c + 512)
        group_info.append((prefix, files, max_r, max_c))
        print(f"{prefix}: {len(files)} tiles, {max_c}x{max_r}")
    print(f"Total: {sum(len(f) for _, f, _, _ in group_info)} tiles across {len(group_info)} sub-images")

    variants = ["deeplabv3plus", "unetformer", "baseline", "plus_x3", "plus_x4", "full"]

    print(f"\n{'='*70}")
    print(f"Ablation Test [{label}] (stitch per sub-image, global accumulation)")
    print(f"{'='*70}")

    all_results = {}
    ckpt_dir = config.CKPT_DIR
    for v in variants:
        ckpt_path = os.path.join(ckpt_dir, v, f"{v}-best.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ckpt_dir, v, f"{v}-latest.pth")
        if not os.path.exists(ckpt_path):
            print(f"\n{v}: checkpoint not found, skipping")
            continue
        print(f"\nEvaluating {v} (latest)...")
        results = evaluate_variant(v, ckpt_path, group_info, img_dir, gt_dir)
        if results is not None:
            all_results[v] = results

    # Table
    print(f"\n{'='*78}")
    print(f"{'Variant':<16s} {'IoU':>7s} {'Recall':>7s} {'Prec':>7s} {'F1':>7s}  {'TP':>10s} {'FP':>10s} {'FN':>10s}")
    print(f"{'-'*78}")
    for v in all_results:
        r = all_results[v]
        print(f"{v:<16s} {r['iou']:>7.4f} {r['recall']:>7.4f} {r['precision']:>7.4f} {r['f1']:>7.4f}  "
              f"{r['tp']:>10,} {r['fp']:>10,} {r['fn']:>10,}")

    # Delta vs DeepLabV3+
    if 'deeplabv3plus' in all_results:
        dlv3 = all_results['deeplabv3plus']
        print(f"\n{'Variant':<16s} {'dIoU':>7s} {'dRecall':>7s} {'dF1':>7s}")
        print(f"{'-'*38}")
        for v in all_results:
            r = all_results[v]
            print(f"{v:<16s} {r['iou']-dlv3['iou']:>+7.4f} {r['recall']-dlv3['recall']:>+7.4f} "
                  f"{r['f1']-dlv3['f1']:>+7.4f}")


if __name__ == "__main__":
    main()
