"""Run ablation variants sequentially."""
from pathlib import Path
import subprocess
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = PROJECT_ROOT / "outputs" / "ablation" / "checkpoints"

# (variant, epochs, resume_if_exists)
variants = [
    ("deeplabv3plus", 40, False),
    ("unetformer", 40, False),
    ("baseline", 40, False),
    ("plus_x3", 40, False),
    ("plus_x4", 40, False),
]

for variant, epochs, do_resume in variants:
    print()
    print("=" * 60)
    print(f"Running: {variant} (epochs={epochs})")
    print("=" * 60)
    cmd = [sys.executable, "ablation/train_ablation.py", "--variant", variant, "--epoch", str(epochs)]
    if do_resume:
        latest = CKPT_DIR / variant / f"{variant}-latest.pth"
        if latest.exists():
            cmd += ["--resume", str(latest)]
            print(f"  Resuming from: {latest}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"[FAIL] {variant} exited with code {result.returncode}")
        break
    print(f"[DONE] {variant}")

print()
print("All done.")
