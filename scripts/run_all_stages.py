#!/usr/bin/env python3
"""Reproducible split -> specialist training -> exact fusion calibration."""
import argparse
import pathlib
import subprocess
import sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import build_val_sets, scan_real_datasets


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--calibration_per_split", type=int, default=0,
                        help="0 uses the complete independent fusion calibration split")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_demucs", action="store_true")
    args = parser.parse_args()

    # Always regenerate: legacy pre-split MIX rows must never be reused.
    build_val_sets(scan_real_datasets(), out_dir="data/splits", random_state=42)
    common = [
        "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
        "--device", args.device,
    ]
    if args.use_demucs:
        common.append("--use_demucs")
    run([sys.executable, "-m", "src.train", "--task", "voice", "--backbone", "spec_cnn",
         "--save_path", "model/best.pt", *common])
    run([sys.executable, "-m", "src.train", "--task", "music", "--backbone", "spec_cnn",
         "--save_path", "model/music_best.pt", *common])
    run([sys.executable, "scripts/calibrate_fusion.py",
         "--per_split", str(args.calibration_per_split),
         "--batch_size", str(min(args.batch_size, 16)), "--device", args.device])


if __name__ == "__main__":
    main()
