#!/usr/bin/env python3
"""Fail-fast GPU-only runner for the v9 AASIST voice experiment."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--partial_fake_count", type=int, default=1200)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    experiment = pathlib.Path("experiments/v9")
    experiment.mkdir(parents=True, exist_ok=True)
    cuda = bool(torch.cuda.is_available())
    xpu = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    device = "cuda" if cuda else "xpu" if xpu else None
    environment = {
        "torch": torch.__version__, "cuda_available": cuda,
        "cuda_device_count": int(torch.cuda.device_count()),
        "xpu_available": xpu,
        "xpu_device_count": int(torch.xpu.device_count()) if hasattr(torch, "xpu") else 0,
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(0) if cuda else
            torch.xpu.get_device_name(0) if xpu else None
        ),
        "final_holdout": "PROHIBITED / NOT RUN",
    }
    (experiment / "gpu_environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    if device is None:
        raise SystemExit(
            "A CUDA or Intel XPU GPU is required for the v9 AASIST run. Environment report was written; "
            "no CPU fallback and no fake benchmark were performed."
        )

    command = [
        sys.executable, "-m", "src.train",
        "--manifest", "data/splits_v9_candidate/manifest.csv",
        "--splits_dir", "data/splits_v9_candidate",
        "--task", "voice", "--backbone", "aasist", "--device", device,
        "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
        "--num_workers", "4",
        "--base_channels", str(args.base_channels), "--lr", "0.0003",
        "--augmentation_profile", "voice_channel_v9",
        "--partial_fake_count", str(args.partial_fake_count),
        "--aux_eval_interval", "1",
        "--save_path", "model/candidates/voice_aasist_v9.pt",
        "--history_path", "experiments/v9/voice_aasist_training_curve.csv",
    ]
    (experiment / "aasist_command.json").write_text(
        json.dumps({"command": command}, indent=2) + "\n", encoding="utf-8"
    )
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
