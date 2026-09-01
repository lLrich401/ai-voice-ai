#!/usr/bin/env python3
"""Conservative CPU/GPU V13B probe; candidates can never overwrite TEST5."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_v13b_exploratory import HEADS, evaluate_checkpoint
from scripts.manage_v13b_stages import evaluate_gates
from src.dataset import AudioDataset
from src.models.beats_backbone import MusicMultitask
from src.train import specialist_sample_weights, train_one_epoch
from tools.v13_guards import assert_final_holdout_v13b_forbidden


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("music", "file"), required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=23674913)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)
    dataset_report = json.loads((ROOT / "data/splits_v13b/DATASET_V13B.json").read_text())
    shortcut = json.loads((ROOT / "experiments/v13b/source_shortcut_audit.json").read_text())
    policy = json.loads((ROOT / "configs/v13b/selection_policy.json").read_text())
    gates = evaluate_gates(dataset_report, shortcut, policy)
    if not gates["exploratory_allowed"]:
        raise RuntimeError(f"V13B exploratory training gate failed: {gates['exploratory_checks']}")
    train_path = ROOT / "data/splits_v13b/train.csv"
    validation_path = ROOT / "data/splits_v13b/val_generator_disjoint.csv"
    assert_final_holdout_v13b_forbidden(train_path, validation_path)
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial = torch.load(ROOT / "model/music_best.pt", map_location="cpu")
    model = MusicMultitask(base_channels=int(initial.get("base_channels", 32)))
    model.load_state_dict(initial["model"], strict=True)
    model = model.to(device)
    data = AudioDataset(train, sr=16_000, seg_sec=4.0, is_training=True,
                        use_demucs=False, task="music" if args.task == "music" else "multitask",
                        device=str(device), augmentation_profile="baseline")
    sampler = None
    if args.task == "music":
        weights = specialist_sample_weights(train, "music")
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(train), replacement=True)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=sampler is None,
                        sampler=sampler, num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    candidate_dir = ROOT / "model/candidates/v13b"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / f"{args.task}_spec_cnn_probe.pt"
    history = []
    started = time.perf_counter()
    loss_task = "music" if args.task == "music" else "multitask"
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, loader, optimizer, device, scaler, task=loss_task)
        history.append({"epoch": epoch + 1, "loss": loss})
        print(f"{args.task} epoch {epoch + 1}/{args.epochs} loss={loss:.6f}")
    checkpoint = {
        "model": model.state_dict(), "epoch": args.epochs - 1,
        "task": args.task, "backbone": "spec_cnn", "model_name": "MusicMultitask",
        "base_channels": int(initial.get("base_channels", 32)), "sample_rate": 16_000,
        "seg_sec": 4.0, "label_heads": list(HEADS), "use_demucs": False,
        "candidate_status": "NOT_ADOPTABLE_YET", "training_split": "V13B train only",
        "validation_split": "V13B generator-disjoint only", "source_disjoint": "NOT MEASURED",
        "final_holdout": "NOT READ / NOT RUN", "seed": args.seed, "lr": args.lr,
        "history": history,
    }
    torch.save(checkpoint, candidate_path)
    measured = evaluate_checkpoint(
        candidate_path, validation, device, f"V13B_{args.task.upper()}_CANDIDATE")
    report = {
        "status": "EXPLORATORY_TRAINED_NOT_ADOPTABLE", "task": args.task,
        "device": str(device), "candidate": candidate_path.relative_to(ROOT).as_posix(),
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "training_rows": len(train), "validation_rows": len(validation),
        "epochs": args.epochs, "history": history,
        "training_seconds": time.perf_counter() - started,
        "generator_disjoint_metrics": measured["metrics"],
        "source_disjoint": "NOT MEASURED", "final_holdout": "NOT RUN",
        "selected_artifacts_mutated": False, "decision": "KEEP_TEST5",
    }
    report_path = ROOT / f"experiments/v13b/{args.task}_exploratory_training.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
