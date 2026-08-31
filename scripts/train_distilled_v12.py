#!/usr/bin/env python3
"""Train one V12 SpecCNN student from frozen V7 and V9 teachers."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from src.dataset import AudioDataset
from src.distillation import (
    ExponentialMovingAverage, PairedGroupBatchSampler, component_teacher_loss,
    freeze_teacher, paired_margin_ranking_loss, source_balanced_weights,
    teacher_is_frozen, v7_retention_loss, V12CachedWaveDataset,
)
from src.ensemble import assert_final_holdout_forbidden
from src.models.aasist import AASISTMultitask
from src.models.beats_backbone import MusicMultitask
from src.train import HEADS, masked_multitask_loss


CONFIGS = {
    "M1": dict(task="music", supervised=1.0, retention=0.0, candidate=0.0,
               rank=0.0, temperature=1.0, ema=False, lr=1e-5, file_mode="freeze"),
    "M2": dict(task="music", supervised=0.8, retention=0.2, candidate=0.0,
               rank=0.0, temperature=1.0, ema=False, lr=3e-5, file_mode="lr_0.1"),
    "M3": dict(task="music", supervised=0.6, retention=0.25, candidate=0.15,
               rank=0.0, temperature=1.0, ema=False, lr=3e-5, file_mode="retention"),
    "M4": dict(task="music", supervised=0.6, retention=0.25, candidate=0.15,
               rank=0.0, temperature=2.0, ema=True, lr=3e-5, file_mode="retention"),
    "M5": dict(task="music", supervised=0.55, retention=0.25, candidate=0.20,
               rank=0.05, temperature=1.0, ema=False, lr=1e-5, file_mode="retention"),
    "V1": dict(task="voice", supervised=1.0, retention=0.0, candidate=0.0,
               rank=0.0, temperature=1.0, ema=False, lr=1e-5, file_mode="freeze"),
    "V2": dict(task="voice", supervised=0.8, retention=0.2, candidate=0.0,
               rank=0.0, temperature=1.0, ema=False, lr=3e-5, file_mode="lr_0.1"),
    "V3": dict(task="voice", supervised=0.6, retention=0.25, candidate=0.15,
               rank=0.0, temperature=1.0, ema=False, lr=3e-5, file_mode="retention"),
    "V4": dict(task="voice", supervised=0.6, retention=0.25, candidate=0.15,
               rank=0.0, temperature=2.0, ema=True, lr=3e-5, file_mode="retention"),
    "V5": dict(task="voice", supervised=0.55, retention=0.25, candidate=0.20,
               rank=0.05, temperature=1.0, ema=False, lr=1e-5, file_mode="retention"),
}


class IndexedAudioDataset(AudioDataset):
    def __getitem__(self, index):
        wave, labels, path = super().__getitem__(index)
        return wave, labels, path, index


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint_model(path: pathlib.Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    channels = int(checkpoint["base_channels"])
    if checkpoint["backbone"] == "spec_cnn":
        model = MusicMultitask(base_channels=channels)
    elif checkpoint["backbone"] == "aasist":
        model = AASISTMultitask(base_channels=channels)
    else:
        raise ValueError(f"unsupported backbone {checkpoint['backbone']}")
    if tuple(checkpoint.get("label_heads", ())) != HEADS:
        raise ValueError(f"head mismatch in {path}")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device), checkpoint


def make_optimizer(model: MusicMultitask, config: dict) -> torch.optim.Optimizer:
    file_parameters = list(model.heads["file_fake"].parameters())
    file_ids = {id(parameter) for parameter in file_parameters}
    if config["file_mode"] == "freeze":
        for parameter in file_parameters:
            parameter.requires_grad_(False)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(parameters, lr=config["lr"])
    main = [parameter for parameter in model.parameters() if id(parameter) not in file_ids]
    return torch.optim.AdamW([
        {"params": main, "lr": config["lr"]},
        {"params": file_parameters, "lr": config["lr"] * 0.1},
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default=(
        "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"))
    args = parser.parse_args()
    config = dict(CONFIGS[args.candidate])
    task = config["task"]
    split_path = ROOT / "data/splits_v12/train.csv"
    dataset_meta_path = ROOT / "data/splits_v12/DATASET_V12.json"
    output = ROOT / f"model/candidates/v12/{args.candidate.lower()}_student.pt"
    history_path = ROOT / f"experiments/v12/{args.candidate.lower()}_training.json"
    for path in (split_path, dataset_meta_path, output, history_path):
        assert_final_holdout_forbidden(path)
    seed = 20260901 + sum(map(ord, args.candidate))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device)
    frame = pd.read_csv(split_path)
    if not frame["data_role"].astype(str).eq("train_v12").all():
        raise RuntimeError("V12 training split contains non-training roles")

    selected_path = ROOT / ("model/best.pt" if task == "voice" else "model/music_best.pt")
    candidate_teacher_path = ROOT / (
        "model/candidates/voice_aasist_v9.pt" if task == "voice"
        else "model/candidates/music_spec_cnn_v9.pt")
    student, selected_checkpoint = load_checkpoint_model(selected_path, device)
    if selected_checkpoint["backbone"] != "spec_cnn":
        raise RuntimeError("V12 student must remain SpecCNN")
    teacher_cache_path = ROOT / f"experiments/v12/cache/teacher_{task}.csv"
    teacher_meta_path = teacher_cache_path.with_suffix(teacher_cache_path.suffix + ".meta.json")
    if not teacher_cache_path.is_file() or not teacher_meta_path.is_file():
        raise FileNotFoundError(
            f"build the frozen teacher cache first: {teacher_cache_path}")
    teacher_meta = json.loads(teacher_meta_path.read_text(encoding="utf-8"))
    expected_teacher_meta = {
        "task": task, "rows": len(frame), "split_sha256": sha256(split_path),
        "checkpoint_sha256": {
            "v7": sha256(selected_path), "v9": sha256(candidate_teacher_path)},
        "candidate_file_head_cached": False, "teachers_frozen": True,
    }
    stale = {key: (teacher_meta.get(key), value)
             for key, value in expected_teacher_meta.items()
             if teacher_meta.get(key) != value}
    if stale:
        raise RuntimeError(f"stale V12 teacher cache: {stale}")
    teacher_cache = pd.read_csv(teacher_cache_path)
    if (teacher_cache["row_index"].duplicated().any()
            or set(teacher_cache["row_index"].astype(int)) != set(range(len(frame)))):
        raise RuntimeError("teacher cache row index mismatch")
    teacher_lookup = teacher_cache.set_index("row_index").to_dict("index")

    wave_cache_path = ROOT / "experiments/v12/cache/train_waves_4s.npy"
    wave_meta_path = wave_cache_path.with_suffix(wave_cache_path.suffix + ".meta.json")
    if not wave_cache_path.is_file() or not wave_meta_path.is_file():
        raise FileNotFoundError("build experiments/v12/cache/train_waves_4s.npy first")
    wave_meta = json.loads(wave_meta_path.read_text(encoding="utf-8"))
    if wave_meta.get("split_sha256") != sha256(split_path):
        raise RuntimeError("stale V12 waveform cache")
    dataset = V12CachedWaveDataset(
        frame, wave_cache_path, wave_meta, task=task,
        augmentation_profile="voice_channel_v9")
    batch_sampler = None
    if config["rank"] > 0:
        batch_sampler = PairedGroupBatchSampler(frame, task, args.batch_size, seed)
        loader = DataLoader(dataset, batch_sampler=batch_sampler,
                            num_workers=args.num_workers,
                            persistent_workers=args.num_workers > 0)
    else:
        weights = source_balanced_weights(frame, task)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(frame), replacement=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.num_workers,
                            persistent_workers=args.num_workers > 0)
    optimizer = make_optimizer(student, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ema = ExponentialMovingAverage(student, decay=0.999) if config["ema"] else None
    path_to_group = dict(zip(frame["path"].astype(str), frame["split_group_id"].astype(str)))
    history = []
    began = time.perf_counter()
    component = f"{task}_fake"
    component_index = 1 if task == "voice" else 2
    for epoch in range(args.epochs):
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        student.train(); total = defaultdict(float); batches = 0
        for wave, labels, paths, row_indices in loader:
            wave = wave.to(device); labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            output_heads = student(wave)
            cached_rows = [teacher_lookup[int(index)] for index in row_indices]
            v7_output = {
                "file_fake": torch.tensor([row["v7_file_logit"] for row in cached_rows], device=device),
                component: torch.tensor([row["v7_component_logit"] for row in cached_rows], device=device),
                f"{task}_present": torch.tensor(
                    [row["v7_presence_logit"] for row in cached_rows], device=device),
            }
            candidate_output = {
                component: torch.tensor(
                    [row["v9_component_logit"] for row in cached_rows], device=device),
            }
            logits = torch.stack([output_heads[head] for head in HEADS], dim=1)
            supervised = masked_multitask_loss(logits, labels, task=task)
            retention = v7_retention_loss(
                output_heads, v7_output, labels, task, config["temperature"])
            candidate_loss = component_teacher_loss(
                output_heads, candidate_output, labels, task, config["temperature"])
            ranking = paired_margin_ranking_loss(
                torch.sigmoid(output_heads[component]), labels[:, component_index],
                paths, path_to_group, margin=0.1)
            loss = (config["supervised"] * supervised
                    + config["retention"] * retention
                    + config["candidate"] * candidate_loss
                    + config["rank"] * ranking)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite V12 loss: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()
            if ema is not None:
                ema.update(student)
            for name, value in (("loss", loss), ("supervised", supervised),
                                ("retention", retention), ("candidate", candidate_loss),
                                ("ranking", ranking)):
                total[name] += float(value.detach().cpu())
            batches += 1
        scheduler.step()
        record = {"epoch": epoch + 1, **{name: value / batches for name, value in total.items()}}
        history.append(record)
        print(json.dumps(record), flush=True)

    state = ema.state_dict() if ema is not None else {
        name: value.detach().cpu() for name, value in student.state_dict().items()}
    dataset_meta = json.loads(dataset_meta_path.read_text(encoding="utf-8"))
    checkpoint = {
        **selected_checkpoint,
        "model": state,
        "epoch": args.epochs - 1,
        "task": task,
        "backbone": "spec_cnn",
        "model_name": "MusicMultitask",
        "base_channels": int(selected_checkpoint["base_channels"]),
        "sample_rate": 16000,
        "seg_sec": 4.0,
        "label_heads": list(HEADS),
        "v12_candidate": args.candidate,
        "distillation_config": config,
        "teacher_sha256": {
            "v7": sha256(selected_path),
            "v9": sha256(candidate_teacher_path),
        },
        "dataset_version": dataset_meta["dataset_version"],
        "manifest_sha256": dataset_meta["manifest_sha256"],
        "split_sha256": dataset_meta["split_sha256"]["train.csv"],
        "training_seconds": time.perf_counter() - began,
        "final_holdout": "NOT RUN",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({
        "candidate": args.candidate, "configuration": config,
        "history": history, "checkpoint": str(output),
        "checkpoint_sha256": sha256(output),
        "training_seconds": checkpoint["training_seconds"],
        "final_holdout": "NOT RUN",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"saved": str(output), "sha256": sha256(output),
                      "seconds": checkpoint["training_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
