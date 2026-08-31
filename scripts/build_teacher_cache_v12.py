#!/usr/bin/env python3
"""Cache frozen V7/V9 teacher logits once for all V12 student candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import AudioDataset
from src.distillation import freeze_teacher, teacher_is_frozen
from src.ensemble import assert_final_holdout_forbidden
from scripts.train_distilled_v12 import load_checkpoint_model, sha256


class IndexedAudioDataset(AudioDataset):
    def __getitem__(self, index):
        wave, labels, path = super().__getitem__(index)
        return wave, labels, path, index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["music", "voice"], required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default=(
        "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"))
    args = parser.parse_args()
    task = args.task
    split = ROOT / "data/splits_v12/train.csv"
    output = ROOT / f"experiments/v12/cache/teacher_{task}.csv"
    assert_final_holdout_forbidden(split, output)
    device = torch.device(args.device)
    selected = ROOT / ("model/best.pt" if task == "voice" else "model/music_best.pt")
    candidate = ROOT / ("model/candidates/voice_aasist_v9.pt" if task == "voice"
                        else "model/candidates/music_spec_cnn_v9.pt")
    v7, _ = load_checkpoint_model(selected, device)
    v9, _ = load_checkpoint_model(candidate, device)
    freeze_teacher(v7); freeze_teacher(v9)
    if not teacher_is_frozen(v7) or not teacher_is_frozen(v9):
        raise RuntimeError("teacher freeze failed")
    frame = pd.read_csv(split)
    dataset = IndexedAudioDataset(frame, sr=16000, seg_sec=4.0, is_training=False,
                                  use_demucs=False, task=task, device=str(device))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0)
    component, presence = f"{task}_fake", f"{task}_present"
    records = []
    began = time.perf_counter()
    with torch.no_grad():
        for batch_index, (waves, _, paths, row_indices) in enumerate(loader):
            waves = waves.to(device)
            v7_output = v7(waves)
            v9_output = v9(waves)
            for index, (path, row_index) in enumerate(zip(paths, row_indices)):
                records.append({
                    "row_index": int(row_index),
                    "path": str(path),
                    "v7_file_logit": float(v7_output["file_fake"][index].cpu()),
                    "v7_component_logit": float(v7_output[component][index].cpu()),
                    "v7_presence_logit": float(v7_output[presence][index].cpu()),
                    # Candidate FILE logits are deliberately not persisted.
                    "v9_component_logit": float(v9_output[component][index].cpu()),
                })
            print(f"{task}: {len(records)}/{len(frame)}", flush=True)
    cached = pd.DataFrame(records)
    if (cached["row_index"].duplicated().any()
            or set(cached["row_index"]) != set(range(len(frame)))
            or cached.sort_values("row_index")["path"].astype(str).tolist()
            != frame["path"].astype(str).tolist()):
        raise RuntimeError("teacher cache row/path mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    cached.to_csv(output, index=False)
    meta = {
        "schema_version": "v12-teacher-logits-1-clean-center-crop",
        "task": task,
        "rows": len(cached),
        "split_sha256": sha256(split),
        "checkpoint_sha256": {"v7": sha256(selected), "v9": sha256(candidate)},
        "candidate_file_head_cached": False,
        "teachers_frozen": True,
        "elapsed_seconds": time.perf_counter() - began,
        "device": str(device),
        "final_holdout": "NOT RUN",
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
