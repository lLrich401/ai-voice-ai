#!/usr/bin/env python3
"""Cache V12 student heads on canonical non-final rows with shared decoding."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_canonical_cache_v12 import SCHEMA_VERSION as BASE_SCHEMA, selected_rows, sha256
from scripts.train_distilled_v12 import load_checkpoint_model
from src.dataset import load_manifest_row_wave
from src.ensemble import assert_final_holdout_forbidden
import script as submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["music", "voice"], required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=[
        "val_a", "val_b", "val_c", "val_d", "expanded_unseen", "cal_old", "cal_v12"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default=(
        "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"))
    args = parser.parse_args()
    task = args.task
    device = torch.device(args.device)
    candidate_paths = {
        name.upper(): ROOT / f"model/candidates/v12/{name.lower()}_student.pt"
        for name in args.candidates}
    for path in candidate_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    models = {name: load_checkpoint_model(path, device)[0].eval()
              for name, path in candidate_paths.items()}
    checkpoint_sha = {name: sha256(path) for name, path in candidate_paths.items()}
    output_root = ROOT / f"experiments/v12/cache/{task}_students"
    assert_final_holdout_forbidden(output_root)
    component, presence = f"{task}_fake", f"{task}_present"
    aggregation = "max" if task == "voice" else "topk_mean"
    runtime = {name: 0.0 for name in models}
    for split_name in args.splits:
        _, rows, _ = selected_rows(split_name)
        canonical_path = ROOT / f"experiments/v12/cache/v7_canonical/{split_name}.csv"
        assert_final_holdout_forbidden(canonical_path)
        canonical = pd.read_csv(canonical_path)
        if canonical["path"].astype(str).tolist() != rows["path"].astype(str).tolist():
            raise RuntimeError(f"canonical/student row mismatch for {split_name}")
        records = {name: [] for name in models}
        for start in range(0, len(rows), args.batch_size):
            batch = rows.iloc[start:start + args.batch_size]
            waves = [load_manifest_row_wave(
                row, sr=16000, is_training=False, use_demucs=False)
                for _, row in batch.iterrows()]
            groups = [submission.select_aux_segments(
                wave, sr=16000, seg_sec=4.0, policy="high_energy") for wave in waves]
            for name, model in models.items():
                began = time.perf_counter()
                output, bounds = submission._run_torch_segments(
                    model, groups, device, use_amp=True)
                runtime[name] += time.perf_counter() - began
                for path, (left, right) in zip(batch["path"].astype(str), bounds):
                    records[name].append({
                        "path": path,
                        "component_fake": submission.aggregate_predictions(
                            output[component][left:right], aggregation, 2),
                        "file_fake": submission.aggregate_predictions(
                            output["file_fake"][left:right], "topk_mean", 2),
                        "component_present": float(np.mean(output[presence][left:right])),
                    })
            print(f"{task} {split_name}: {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
        for name, values in records.items():
            output_dir = output_root / name.lower()
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{split_name}.csv"
            pd.DataFrame(values).to_csv(output_path, index=False)
            output_path.with_suffix(output_path.suffix + ".meta.json").write_text(json.dumps({
                "schema_version": "v12-student-head-cache-1",
                "base_schema_version": BASE_SCHEMA,
                "task": task, "candidate": name,
                "checkpoint_sha256": checkpoint_sha[name],
                "canonical_cache_sha256": sha256(canonical_path),
                "split": split_name, "rows": len(values),
                "aggregation": aggregation,
                "segment_policy": "high_energy-maximum-3",
                "final_holdout": "NOT RUN",
            }, indent=2) + "\n", encoding="utf-8")
    report = {"status": "MEASURED", "task": task, "device": str(device),
              "model_seconds": runtime, "final_holdout": "NOT RUN"}
    (output_root / "runtime.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
