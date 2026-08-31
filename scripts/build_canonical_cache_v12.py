#!/usr/bin/env python3
"""Fresh canonical V7 features for all non-final V12 selection domains."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_fusion import balanced_subset
from src.dataset import load_manifest_row_wave
from src.ensemble import assert_final_holdout_forbidden
import script as submission


SCHEMA_VERSION = "v12-canonical-cache-1"
HEADS = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_rows(name: str) -> tuple[pathlib.Path, pd.DataFrame, dict]:
    if name in {"val_a", "val_b", "val_c", "val_d"}:
        split = ROOT / f"data/splits/{name}.csv"
        selector = ROOT / f"experiments/{name}_voice_aggregation.csv"
        source = pd.read_csv(split)
        paths = pd.read_csv(selector)["path"].astype(str)
        indexed = source.set_index(source["path"].astype(str), drop=False)
        rows = indexed.loc[paths].reset_index(drop=True)
        return split, rows, {"selector_sha256": sha256(selector)}
    if name == "cal_old":
        split = ROOT / "data/splits/fusion_calibration.csv"
        return split, balanced_subset(pd.read_csv(split), 0, 20260831), {}
    if name == "cal_v12":
        split = ROOT / "data/splits_v12/cal_v12.csv"
        return split, pd.read_csv(split), {}
    if name == "expanded_unseen":
        split = ROOT / "data/splits_v9_candidate/val_b.csv"
        rows = pd.read_csv(split)
        rows = rows[rows["data_role"].isin(
            ["val_b_unseen_generator", "val_b_unseen_music_generator"]
        )].reset_index(drop=True)
        return split, rows, {}
    raise ValueError(name)


def build_one(name: str, models: tuple, df_session, panns, weights: dict,
              device: torch.device, batch_size: int, output_dir: pathlib.Path) -> dict:
    split, rows, extra_meta = selected_rows(name)
    output = output_dir / f"{name}.csv"
    for path in (split, output):
        assert_final_holdout_forbidden(path)
    voice, music = models
    records = []
    began = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        batch = rows.iloc[start:start + batch_size]
        waves = [load_manifest_row_wave(
            row, sr=16000, is_training=False, use_demucs=False)
            for _, row in batch.iterrows()]
        features = submission.infer_wave_features_batch(
            voice, music, df_session, panns, waves, device,
            use_demucs=False,
            df_config={"enabled": False, "low": 0.0, "high": 1.0},
            specialist_max_segments=int(weights["specialist_max_segments"]),
            panns_max_segments=int(weights["panns_max_segments"]),
            aggregation_config=weights,
        )
        for (_, row), feature in zip(batch.iterrows(), features):
            records.append({
                "path": str(row["path"]),
                "split": name,
                "calibration_fold": str(row.get("calibration_fold", name)),
                "source": str(row.get("source", "unknown")),
                "generator": str(row.get("generator", "unknown")),
                "data_role": str(row.get("data_role", name)),
                **{f"y_{head}": int(row[head]) for head in HEADS},
                **feature,
            })
        print(f"{name}: {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    frame = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    elapsed = time.perf_counter() - began
    meta = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": submission.PIPELINE_VERSION,
        "aggregation_version": "selected-v7-canonical-voice-max",
        "preprocess_version": "16k-high-energy-3-primary-df-panns16k",
        "checkpoint_sha256": {
            "voice": sha256(ROOT / "model/best.pt"),
            "music": sha256(ROOT / "model/music_best.pt"),
            "panns": sha256(ROOT / "model/panns/Cnn14_16k_mAP=0.438.pth"),
            "df_arena": sha256(ROOT / "model/df_arena/df_arena_1b_int8.onnx"),
            "fusion": sha256(ROOT / "model/fusion_weights.json"),
        },
        "split_sha256": sha256(split),
        "rows": len(frame),
        "elapsed_seconds": elapsed,
        "device": str(device),
        "final_holdout": "NOT RUN",
        **extra_meta,
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"split": name, "rows": len(frame), "seconds": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=[
        "cal_old", "cal_v12", "val_a", "val_b", "val_c", "val_d", "expanded_unseen"])
    parser.add_argument("--output_dir", default="experiments/v12/cache/v7_canonical")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=(
        "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"))
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    assert_final_holdout_forbidden(output_dir)
    device = torch.device(args.device)
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    models = (submission.load_voice_model(device), submission.load_music_model(device))
    panns = submission.load_panns(device)
    df_session = submission.load_df_arena("cpu")
    results = [build_one(name, models, df_session, panns, weights, device,
                         args.batch_size, output_dir) for name in args.splits]
    (output_dir.parent.parent / "canonical_cache_runtime.json").write_text(
        json.dumps({"status": "MEASURED", "results": results,
                    "final_holdout": "NOT RUN"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
