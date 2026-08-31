#!/usr/bin/env python3
"""Build SHA-guarded head caches without reading the protected final holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import working src before script.py prepends the bundled runtime source.
from src.dataset import load_manifest_row_wave
from src.ensemble import assert_final_holdout_forbidden
from src.models.aasist import AASISTMultitask
from src.models.beats_backbone import MusicMultitask
from scripts.calibrate_fusion import balanced_subset
import script as submission


SCHEMA_VERSION = "v11-head-cache-2-recomputed-v7"
AGGREGATION_VERSION = "voice-max_music-top2_file-top2_presence-mean_high-energy-3"
HEADS = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    columns = [column for column in (
        "path", *HEADS, "augment", "data_role", "split_group_id") if column in frame]
    payload = frame[columns].to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_model(path: pathlib.Path, task: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("task") != task:
        raise RuntimeError(f"{path}: task mismatch")
    backbone = checkpoint.get("backbone")
    channels = int(checkpoint.get("base_channels", 32))
    if backbone == "aasist":
        model = AASISTMultitask(base_channels=channels)
    elif backbone == "spec_cnn":
        model = MusicMultitask(base_channels=channels)
    else:
        raise RuntimeError(f"{path}: unsupported backbone={backbone}")
    if int(checkpoint.get("sample_rate", 0)) != 16000:
        raise RuntimeError(f"{path}: sample rate mismatch")
    if tuple(checkpoint.get("label_heads", ())) != HEADS:
        raise RuntimeError(f"{path}: label heads mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def infer_rows(rows: pd.DataFrame, models: dict, device: torch.device, batch_size: int):
    records = []
    elapsed = {name: 0.0 for name in models}
    for start in range(0, len(rows), batch_size):
        batch_rows = rows.iloc[start:start + batch_size]
        waves = [load_manifest_row_wave(row, sr=16000, is_training=False, use_demucs=False)
                 for _, row in batch_rows.iterrows()]
        groups = [submission.select_aux_segments(wave, policy="high_energy") for wave in waves]
        batch_outputs = {}
        for name, model in models.items():
            began = time.perf_counter()
            output, bounds = submission._run_torch_segments(model, groups, device, use_amp=True)
            elapsed[name] += time.perf_counter() - began
            prefix = "vf" if "voice" in name else "mf"
            fake_head = "voice_fake" if prefix == "vf" else "music_fake"
            presence_head = "voice_present" if prefix == "vf" else "music_present"
            fake_method = "max" if prefix == "vf" else "topk_mean"
            batch_outputs[name] = [{
                "fake": submission.aggregate_predictions(output[fake_head][a:b], fake_method, 2),
                "file": submission.aggregate_predictions(output["file_fake"][a:b], "topk_mean", 2),
                "presence": float(np.mean(output[presence_head][a:b])),
            } for a, b in bounds]
        for index, (_, row) in enumerate(batch_rows.iterrows()):
            record = {
                "path": str(row["path"]),
                **{f"y_{head}": int(row[head]) for head in HEADS},
                "source": str(row.get("source", "unknown")),
                "generator": str(row.get("generator", "unknown")),
                "data_role": str(row.get("data_role", "unknown")),
            }
            for name, outputs in batch_outputs.items():
                prefix = "v9_v" if name == "v9_voice" else "v9_m"
                if name == "v7_voice": prefix = "v7_v"
                if name == "v7_music": prefix = "v7_m"
                record[f"{prefix}f"] = outputs[index]["fake"]
                record[f"{prefix}file"] = outputs[index]["file"]
                record[f"{prefix}p_model"] = outputs[index]["presence"]
            records.append(record)
    return pd.DataFrame(records), elapsed


def validate_alignment(base: pd.DataFrame, rows: pd.DataFrame, name: str) -> None:
    if len(base) != len(rows):
        raise RuntimeError(f"{name}: row-count mismatch")
    for head in HEADS:
        if not np.array_equal(base[f"y_{head}"].to_numpy(dtype=int),
                              rows[head].to_numpy(dtype=int)):
            raise RuntimeError(f"{name}: label mismatch for {head}")


def save_cache(output: pathlib.Path, frame: pd.DataFrame, metadata: dict) -> None:
    assert_final_holdout_forbidden(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"saved {output} rows={len(frame)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experiments/v11/cache")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else
        "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"))
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    assert_final_holdout_forbidden(output_dir)
    device = torch.device(args.device)
    paths = {
        "v7_voice": ROOT / "model/best.pt",
        "v7_music": ROOT / "model/music_best.pt",
        "v9_voice": ROOT / "model/candidates/voice_aasist_v9.pt",
        "v9_music": ROOT / "model/candidates/music_spec_cnn_v9.pt",
    }
    checkpoint_sha = {name: sha256(path) for name, path in paths.items()}
    all_models = {name: load_model(path, name.split("_", 1)[1], device)
                  for name, path in paths.items()}
    runtime = {name: 0.0 for name in paths}

    for split in ("val_a", "val_b", "val_c", "val_d"):
        split_path = ROOT / f"data/splits/{split}.csv"
        selector_path = ROOT / f"experiments/{split}_voice_aggregation.csv"
        base_path = ROOT / f"experiments/{split}_features_16k.csv"
        assert_final_holdout_forbidden(split_path, selector_path, base_path)
        source = pd.read_csv(split_path)
        selector = pd.read_csv(selector_path)
        indexed = source.set_index(source["path"].astype(str), drop=False)
        rows = indexed.loc[selector["path"].astype(str)].reset_index(drop=True)
        base = pd.read_csv(base_path)
        validate_alignment(base, rows, split)
        candidate, elapsed = infer_rows(rows, all_models, device, args.batch_size)
        for name, seconds in elapsed.items():
            runtime[name] += seconds
        combined = pd.concat([candidate[["path"]].reset_index(drop=True),
                              base.reset_index(drop=True)], axis=1)
        # The historical base feature files predate the selected voice-max
        # policy. Recompute both v7 specialists on exactly the same segments
        # so alpha=0 represents the current submission rather than stale input.
        combined["vf"], combined["vfile"], combined["vp_model"] = (
            candidate["v7_vf"], candidate["v7_vfile"], candidate["v7_vp_model"])
        combined["mf"], combined["mfile"], combined["mp_model"] = (
            candidate["v7_mf"], candidate["v7_mfile"], candidate["v7_mp_model"])
        for column in ("v9_vf", "v9_vfile", "v9_vp_model",
                       "v9_mf", "v9_mfile", "v9_mp_model"):
            combined[column] = candidate[column]
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": submission.PIPELINE_VERSION,
            "aggregation_version": AGGREGATION_VERSION,
            "checkpoint_sha256": checkpoint_sha,
            "split_sha256": sha256(split_path),
            "selector_sha256": sha256(selector_path),
            "selected_rows_sha256": frame_sha256(rows),
            "rows": len(combined), "final_holdout": "NOT RUN",
        }
        save_cache(output_dir / f"original_{split}.csv", combined, metadata)

    calibration_split = ROOT / "data/splits/fusion_calibration.csv"
    calibration_base = ROOT / "experiments/v7/fusion_calibration_predictions.csv"
    assert_final_holdout_forbidden(calibration_split, calibration_base)
    calibration_rows = balanced_subset(pd.read_csv(calibration_split), 0, 20260831)
    base = pd.read_csv(calibration_base)
    validate_alignment(base, calibration_rows, "fusion_calibration")
    candidate, elapsed = infer_rows(calibration_rows, all_models, device, args.batch_size)
    for name, seconds in elapsed.items():
        runtime[name] += seconds
    combined = pd.concat([candidate[["path"]].reset_index(drop=True),
                          base.reset_index(drop=True)], axis=1)
    combined["vf"], combined["vfile"], combined["vp_model"] = (
        candidate["v7_vf"], candidate["v7_vfile"], candidate["v7_vp_model"])
    combined["mf"], combined["mfile"], combined["mp_model"] = (
        candidate["v7_mf"], candidate["v7_mfile"], candidate["v7_mp_model"])
    for column in ("v9_vf", "v9_vfile", "v9_vp_model",
                   "v9_mf", "v9_mfile", "v9_mp_model"):
        combined[column] = candidate[column]
    save_cache(output_dir / "fusion_calibration.csv", combined, {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": submission.PIPELINE_VERSION,
        "aggregation_version": AGGREGATION_VERSION,
        "checkpoint_sha256": checkpoint_sha,
        "split_sha256": sha256(calibration_split),
        "selected_rows_sha256": frame_sha256(calibration_rows),
        "rows": len(combined), "final_holdout": "NOT RUN",
    })

    expanded_path = ROOT / "data/splits_v9_candidate/val_b.csv"
    assert_final_holdout_forbidden(expanded_path)
    expanded = pd.read_csv(expanded_path)
    unseen = expanded[expanded["data_role"].isin((
        "val_b_unseen_generator", "val_b_unseen_music_generator"))].reset_index(drop=True)
    unseen_cache, elapsed = infer_rows(unseen, all_models, device, args.batch_size)
    for name, seconds in elapsed.items():
        runtime[name] += seconds
    save_cache(output_dir / "expanded_unseen.csv", unseen_cache, {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": submission.PIPELINE_VERSION,
        "aggregation_version": AGGREGATION_VERSION,
        "checkpoint_sha256": checkpoint_sha,
        "split_sha256": sha256(expanded_path),
        "selected_rows_sha256": frame_sha256(unseen),
        "rows": len(unseen_cache), "voice_unseen_rows": int(
            unseen["data_role"].eq("val_b_unseen_generator").sum()),
        "music_unseen_rows": int(
            unseen["data_role"].eq("val_b_unseen_music_generator").sum()),
        "final_holdout": "NOT RUN",
    })
    (output_dir.parent / "cache_runtime.json").write_text(json.dumps({
        "status": "MEASURED_NON_FINAL_FEATURE_EXTRACTION",
        "device": str(device), "model_seconds": runtime,
        "final_holdout": "NOT RUN",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
