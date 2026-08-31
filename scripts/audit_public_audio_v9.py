#!/usr/bin/env python3
"""Decode and quality-audit every public-v9 candidate without loading full files."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import soundfile as sf


MANIFESTS = (
    "mlaad_tiny_manifest.csv",
    "sonics_manifest.csv",
    "real_music_manifest.csv",
    "echoes_fma_paired_manifest.csv",
)


def inspect(path: pathlib.Path, seconds: float = 2.0) -> dict:
    with sf.SoundFile(path) as audio:
        sample_rate = int(audio.samplerate)
        channels = int(audio.channels)
        frames = int(audio.frames)
        duration = frames / max(1, sample_rate)
        take = min(frames, max(1, int(round(seconds * sample_rate))))
        offsets = sorted({0, max(0, (frames - take) // 2), max(0, frames - take)})
        chunks = []
        for offset in offsets:
            audio.seek(offset)
            block = audio.read(take, dtype="float32", always_2d=True)
            if len(block):
                chunks.append(block.mean(axis=1))
    wave = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
    finite = bool(len(wave) and np.isfinite(wave).all())
    rms = float(np.sqrt(np.mean(wave ** 2) + 1e-12)) if finite else 0.0
    peak = float(np.max(np.abs(wave))) if finite else 0.0
    clipping = float(np.mean(np.abs(wave) >= 0.999)) if finite else 1.0
    return {
        "sample_rate": sample_rate, "channels": channels,
        "duration_seconds": duration, "finite": finite,
        "rms_dbfs": float(20.0 * np.log10(max(rms, 1e-12))),
        "peak": peak, "clipping_fraction": clipping,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/public_v9")
    parser.add_argument("--output", default="experiments/v9/public_audio_quality.json")
    args = parser.parse_args()
    root = pathlib.Path(args.root)
    frames = []
    for name in MANIFESTS:
        path = root / name
        if path.exists():
            frame = pd.read_csv(path)
            frame["manifest"] = name
            frames.append(frame)
    data = pd.concat(frames, ignore_index=True, sort=False)
    failures = []
    records = []
    for row in data.to_dict("records"):
        path = pathlib.Path(str(row["path"]))
        try:
            result = inspect(path)
            result.update({
                "path": str(path), "dataset": str(row.get("dataset", "unknown")),
                "generator": str(row.get("generator", "unknown")),
            })
            records.append(result)
            if (not result["finite"] or result["duration_seconds"] < 0.5
                    or result["rms_dbfs"] < -60.0 or result["clipping_fraction"] > 0.05):
                failures.append({"path": str(path), "reason": "quality_threshold", **result})
        except Exception as error:
            failures.append({"path": str(path), "reason": f"decode: {error}"})
    measured = pd.DataFrame(records)
    by_dataset = {}
    for dataset, group in measured.groupby("dataset"):
        by_dataset[str(dataset)] = {
            "files": int(len(group)),
            "duration_hours": float(group.duration_seconds.sum() / 3600.0),
            "duration_median_seconds": float(group.duration_seconds.median()),
            "sample_rates": {str(k): int(v) for k, v in group.sample_rate.value_counts().items()},
            "channels": {str(k): int(v) for k, v in group.channels.value_counts().items()},
            "rms_dbfs_median": float(group.rms_dbfs.median()),
            "clipping_fraction_p99": float(group.clipping_fraction.quantile(0.99)),
        }
    report = {
        "rows": int(len(data)), "decoded": int(len(measured)),
        "failures": failures, "status": "PASS" if not failures else "FAIL",
        "thresholds": {"min_duration_seconds": 0.5, "min_rms_dbfs": -60.0,
                       "max_clipping_fraction": 0.05},
        "by_dataset": by_dataset,
        "final_holdout": "NOT READ / NOT RUN",
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": report["rows"], "decoded": report["decoded"],
                      "failures": len(failures), "status": report["status"],
                      "by_dataset": by_dataset}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
