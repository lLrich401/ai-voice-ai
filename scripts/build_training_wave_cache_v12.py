#!/usr/bin/env python3
"""Decode the frozen V12 training split once into a 4-second waveform memmap."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_manifest_row_wave
from src.ensemble import assert_final_holdout_forbidden


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def center_crop(wave: np.ndarray, length: int = 64_000) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float32)
    if len(wave) < length:
        return np.pad(wave, (0, length - len(wave))).astype(np.float32)
    start = (len(wave) - length) // 2
    return np.ascontiguousarray(wave[start:start + length], dtype=np.float32)


def main() -> None:
    split = ROOT / "data/splits_v12/train.csv"
    output = ROOT / "experiments/v12/cache/train_waves_4s.npy"
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    assert_final_holdout_forbidden(split, output, meta_path)
    frame = pd.read_csv(split)
    output.parent.mkdir(parents=True, exist_ok=True)
    waves = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=(len(frame), 64_000))
    began = time.perf_counter()
    for index, (_, row) in enumerate(frame.iterrows()):
        waves[index] = center_crop(load_manifest_row_wave(
            row, sr=16_000, is_training=False, use_demucs=False))
        if (index + 1) % 200 == 0 or index + 1 == len(frame):
            print(f"waves: {index + 1}/{len(frame)}", flush=True)
    waves.flush()
    del waves
    meta = {
        "schema_version": "v12-train-wave-cache-1-center-4s",
        "rows": len(frame), "samples": 64_000, "sample_rate": 16_000,
        "split_sha256": sha256(split), "wave_sha256": sha256(output),
        "elapsed_seconds": time.perf_counter() - began,
        "final_holdout": "NOT RUN",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
