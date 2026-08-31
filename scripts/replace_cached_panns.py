#!/usr/bin/env python3
"""Replace only PANNs columns in a verified calibration feature cache."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_fusion import cache_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="experiments/fusion_calibration_predictions.csv")
    parser.add_argument("--panns", default="experiments/panns_ab/fusion_calibration_16k.csv")
    parser.add_argument("--output", default="experiments/fusion_calibration_predictions_16k.csv")
    parser.add_argument("--baseline_ref", default="1b5553200d08dcf4f7867e7ecfc8cc93a5d62d5f")
    args = parser.parse_args()

    base = pd.read_csv(ROOT / args.base)
    panns = pd.read_csv(ROOT / args.panns)
    checks = (
        (base["source"].astype(str), panns["source"].astype(str), "source"),
        (base["generator"].astype(str), panns["generator"].astype(str), "generator"),
        (base["y_voice_present"].astype(int), panns["voice_present"].astype(int), "voice_present"),
        (base["y_music_present"].astype(int), panns["music_present"].astype(int), "music_present"),
    )
    for left, right, name in checks:
        if not left.reset_index(drop=True).equals(right.reset_index(drop=True)):
            raise RuntimeError(f"PANNs replacement ordering mismatch: {name}")
    upgraded = base.copy()
    upgraded["vp_panns"] = panns["vp_panns_16k"].to_numpy()
    upgraded["mp_panns"] = panns["mp_panns_16k"].to_numpy()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    upgraded.to_csv(output, index=False)
    metadata = cache_metadata([ROOT / "data/splits/fusion_calibration.csv"])
    metadata["baseline_ref"] = args.baseline_ref
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {output} with {len(upgraded)} rows")


if __name__ == "__main__":
    main()
