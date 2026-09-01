#!/usr/bin/env python3
"""Audit label shortcuts after deterministic V13B partial/mix rendering."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_manifest_row_wave
from tools.v13_guards import assert_final_holdout_v13b_forbidden


FEATURES = ("duration", "peak", "rms", "silence_ratio", "zcr",
            "spectral_centroid", "high_frequency_ratio")


def observation(wave: np.ndarray, length: int = 64_000) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float32)
    if len(wave) < length:
        return np.pad(wave, (0, length - len(wave)))
    start = (len(wave) - length) // 2
    return wave[start:start + length]


def features(row: pd.Series) -> dict:
    wave = observation(load_manifest_row_wave(
        row, sr=16_000, is_training=False, use_demucs=False))
    if not np.isfinite(wave).all():
        raise RuntimeError(f"non-finite rendered wave: {row.path}")
    magnitude = np.abs(np.fft.rfft(wave * np.hanning(len(wave)))) + 1e-12
    power = magnitude ** 2
    frequency = np.fft.rfftfreq(len(wave), 1.0 / 16_000)
    role = str(row.data_role)
    if role == "mixed":
        category = str(row.mix_state)
    elif role.startswith("partial"):
        category = "PARTIAL_FAKE" if int(row.file_fake) else "PARTIAL_CONTROL"
    else:
        category = "PAIRED_CORE"
    return {
        "label": int(row.file_fake), "category": category, "data_role": role,
        "group": str(row.get("base_audio_id", row.content_group)),
        "transition_location": str(row.get("partial_fake_position", "none")),
        "duration": len(wave) / 16_000.0,
        "peak": float(np.max(np.abs(wave))),
        "rms": float(np.sqrt(np.mean(wave ** 2))),
        "silence_ratio": float(np.mean(np.abs(wave) < 1e-3)),
        "zcr": float(np.mean(np.signbit(wave[1:]) != np.signbit(wave[:-1]))),
        "spectral_centroid": float(np.sum(frequency * magnitude) / magnitude.sum()),
        "high_frequency_ratio": float(power[frequency >= 4_000].sum() / power.sum()),
    }


def grouped_auc(frame: pd.DataFrame) -> float | None:
    if len(frame) < 20 or frame.label.nunique() != 2:
        return None
    splits = min(5, int(frame.label.value_counts().min()))
    splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=23674913)
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, C=0.2, class_weight="balanced")),
    ])
    probability = cross_val_predict(
        model, frame[list(FEATURES)], frame.label,
        groups=frame.group, cv=splitter, method="predict_proba")[:, 1]
    return float(roc_auc_score(frame.label, probability))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/splits_v13b/train.csv")
    parser.add_argument("--output", default="experiments/v13b/rendered_training_shortcut_audit.json")
    args = parser.parse_args()
    manifest = ROOT / args.manifest
    output = ROOT / args.output
    assert_final_holdout_v13b_forbidden(manifest, output)
    source = pd.read_csv(manifest)
    rendered = pd.DataFrame([features(row) for _, row in source.iterrows()])
    raw_auc = {
        "all": grouped_auc(rendered),
        "paired_core": grouped_auc(rendered[rendered.data_role.eq("paired_core")]),
        "partial_fake_vs_control": grouped_auc(
            rendered[rendered.data_role.isin(("partial_fake", "partial_real_control"))]),
        "mixed_rr_vs_fake_states": grouped_auc(rendered[rendered.data_role.eq("mixed")]),
    }
    effective_auc = {
        key: (None if value is None else float(max(value, 1.0 - value)))
        for key, value in raw_auc.items()
    }
    distance_from_random = {
        key: (None if value is None else float(abs(value - 0.5)))
        for key, value in raw_auc.items()
    }
    fingerprints = rendered.groupby(["category", "label"])[list(FEATURES)].agg(
        ["count", "mean", "std"])
    fingerprints.columns = ["_".join(column) for column in fingerprints.columns]
    fingerprints = fingerprints.reset_index()
    feature_path = output.with_name(output.stem + "_features.csv")
    fingerprint_path = output.with_name(output.stem + "_fingerprints.csv")
    rendered.to_csv(feature_path, index=False)
    fingerprints.to_csv(fingerprint_path, index=False)
    worst = max(value for value in effective_auc.values() if value is not None)
    report = {
        "status": "PASS" if worst <= 0.75 else "WARN_RENDERER_SHORTCUT",
        "scope": "deterministic rendered TRAIN observations only; supplemental safety evidence",
        "observation": "center 4 seconds for every row",
        "rows": len(rendered), "group_cv": "StratifiedGroupKFold(base_audio_id)",
        "features": list(FEATURES),
        "raw_auc": raw_auc,
        "effective_auc": effective_auc,
        "distance_from_random": distance_from_random,
        "auc": raw_auc,
        "worst_effective_auc": worst,
        "hard_reference": 0.75,
        "symmetric_interpretation": "effective_auc = max(raw_auc, 1 - raw_auc)",
        "transition_location_by_label": {
            str(label): {str(key[1] if isinstance(key, tuple) else key): int(value)
                         for key, value in counts.items()}
            for label, counts in rendered.groupby("label").transition_location.value_counts().groupby(level=0)
        },
        "renderer_policy": "partial positive/control share identical splice/crossfade function; all mixed states share one renderer",
        "features_file": feature_path.relative_to(ROOT).as_posix(),
        "fingerprints_file": fingerprint_path.relative_to(ROOT).as_posix(),
        "final_holdout": "NOT READ / NOT RUN",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
