#!/usr/bin/env python3
"""Audit v8 procedural audio quality, provenance, labels, and split isolation."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import stft

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.procedural_audio_v8 import SAMPLE_RATE, audio_stats, quality_errors  # noqa: E402


LABELS = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(wave: np.ndarray, bands: int = 24, time_bins: int = 48) -> np.ndarray:
    rms = float(np.sqrt(np.mean(wave ** 2) + 1e-12))
    _, _, spectrum = stft(wave / max(rms, 1e-6), fs=SAMPLE_RATE, nperseg=512,
                          noverlap=384, boundary=None)
    power = np.log1p(np.abs(spectrum) ** 2)
    edges = np.linspace(0, power.shape[0], bands + 1, dtype=int)
    block = np.stack([power[left:max(left + 1, right)].mean(axis=0)
                      for left, right in zip(edges[:-1], edges[1:])])
    source_axis = np.linspace(0.0, 1.0, block.shape[1])
    target_axis = np.linspace(0.0, 1.0, time_bins)
    vector = np.stack([np.interp(target_axis, source_axis, row) for row in block]).ravel()
    vector = (vector - vector.mean()) / (vector.std() + 1e-6)
    return (vector / max(np.linalg.norm(vector), 1e-6)).astype(np.float32)


def audit(manifest_path: pathlib.Path, near_threshold: float) -> dict[str, object]:
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    failures: list[dict[str, object]] = []
    hashes: dict[str, list[str]] = {}
    vectors: list[np.ndarray] = []
    stats_rows: list[dict[str, float]] = []
    for row in frame.to_dict("records"):
        path = ROOT / str(row["path"])
        if not path.is_file():
            failures.append({"path": str(row["path"]), "error": "missing_file"})
            continue
        wave, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if wave.ndim != 1:
            failures.append({"path": str(row["path"]), "error": "not_mono"})
            continue
        if sample_rate != SAMPLE_RATE:
            failures.append({"path": str(row["path"]), "error": f"sample_rate_{sample_rate}"})
        errors = quality_errors(wave, float(row["duration_sec"]))
        failures.extend({"path": str(row["path"]), "error": error} for error in errors)
        actual_sha = _sha256(path)
        if actual_sha != row["audio_sha256"]:
            failures.append({"path": str(row["path"]), "error": "sha256_mismatch"})
        hashes.setdefault(actual_sha, []).append(str(row["path"]))
        vectors.append(_fingerprint(wave))
        stats_rows.append(audio_stats(wave).__dict__)

    expected_labels = {
        (1, 0): (1, 1, 0, 1, 0),
        (0, 1): (1, 0, 1, 0, 1),
        (1, 1): (1, 1, 1, 1, 1),
    }
    for row in frame.to_dict("records"):
        presence = (int(row["voice_present"]), int(row["music_present"]))
        labels = tuple(int(row[column]) for column in LABELS)
        if expected_labels.get(presence) != labels:
            failures.append({"path": row["path"], "error": "invalid_generated_labels", "labels": labels})
        if row["external_assets_used"] != "NO" or row["allowed_for_competition"] != "YES":
            failures.append({"path": row["path"], "error": "provenance_not_approved"})

    train = frame[frame.recommended_split == "train"]
    valid = frame[frame.recommended_split == "val_unseen_generator"]
    isolation = {}
    for column in ("original_id", "split_group_id", "near_duplicate_group", "speaker_id",
                   "base_voice_id", "base_music_id"):
        left = set(train[column].astype(str)) - {""}
        right = set(valid[column].astype(str)) - {""}
        overlap = sorted(left & right)
        isolation[column] = {"overlap_count": len(overlap), "examples": overlap[:10]}
        if overlap:
            failures.append({"error": f"cross_split_{column}", "examples": overlap[:10]})
    train_families = set(train.generator_family.astype(str))
    valid_families = set(valid.generator_family.astype(str))
    family_overlap = sorted(train_families & valid_families)
    if family_overlap:
        failures.append({"error": "generator_family_overlap", "examples": family_overlap})

    exact_duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    if exact_duplicates:
        failures.append({"error": "exact_audio_duplicates", "groups": exact_duplicates[:10]})

    near_pairs: list[dict[str, object]] = []
    if vectors and len(vectors) == len(frame):
        matrix = np.stack(vectors)
        train_idx = np.flatnonzero(frame.recommended_split.to_numpy() == "train")
        valid_idx = np.flatnonzero(frame.recommended_split.to_numpy() == "val_unseen_generator")
        similarity = matrix[valid_idx] @ matrix[train_idx].T
        for local_index, valid_row in enumerate(valid_idx):
            best_position = int(np.argmax(similarity[local_index]))
            best = float(similarity[local_index, best_position])
            if best >= near_threshold:
                train_row = int(train_idx[best_position])
                near_pairs.append({"validation": str(frame.iloc[valid_row].path),
                                   "train": str(frame.iloc[train_row].path), "similarity": best})
    if near_pairs:
        failures.append({"error": "cross_split_near_duplicates", "count": len(near_pairs)})

    summary_stats = {}
    if stats_rows:
        for name in ("duration_sec", "peak", "rms", "dc_offset", "clipping_fraction"):
            values = np.asarray([row[name] for row in stats_rows])
            summary_stats[name] = {"min": float(values.min()), "mean": float(values.mean()),
                                   "max": float(values.max())}
    audio_bytes = sum((ROOT / str(path)).stat().st_size for path in frame.path if (ROOT / str(path)).is_file())
    return {
        "status": "PASS" if not failures else "FAIL",
        "rows": int(len(frame)), "train_rows": int(len(train)), "validation_rows": int(len(valid)),
        "audio_bytes": int(audio_bytes),
        "external_assets_used": False,
        "provenance_statement": "Waveforms are derived only from numeric seeds and project source code.",
        "class_counts": frame[list(LABELS)].astype(int).sum().to_dict(),
        "generator_counts": frame.generator.value_counts().sort_index().to_dict(),
        "split_isolation": isolation,
        "generator_family_overlap": family_overlap,
        "exact_duplicate_groups": exact_duplicates,
        "near_duplicate_threshold": near_threshold,
        "cross_split_near_duplicates": near_pairs,
        "audio_stats": summary_stats,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/generated_v8/manifest.csv")
    parser.add_argument("--output", default="experiments/v8/generated_dataset_audit.json")
    parser.add_argument("--near-threshold", type=float, default=0.9995)
    args = parser.parse_args()
    report = audit(ROOT / args.manifest, args.near_threshold)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": report["rows"],
                      "train_rows": report["train_rows"],
                      "validation_rows": report["validation_rows"],
                      "failures": len(report["failures"]),
                      "near_duplicates": len(report["cross_split_near_duplicates"])}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
