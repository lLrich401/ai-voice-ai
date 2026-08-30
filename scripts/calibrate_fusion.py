#!/usr/bin/env python3
"""Calibrate the exact submitted fusion on a reproducible VAL-A/B/C/D subset."""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import apply_codec_sim, apply_telephone_sim, render_mixed_wave
from src.metrics import compute_dacon_metrics
from src.train import checkpoint_selection_score
import script as submission


HEADS = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_subset(df, limit, seed):
    keys = ["file_fake", "voice_fake", "music_fake", "voice_present", "music_present"]
    groups = list(df.groupby(keys, dropna=False))
    per_group = max(1, limit // max(1, len(groups)))
    pieces = [g.sample(min(len(g), per_group), random_state=seed + i) for i, (_, g) in enumerate(groups)]
    selected_indices = set().union(*(set(piece.index) for piece in pieces))
    sampled = pd.concat(pieces, ignore_index=False)
    if len(sampled) < min(limit, len(df)):
        remaining = df.drop(index=selected_indices, errors="ignore")
        sampled = pd.concat([
            sampled,
            remaining.sample(min(limit - len(sampled), len(remaining)), random_state=seed + 999),
        ], ignore_index=True)
    return sampled.sample(frac=1, random_state=seed).head(limit).reset_index(drop=True)


def load_row_wave(row):
    path = str(row["path"])
    if path.startswith("MIX::"):
        voice_path, music_path = path.split("MIX::", 1)[1].split("|", 1)
        voice, _ = submission.load_audio(voice_path)
        music, _ = submission.load_audio(music_path)
        wave = render_mixed_wave(
            voice, music, str(row.get("mix_mode", "simultaneous")),
            float(row.get("mix_snr_db", 0.0)), float(row.get("mix_crossfade_sec", 0.25)), 16000)
    else:
        wave, _ = submission.load_audio(path)
    augment = str(row.get("augment", "none")).lower()
    if augment in ("codec_mp3", "codec"):
        wave = apply_codec_sim(wave)
    elif augment in ("telephone", "tel"):
        wave = apply_telephone_sim(wave)
    return wave


def collect(split_name, df, models, df_session, panns, device, batch_size):
    voice, music = models
    records = []
    for start in range(0, len(df), batch_size):
        rows = df.iloc[start:start + batch_size]
        waves = [load_row_wave(row) for _, row in rows.iterrows()]
        features = submission.infer_wave_features_batch(
            voice, music, df_session, panns, waves, device, use_demucs=False)
        for (_, row), feature in zip(rows.iterrows(), features):
            records.append({"split": split_name,
                            **{f"y_{h}": float(row[h]) for h in HEADS}, **feature})
    return records


def score_cached(cache, weights):
    by_split = {}
    for split_name, frame in cache.groupby("split"):
        predicted = [submission.fuse_prediction_features(
            r.df, r.vf, r.mf, r.vfile, r.mfile, r.vp_model, r.mp_model,
            r.vp_panns, r.mp_panns, weights) for r in frame.itertuples()]
        predicted = np.asarray(predicted)
        y_true = {h: frame[f"y_{h}"].to_numpy() for h in HEADS}
        y_pred = {h: predicted[:, i] for i, h in enumerate(HEADS)}
        by_split[split_name] = compute_dacon_metrics(y_true, y_pred)
    return checkpoint_selection_score(by_split), by_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_split", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache", default="experiments/fusion_calibration_predictions.csv")
    parser.add_argument("--reuse_cache", action="store_true", help="reuse only when checkpoints/splits are unchanged")
    args = parser.parse_args()
    device = torch.device(args.device)
    cache_path = pathlib.Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reuse_cache and cache_path.exists():
        cache = pd.read_csv(cache_path)
        print(f"Loaded cached validation predictions: {cache_path} ({len(cache)})")
    else:
        voice = submission.load_voice_model(device)
        music = submission.load_music_model(device)
        panns = submission.load_panns(device)
        df_session = submission.load_df_arena(device)
        records = []
        for i, split in enumerate(("val_a", "val_b", "val_c", "val_d")):
            frame = pd.read_csv(ROOT / "data" / "splits" / f"{split}.csv")
            frame = balanced_subset(frame, args.per_split, 20260830 + i)
            print(f"Collecting exact inference features for {split}: {len(frame)}")
            records.extend(collect(split, frame, (voice, music), df_session, panns,
                                   device, args.batch_size))
        cache = pd.DataFrame(records)
        cache.to_csv(cache_path, index=False)

    detector_weights = []
    for a in range(5):
        for b in range(5 - a):
            c = 4 - a - b
            detector_weights.append((a / 4, b / 4, c / 4))
    best = None
    for wv, wm, wo in detector_weights:
        for wdf in (0.0, 0.25, 0.5, 0.75, 1.0):
            # DF-Arena is generic file/spoof evidence, never the primary
            # component score. Specialists retain at least 75% responsibility.
            for component in (0.0, 0.05, 0.1, 0.25):
                for panns_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                    weights = {
                        "w_voice_file": wv, "w_music_file": wm, "w_prob_or": wo,
                        "w_df_arena": wdf, "w_df_component": component,
                        "w_panns_presence": panns_weight,
                    }
                    selection, metrics = score_cached(cache, weights)
                    if best is None or selection > best[0]:
                        best = (selection, weights, metrics)
    selection, weights, metrics = best
    weights.update({
        "metric_version": "dacon236749-official-noninterpolated-v1",
        "pipeline_version": submission.PIPELINE_VERSION,
        "voice_checkpoint_sha256": sha256(ROOT / "model" / "best.pt"),
        "music_checkpoint_sha256": sha256(ROOT / "model" / "music_best.pt"),
        "df_model_version": "pranjal-pravesh/df_arena_1b:int8-onnx-64600",
        "df_arena_class_0_is_fake": True,
        "df_arena_calibrated": True,
        "calibration_selection_score": selection,
        "calibration_samples": int(len(cache)),
        "calibration_metrics_by_split": metrics,
    })
    out = ROOT / "model" / "fusion_weights.json"
    out.write_text(json.dumps(weights, indent=2), encoding="utf-8")
    print(json.dumps(weights, indent=2))


if __name__ == "__main__":
    main()
