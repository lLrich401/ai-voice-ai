#!/usr/bin/env python3
"""Merge public-v9 candidates into model-selection splits without holdout use."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.dataset import assert_no_base_source_overlap


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_splits", default="data/splits")
    parser.add_argument("--public_root", default="data/public_v9")
    parser.add_argument("--base_manifest", default="data/manifest.csv")
    parser.add_argument("--output", default="data/splits_v9_candidate")
    args = parser.parse_args()

    base = pathlib.Path(args.base_splits)
    public = pathlib.Path(args.public_root)
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    mlaad = pd.read_csv(public / "mlaad_tiny_manifest.csv")
    sonics = pd.read_csv(public / "sonics_manifest.csv")
    real_music_path = public / "real_music_manifest.csv"
    real_music = pd.read_csv(real_music_path) if real_music_path.exists() else pd.DataFrame()
    echoes_path = public / "echoes_fma_paired_manifest.csv"
    echoes = pd.read_csv(echoes_path) if echoes_path.exists() else pd.DataFrame()
    voice_train = mlaad[mlaad["data_role"].eq("train")].copy()
    voice_unseen = mlaad[~mlaad["data_role"].eq("train")].copy()
    echoes_train = (
        echoes[echoes["data_role"].eq("train")].copy() if len(echoes) else pd.DataFrame()
    )
    echoes_unseen = (
        echoes[~echoes["data_role"].eq("train")].copy() if len(echoes) else pd.DataFrame()
    )

    train = pd.read_csv(base / "train.csv")
    val_a = pd.read_csv(base / "val_a.csv")
    val_b = pd.read_csv(base / "val_b.csv")
    val_c = pd.read_csv(base / "val_c.csv")
    val_d = pd.read_csv(base / "val_d.csv")
    calibration = pd.read_csv(base / "fusion_calibration.csv")

    candidate_train = pd.concat(
        [frame for frame in (train, voice_train, sonics, real_music, echoes_train) if len(frame)],
        ignore_index=True, sort=False,
    )
    candidate_val_b = pd.concat(
        [frame for frame in (val_b, voice_unseen, echoes_unseen) if len(frame)],
        ignore_index=True, sort=False,
    )
    candidate_train["data_role"] = "train"
    candidate_val_b.loc[candidate_val_b["data_role"].isna(), "data_role"] = "val_b"
    for name, frame in (("val_a", val_a), ("val_c", val_c), ("val_d", val_d)):
        frame["data_role"] = name
    calibration["data_role"] = "fusion_calibration"

    # Public-v9 original groups cannot cross any model-selection boundary.
    for name, frame in (("val_a", val_a), ("val_b", candidate_val_b),
                        ("val_c", val_c), ("val_d", val_d),
                        ("fusion_calibration", calibration)):
        assert_no_base_source_overlap(candidate_train, frame, ("train_v9", name))

    frames = {
        "train": candidate_train, "val_a": val_a, "val_b": candidate_val_b,
        "val_c": val_c, "val_d": val_d, "fusion_calibration": calibration,
    }
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)

    full_manifest = pd.concat(
        [
            frame for frame in
            (pd.read_csv(args.base_manifest), mlaad, sonics, real_music, echoes)
            if len(frame)
        ],
        ignore_index=True, sort=False
    ).drop_duplicates("path")
    full_manifest.to_csv(output / "manifest.csv", index=False)

    report = {
        "policy": "candidate-only; v6 final_holdout was not read or copied",
        "base_split_sha256": {
            name: file_sha256(base / f"{name}.csv")
            for name in ("train", "val_a", "val_b", "val_c", "val_d", "fusion_calibration")
        },
        "rows": {name: int(len(frame)) for name, frame in frames.items()},
        "new_voice_train_rows": int(len(voice_train)),
        "new_voice_unseen_val_b_rows": int(len(voice_unseen)),
        "new_sonics_train_rows": int(len(sonics)),
        "new_real_music_train_rows": int(len(real_music)),
        "new_echoes_paired_train_rows": int(len(echoes_train)),
        "new_echoes_paired_unseen_val_b_rows": int(len(echoes_unseen)),
        "new_echoes_paired_content_groups": int(
            echoes["split_group_id"].nunique() if len(echoes) else 0
        ),
        "new_real_music_sources": sorted(set(
            (real_music.dataset.astype(str).tolist() if len(real_music) else [])
            + (["echoes_fma_originals"] if len(echoes) else [])
        )),
        "voice_train_generator_count": int(
            voice_train.loc[voice_train.voice_fake.eq(1), "generator"].nunique()
        ),
        "voice_unseen_generator_count": int(
            voice_unseen.loc[voice_unseen.voice_fake.eq(1), "generator"].nunique()
        ),
        "music_new_generator_count": int(pd.concat([
            sonics.loc[sonics.music_fake.eq(1), "generator"],
            echoes_train.loc[echoes_train.music_fake.eq(1), "generator"]
            if len(echoes_train) else pd.Series(dtype=str),
        ]).nunique()),
        "music_total_fake_generators": sorted(
            candidate_train.loc[candidate_train.music_fake.eq(1), "generator"].astype(str).unique()
        ),
        "music_total_generator_family_count_excluding_mixes": int(
            candidate_train.loc[
                candidate_train.music_fake.eq(1)
                & ~candidate_train.generator.astype(str).str.startswith("mix::"),
                "generator",
            ].nunique()
        ),
        "final_holdout": "NOT READ / NOT COPIED / NOT RUN",
    }
    (output / "prepare_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
