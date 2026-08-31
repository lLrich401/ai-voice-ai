#!/usr/bin/env python3
"""Prepare license-recorded public v9 diversity candidates.

This script intentionally does not touch ``data/manifest.csv`` or any v6/v7
split.  It creates candidate-only manifests under ``data/public_v9`` and puts
downloaded audio under the git-ignored ``data/raw`` tree.

Voice data comes from MLAAD-tiny.  Every selected fake has its exact original
recording in the same split group, and content groups are disjoint between the
candidate train and unseen-generator validation sets.

Music data comes from the official SONICS metadata/archive.  The first archive
contains two independent generator families (Suno chirp-v3.5 and Udio 120s).
SONICS is not used as real music because its repository distributes only the
synthetic files; using its YouTube IDs would add an unverifiable license path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import zipfile

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


MLAAD_REPO = "mueller91/MLAAD-tiny"
MLAAD_REVISION = "9143e5ea709575ebab6bec52840a1043aada7bb1"
MLAAD_SOURCE = "https://huggingface.co/datasets/mueller91/MLAAD-tiny"
SONICS_SOURCE = "https://github.com/awsaf49/sonics"
SONICS_DATASET = "https://huggingface.co/datasets/awsaf49/sonics"

MLAAD_TRAIN_GENERATORS = (
    "RVC",
    "OpenVoiceV2",
    "Chatterbox",
    "FishTTS",
    "Higgs-Audio-V2",
    "FireRedTTS-2.0",
    "f5-tts",
    "Qwen2.5-Omni",
    "MiniCPM-o-2.6",
    "Microsoft VibeVoice 1.5B",
    "suno_bark",
    "zonosTTS-v0.1",
)
MLAAD_UNSEEN_GENERATORS = (
    "Cartesia.ai (Sonic-3)",
    "DeepGram",
    "Edge-TTS",
    "Resemble.ai (April 12th, 2025)",
)
SONICS_GENERATORS = (
    "chirp-v3.5", "udio-120s", "chirp-v3", "udio-30s", "chirp-v2-xxl-alpha",
)

MANIFEST_COLUMNS = (
    "path", "file_fake", "voice_fake", "music_fake", "voice_present",
    "music_present", "speaker_id", "generator", "source", "dataset",
    "hf_id", "original_id", "split_group_id", "source_url", "version",
    "license", "allowed_for_competition", "redistribution_allowed",
    "commercial_restriction", "dataset_name", "content_hash",
    "near_duplicate_group", "data_role",
    "upstream_split", "upstream_label",
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(rows: list[dict], path: pathlib.Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for optional in ("upstream_split", "upstream_label"):
        if optional not in frame.columns:
            frame[optional] = "not_applicable"
    missing = set(MANIFEST_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"manifest columns missing: {sorted(missing)}")
    frame = frame[list(MANIFEST_COLUMNS)].sort_values(
        ["data_role", "generator", "original_id", "path"]
    ).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _mlaad_index() -> tuple[dict[str, list[str]], dict[str, str]]:
    files = HfApi().list_repo_files(
        MLAAD_REPO, repo_type="dataset", revision=MLAAD_REVISION
    )
    originals = {
        pathlib.PurePosixPath(name).name: name
        for name in files
        if name.startswith("original/en/") and name.endswith(".wav")
    }
    fake_by_generator: dict[str, list[str]] = {}
    for name in files:
        parts = pathlib.PurePosixPath(name).parts
        if len(parts) == 4 and parts[:2] == ("fake", "en") and name.endswith(".wav"):
            fake_by_generator.setdefault(parts[2], []).append(name)
    return fake_by_generator, originals


def select_mlaad_pairs(per_generator: int = 20) -> tuple[list[dict], list[dict]]:
    """Select content-disjoint exact pairs before any download occurs."""
    fake_by_generator, originals = _mlaad_index()
    selected_content: set[str] = set()

    def choose(generators: tuple[str, ...], role: str) -> list[dict]:
        pairs: list[dict] = []
        for generator in generators:
            if generator not in fake_by_generator:
                raise RuntimeError(f"MLAAD generator missing at pinned revision: {generator}")
            available = sorted(
                item for item in fake_by_generator[generator]
                if pathlib.PurePosixPath(item).name in originals
                and pathlib.PurePosixPath(item).name not in selected_content
            )
            if len(available) < per_generator:
                raise RuntimeError(
                    f"Only {len(available)} unused exact pairs for {generator}; "
                    f"requested {per_generator}"
                )
            for fake_path in available[:per_generator]:
                basename = pathlib.PurePosixPath(fake_path).name
                selected_content.add(basename)
                pairs.append({
                    "generator": generator,
                    "fake_repo_path": fake_path,
                    "real_repo_path": originals[basename],
                    "content_id": pathlib.PurePosixPath(basename).stem,
                    "data_role": role,
                })
        return pairs

    # Reserve unseen-generator contents first, then make the train set disjoint.
    validation = choose(MLAAD_UNSEEN_GENERATORS, "val_b_unseen_generator")
    training = choose(MLAAD_TRAIN_GENERATORS, "train")
    return training, validation


def _download_mlaad(repo_path: str, root: pathlib.Path) -> pathlib.Path:
    cached = pathlib.Path(hf_hub_download(
        repo_id=MLAAD_REPO,
        repo_type="dataset",
        filename=repo_path,
        revision=MLAAD_REVISION,
    ))
    destination = root.joinpath(*pathlib.PurePosixPath(repo_path).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.stat().st_size != cached.stat().st_size:
        shutil.copy2(cached, destination)
    return destination


def prepare_mlaad(root: pathlib.Path, output: pathlib.Path, per_generator: int) -> pd.DataFrame:
    training, validation = select_mlaad_pairs(per_generator)
    rows: list[dict] = []
    downloaded_real: dict[tuple[str, str], pathlib.Path] = {}
    for pair in training + validation:
        role = pair["data_role"]
        content_id = pair["content_id"]
        group = f"mlaad_tiny::en::{content_id}"
        fake_path = _download_mlaad(pair["fake_repo_path"], root)
        real_key = (role, content_id)
        if real_key not in downloaded_real:
            downloaded_real[real_key] = _download_mlaad(pair["real_repo_path"], root)
            real_path = downloaded_real[real_key]
            rows.append({
                "path": str(real_path), "file_fake": 0, "voice_fake": 0,
                "music_fake": 0, "voice_present": 1, "music_present": 0,
                "speaker_id": f"mlaad::{content_id.rsplit('_f', 1)[0]}",
                "generator": "MLAAD_original", "source": "mlaad_tiny_matched",
                "dataset": "mlaad_tiny_matched", "hf_id": MLAAD_REPO,
                "original_id": content_id, "split_group_id": group,
                "source_url": MLAAD_SOURCE, "version": MLAAD_REVISION,
                "license": "M-AILABS original corpus license (see MLAAD LICENSE)",
                "allowed_for_competition": "YES",
                "redistribution_allowed": "YES_WITH_NOTICE",
                "commercial_restriction": "NO_FOR_ORIGINAL_SUBSET",
                "dataset_name": "MLAAD-tiny-original",
                "content_hash": sha256_file(real_path),
                "near_duplicate_group": group, "data_role": role,
            })
        rows.append({
            "path": str(fake_path), "file_fake": 1, "voice_fake": 1,
            "music_fake": 0, "voice_present": 1, "music_present": 0,
            "speaker_id": f"mlaad::{content_id.rsplit('_f', 1)[0]}",
            "generator": f"MLAAD::{pair['generator']}",
            "source": "mlaad_tiny_matched", "dataset": "mlaad_tiny_matched",
            "hf_id": MLAAD_REPO, "original_id": content_id,
            "split_group_id": group, "source_url": MLAAD_SOURCE,
            "version": MLAAD_REVISION, "license": "CC BY-NC 4.0",
            "allowed_for_competition": "YES",
            "redistribution_allowed": "YES_WITH_ATTRIBUTION_NONCOMMERCIAL",
            "commercial_restriction": "YES_NONCOMMERCIAL_ONLY",
            "dataset_name": "MLAAD-tiny-fake",
            "content_hash": sha256_file(fake_path),
            "near_duplicate_group": group, "data_role": role,
        })
    return _write_manifest(rows, output)


def prepare_sonics(
    metadata_path: pathlib.Path,
    archive_paths: list[pathlib.Path],
    audio_root: pathlib.Path,
    output: pathlib.Path,
    per_generator: int,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path, low_memory=False)
    archives = [zipfile.ZipFile(path) for path in archive_paths]
    try:
        members = {}
        for archive_index, archive in enumerate(archives):
            for name in archive.namelist():
                if name.lower().endswith(".mp3"):
                    members[pathlib.PurePosixPath(name).stem] = (archive_index, name)
        available = metadata[metadata["filename"].astype(str).isin(members)].copy()
        available = available[available["algorithm"].astype(str).isin(SONICS_GENERATORS)]
        # A content ID may have multiple generated renditions. Keep groups intact
        # by selecting one rendition from distinct IDs for each generator.
        chosen = []
        used_ids: set[str] = set()
        for generator in SONICS_GENERATORS:
            target = min(per_generator, 40) if generator in {
                "chirp-v3", "udio-30s", "chirp-v2-xxl-alpha"
            } else per_generator
            candidates = available[available["algorithm"].astype(str).eq(generator)]
            candidates = candidates.sort_values(["id", "filename"])
            candidates = candidates[~candidates["id"].astype(str).isin(used_ids)]
            candidates = candidates.drop_duplicates("id").head(target)
            if len(candidates) != target:
                raise RuntimeError(f"SONICS {generator}: requested {target}, got {len(candidates)}")
            used_ids.update(candidates["id"].astype(str))
            chosen.append(candidates)
        selected = pd.concat(chosen, ignore_index=True)
        rows: list[dict] = []
        for row in selected.to_dict("records"):
            archive_index, member = members[str(row["filename"])]
            destination = audio_root / f"{row['filename']}.mp3"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                with archives[archive_index].open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
            content_id = str(row["id"])
            group = f"sonics::{content_id}"
            # part_01 consists of vocal songs. Both components are generated.
            rows.append({
                "path": str(destination), "file_fake": 1, "voice_fake": 1,
                "music_fake": 1, "voice_present": 1, "music_present": 1,
                "speaker_id": group, "generator": f"SONICS::{row['algorithm']}",
                "source": "sonics_official", "dataset": "sonics_official",
                "hf_id": "awsaf49/sonics", "original_id": str(row["filename"]),
                "split_group_id": group, "source_url": SONICS_SOURCE,
                "version": "official part_01 + metadata snapshot 2026-08-31",
                "license": "CC BY-NC 4.0", "allowed_for_competition": "YES",
                "redistribution_allowed": "YES_WITH_ATTRIBUTION_NONCOMMERCIAL",
                "commercial_restriction": "YES_NONCOMMERCIAL_ONLY",
                "dataset_name": "SONICS-fake", "content_hash": sha256_file(destination),
                "near_duplicate_group": group, "data_role": "train",
                "upstream_split": str(row["split"]),
                "upstream_label": str(row["label"]),
            })
        return _write_manifest(rows, output)
    finally:
        for archive in archives:
            archive.close()


def audit_manifests(mlaad: pd.DataFrame, sonics: pd.DataFrame) -> dict:
    train_groups = set(mlaad.loc[mlaad.data_role.eq("train"), "split_group_id"])
    val_groups = set(mlaad.loc[~mlaad.data_role.eq("train"), "split_group_id"])
    overlap = train_groups & val_groups
    if overlap:
        raise RuntimeError(f"MLAAD content leakage: {sorted(overlap)[:3]}")
    paired = mlaad.groupby(["data_role", "split_group_id"])["voice_fake"].agg(set)
    bad_pairs = paired[paired.map(lambda labels: labels != {0, 1})]
    if len(bad_pairs):
        raise RuntimeError(f"MLAAD groups without exact real/fake pair: {len(bad_pairs)}")
    return {
        "mlaad_rows": int(len(mlaad)),
        "mlaad_train_rows": int(mlaad.data_role.eq("train").sum()),
        "mlaad_unseen_validation_rows": int((~mlaad.data_role.eq("train")).sum()),
        "mlaad_train_generators": sorted(
            mlaad.loc[mlaad.data_role.eq("train") & mlaad.voice_fake.eq(1), "generator"].unique()
        ),
        "mlaad_unseen_generators": sorted(
            mlaad.loc[~mlaad.data_role.eq("train") & mlaad.voice_fake.eq(1), "generator"].unique()
        ),
        "mlaad_cross_role_group_overlap": 0,
        "mlaad_all_groups_paired_real_fake": True,
        "sonics_rows": int(len(sonics)),
        "sonics_generators": sorted(sonics.generator.unique()),
        "sonics_all_vocal_mixed_fake": bool(
            (sonics[["voice_present", "music_present", "voice_fake", "music_fake"]].eq(1)).all().all()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlaad_per_generator", type=int, default=20)
    parser.add_argument("--sonics_per_generator", type=int, default=100)
    parser.add_argument("--output_root", default="data/public_v9")
    parser.add_argument("--raw_root", default="data/raw/public_v9")
    parser.add_argument("--sonics_metadata", default="data/raw/sonics_official/fake_songs.csv")
    parser.add_argument(
        "--sonics_archives", nargs="+",
        default=[
            "data/raw/sonics_official/fake_songs/part_01.zip",
            "data/raw/sonics_official/fake_songs/part_02.zip",
        ],
    )
    args = parser.parse_args()

    output_root = pathlib.Path(args.output_root)
    raw_root = pathlib.Path(args.raw_root)
    mlaad = prepare_mlaad(
        raw_root / "mlaad_tiny", output_root / "mlaad_tiny_manifest.csv",
        args.mlaad_per_generator,
    )
    sonics = prepare_sonics(
        pathlib.Path(args.sonics_metadata), [pathlib.Path(path) for path in args.sonics_archives],
        raw_root / "sonics", output_root / "sonics_manifest.csv",
        args.sonics_per_generator,
    )
    report = audit_manifests(mlaad, sonics)
    report.update({
        "mlaad_revision": MLAAD_REVISION,
        "sonics_archives": list(args.sonics_archives),
        "notes": [
            "Candidate data only; selected v7 model and final holdout are untouched.",
            "CC BY-NC sources are suitable only for a non-commercial competition use case.",
            "SONICS real YouTube IDs were deliberately not downloaded.",
        ],
    })
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "diversity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
