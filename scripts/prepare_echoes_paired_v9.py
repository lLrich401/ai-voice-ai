#!/usr/bin/env python3
"""Build content-matched real/fake music pairs from Echoes + official FMA.

Echoes distributes generated files and the exact FMA ``title - artist`` key,
but not the bona fide audio.  This script resolves that key against the
official FMA metadata archive and downloads the artist-licensed file directly
from FMA storage.  Missing, ambiguous, unknown-license, and NoDerivatives rows
are rejected before audio acquisition.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
import shutil
import time
import unicodedata
import zipfile

import pandas as pd
import requests


ECHOES_REVISION = "14b0c76c6a691c42fadfab9fb6a4eb1ee8c628a2"
ECHOES_SHA256 = "8746dcb367f2f547399201d442ffab9121c36415815947ed4784e29b60e25b59"
ECHOES_SOURCE = "https://huggingface.co/datasets/Octavian97/Echoes"
FMA_SOURCE = "https://github.com/mdeff/fma"
FMA_AUDIO_ROOT = "https://files.freemusicarchive.org/storage-freemusicarchive-org/"
UNSEEN_GENERATORS = ("diffrhythm", "songgen")


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def normalize(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", ascii_value.lower())


def allowed_license(value: str) -> bool:
    value = str(value).lower().strip()
    return bool(value and value != "nan" and "creativecommons.org/licenses/" in value
                and "-nd" not in value and "/nd" not in value)


def download(url: str, destination: pathlib.Path, attempts: int = 4) -> None:
    if destination.exists() and destination.stat().st_size > 4096:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(attempts):
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            handle.write(block)
            temporary.replace(destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)


def load_metadata(echoes_archive: pathlib.Path, fma_metadata_archive: pathlib.Path):
    if echoes_archive.stat().st_size != 8598345242 or digest(echoes_archive) != ECHOES_SHA256:
        raise RuntimeError("Echoes archive does not match the pinned Hugging Face LFS object")
    echoes_zip = zipfile.ZipFile(echoes_archive)
    echoes = pd.read_csv(echoes_zip.open("Echoes/dataset_manifest.csv"))
    fma_zip = zipfile.ZipFile(fma_metadata_archive)
    fma = pd.read_csv(fma_zip.open("fma_metadata/raw_tracks.csv"), low_memory=False)
    fma_zip.close()
    index: dict[str, list[dict]] = collections.defaultdict(list)
    for row in fma.to_dict("records"):
        key = normalize(f"{row['track_title']} - {row['artist_name']}")
        if allowed_license(row.get("license_url")):
            index[key].append(row)
    return echoes_zip, echoes, index


def select_rows(echoes: pd.DataFrame, fma_index: dict, per_generator: int):
    selected = []
    generators = sorted(echoes.generator.astype(str).unique())
    val_content: set[str] = set()

    def choose(generator, role, forbidden):
        candidates = echoes[echoes.generator.astype(str).eq(generator)].sort_values(
            ["original_audio", "type", "path_in_dataset"]
        )
        count = 0
        chosen_keys: set[str] = set()
        for row in candidates.to_dict("records"):
            key = normalize(row["original_audio"])
            matches = fma_index.get(key, [])
            if key in forbidden or key in chosen_keys or len(matches) != 1:
                continue
            chosen_keys.add(key)
            selected.append({**row, "data_role": role, "fma": matches[0]})
            if role != "train":
                val_content.add(key)
            count += 1
            if count == per_generator:
                break
        if count != per_generator:
            raise RuntimeError(f"Echoes {generator}: only {count} unique licensed FMA pairs")
    # Validation generators may share content with each other, but every train
    # generator excludes the complete validation content set.
    for generator in UNSEEN_GENERATORS:
        choose(generator, "val_b_unseen_music_generator", set())
    for generator in generators:
        if generator not in UNSEEN_GENERATORS:
            choose(generator, "train", val_content)
    return selected


def manifest_row(path, fake, generator, role, fma, echoes_row):
    track_id = int(fma["track_id"])
    group = f"echoes_fma::{track_id}"
    license_url = str(fma["license_url"])
    return {
        "path": str(path), "file_fake": int(fake), "voice_fake": 0,
        "music_fake": int(fake), "voice_present": 0, "music_present": 1,
        "speaker_id": f"fma_artist_{int(fma['artist_id'])}",
        "generator": generator, "source": "echoes_fma_paired",
        "dataset": "echoes_fma_paired", "hf_id": "Octavian97/Echoes + official FMA",
        "original_id": f"fma_{track_id}", "split_group_id": group,
        "source_url": f"{ECHOES_SOURCE} ; {FMA_SOURCE}",
        "version": f"Echoes {ECHOES_REVISION}; FMA metadata rc1",
        "license": f"generated CC BY-SA 4.0; original {license_url}",
        "allowed_for_competition": "YES",
        "redistribution_allowed": "FOLLOW_CC_ATTRIBUTION_AND_SHAREALIKE",
        "commercial_restriction": "FOLLOW_ORIGINAL_FMA_TRACK_LICENSE",
        "dataset_name": "Echoes-FMA-paired", "content_hash": digest(path),
        "near_duplicate_group": group, "data_role": role,
        "upstream_split": str(echoes_row.get("type", "not_applicable")),
        "upstream_label": "fake" if fake else "real",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--echoes_archive", default="data/raw/public_v9/downloads/Echoes.zip")
    parser.add_argument("--fma_metadata", default="data/raw/public_v9/downloads/fma_metadata.zip")
    parser.add_argument("--audio_root", default="data/raw/public_v9/echoes_fma_paired")
    parser.add_argument("--output", default="data/public_v9/echoes_fma_paired_manifest.csv")
    parser.add_argument("--per_generator", type=int, default=20)
    args = parser.parse_args()

    archive, echoes, fma_index = load_metadata(
        pathlib.Path(args.echoes_archive), pathlib.Path(args.fma_metadata)
    )
    selected = select_rows(echoes, fma_index, args.per_generator)
    audio_root = pathlib.Path(args.audio_root)
    rows = []
    for item in selected:
        fma = item["fma"]
        track_id = int(fma["track_id"])
        role = item["data_role"]
        real_path = audio_root / "real" / f"fma_{track_id}.mp3"
        download(FMA_AUDIO_ROOT + str(fma["track_file"]), real_path)
        fake_member = "Echoes/" + str(item["path_in_dataset"]).lstrip("/")
        fake_path = audio_root / "fake" / pathlib.PurePosixPath(item["path_in_dataset"]).name
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        if not fake_path.exists():
            with archive.open(fake_member) as source, fake_path.open("wb") as target:
                shutil.copyfileobj(source, target)
        rows.append(manifest_row(real_path, False, "Echoes::FMA_original", role, fma, item))
        rows.append(manifest_row(
            fake_path, True, f"Echoes::{item['generator']}", role, fma, item
        ))
    archive.close()
    frame = pd.DataFrame(rows).drop_duplicates(
        ["path", "data_role", "generator", "music_fake"]
    ).sort_values(
        ["data_role", "split_group_id", "music_fake"]
    ).reset_index(drop=True)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    train_groups = set(frame.loc[frame.data_role.eq("train"), "split_group_id"])
    val_groups = set(frame.loc[~frame.data_role.eq("train"), "split_group_id"])
    paired = frame.groupby(["data_role", "split_group_id"])["music_fake"].agg(set)
    if train_groups & val_groups or not paired.map(lambda value: value == {0, 1}).all():
        raise RuntimeError("Echoes-FMA pair/group audit failed")
    report = {
        "rows": len(frame),
        "content_groups": int(frame.split_group_id.nunique()),
        "train_content_groups": int(frame.loc[frame.data_role.eq("train"), "split_group_id"].nunique()),
        "unseen_validation_content_groups": int(frame.loc[~frame.data_role.eq("train"), "split_group_id"].nunique()),
        "train_generators": sorted(frame.loc[
            frame.data_role.eq("train") & frame.music_fake.eq(1), "generator"].unique()),
        "unseen_generators": sorted(frame.loc[
            ~frame.data_role.eq("train") & frame.music_fake.eq(1), "generator"].unique()),
        "cross_role_group_overlap": 0, "all_groups_paired": True,
        "echoes_archive_sha256": ECHOES_SHA256,
        "final_holdout": "NOT READ / NOT RUN",
    }
    (output.parent / "echoes_fma_paired_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
