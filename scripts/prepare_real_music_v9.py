#!/usr/bin/env python3
"""Create real-music manifests from official GuitarSet/Jamendo/FMA archives.

Archives are intentionally kept below ``data/raw`` and never committed.  FMA
rows are accepted only when the artist supplied an explicit Creative Commons
license that permits derivatives; CC-ND and missing-license tracks are
excluded.  The historical v7 splits and final holdout are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import tarfile
import zipfile

import pandas as pd


FMA_SOURCE = "https://github.com/mdeff/fma"
GUITARSET_SOURCE = "https://zenodo.org/records/3371780"
JAMENDO_SOURCE = "https://github.com/MTG/mtg-jamendo-dataset"
FMA_METADATA_SHA1 = "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"
FMA_SMALL_SHA1 = "ade154f733639d52e35e32f5593efe5be76c6d70"
GUITARSET_MIC_SIZE = 656927981
JAMENDO_SHARD_SHA256 = "c89826f7fa271a1c6c8f3b63786462cd72a97ec02531ca2f566310418dace88b"


def digest(path: pathlib.Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def extract_all(archive: pathlib.Path, destination: pathlib.Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".complete"
    if marker.exists():
        return
    with zipfile.ZipFile(archive) as package:
        package.extractall(destination)
    marker.write_text(digest(archive) + "\n", encoding="utf-8")


def row_for_real_music(
    path: pathlib.Path, dataset: str, source_url: str, version: str,
    license_name: str, original_id: str, group: str, artist: str,
) -> dict:
    return {
        "path": str(path), "file_fake": 0, "voice_fake": 0, "music_fake": 0,
        "voice_present": 0, "music_present": 1, "speaker_id": artist,
        "generator": f"{dataset}::real", "source": dataset, "dataset": dataset,
        "hf_id": "not_huggingface_official_archive", "original_id": original_id,
        "split_group_id": group, "source_url": source_url, "version": version,
        "license": license_name, "allowed_for_competition": "YES",
        "redistribution_allowed": "PER_TRACK_LICENSE_WITH_ATTRIBUTION",
        "commercial_restriction": "FOLLOW_RECORDED_CC_LICENSE",
        "dataset_name": dataset, "content_hash": digest(path),
        "near_duplicate_group": f"{dataset}::track::{original_id}",
        "data_role": "train",
    }


def prepare_guitarset(archive: pathlib.Path, extraction: pathlib.Path) -> pd.DataFrame:
    if archive.stat().st_size != GUITARSET_MIC_SIZE:
        raise RuntimeError(
            f"GuitarSet archive size mismatch: {archive.stat().st_size} != {GUITARSET_MIC_SIZE}"
        )
    extract_all(archive, extraction)
    audio = sorted(extraction.rglob("*.wav"))
    if len(audio) != 360:
        raise RuntimeError(f"Expected 360 GuitarSet mic recordings, got {len(audio)}")
    rows = []
    for path in audio:
        match = re.match(r"(?P<player>\d\d)_", path.stem)
        player = match.group("player") if match else "unknown"
        rows.append(row_for_real_music(
            path, "guitarset_mic", GUITARSET_SOURCE, "Zenodo record 3371780",
            "CC BY 4.0", path.stem, f"guitarset::player::{player}",
            f"guitarset_player_{player}",
        ))
    return pd.DataFrame(rows)


def _jamendo_licenses(path: pathlib.Path) -> dict[str, tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    result: dict[str, tuple[str, str]] = {}
    index = 0
    while index + 2 < len(lines):
        if re.fullmatch(r"\d{2}/\d+\.mp3", lines[index].strip()):
            track_id = pathlib.PurePosixPath(lines[index].strip()).stem
            attribution = lines[index + 1].strip()
            license_line = lines[index + 2].strip()
            match = re.search(r"(https?://creativecommons\.org/licenses/[^ ]+)", license_line)
            artist_match = re.search(r" by (.+?) from Jamendo:", attribution)
            if match:
                result[track_id] = (
                    match.group(1).rstrip("."),
                    artist_match.group(1) if artist_match else f"track_{track_id}",
                )
            index += 3
        else:
            index += 1
    return result


def prepare_jamendo(
    archive: pathlib.Path, repository: pathlib.Path,
    extraction: pathlib.Path, maximum: int,
) -> tuple[pd.DataFrame, dict]:
    if digest(archive) != JAMENDO_SHARD_SHA256:
        raise RuntimeError("MTG-Jamendo first low-quality shard SHA256 mismatch")
    license_map = _jamendo_licenses(repository / "audio_licenses.txt")
    checksum_file = repository / "data/download/raw_30s_audio-low_sha256_tracks.txt"
    expected = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        checksum, name = line.split(" ", 1)
        expected[name.strip()] = checksum

    extraction.mkdir(parents=True, exist_ok=True)
    rows = []
    artist_counts: dict[str, int] = {}
    excluded_nd_or_unknown = 0
    with tarfile.open(archive) as package:
        members = sorted(
            (member for member in package.getmembers() if member.isfile() and member.name.endswith(".mp3")),
            key=lambda member: member.name,
        )
        for member in members:
            track_id = pathlib.PurePosixPath(member.name).stem.replace(".low", "")
            license_artist = license_map.get(track_id)
            if license_artist is None or not _allowed_fma_license(license_artist[0]):
                excluded_nd_or_unknown += 1
                continue
            license_url, artist = license_artist
            artist_key = hashlib.sha256(artist.encode("utf-8")).hexdigest()[:16]
            if artist_counts.get(artist_key, 0) >= 3:
                continue
            relative = pathlib.PurePosixPath(member.name)
            output = extraction / relative.name
            if not output.exists():
                source = package.extractfile(member)
                if source is None:
                    continue
                with source, output.open("wb") as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
            expected_name = "/".join(relative.parts[-2:])
            actual = digest(output)
            if expected.get(expected_name) != actual:
                output.unlink(missing_ok=True)
                raise RuntimeError(f"MTG-Jamendo track SHA256 mismatch: {expected_name}")
            rows.append(row_for_real_music(
                output, "mtg_jamendo_cc", JAMENDO_SOURCE,
                "raw_30s audio-low shard 00", license_url,
                f"jamendo_{track_id}", f"jamendo::artist::{artist_key}",
                f"jamendo_artist_{artist_key}",
            ))
            artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1
            if len(rows) >= maximum:
                break
    report = {
        "selected_tracks": len(rows),
        "selected_artists": len(artist_counts),
        "excluded_nd_or_unknown_seen": excluded_nd_or_unknown,
        "archive_sha256": JAMENDO_SHARD_SHA256,
    }
    if len(rows) < maximum:
        raise RuntimeError(f"Only {len(rows)} eligible Jamendo tracks in shard; requested {maximum}")
    return pd.DataFrame(rows), report


def _allowed_fma_license(value: str) -> bool:
    normalized = str(value).strip().lower()
    if not normalized or normalized == "nan" or "creativecommons" not in normalized:
        return False
    return "nd" not in normalized and "no-deriv" not in normalized


def prepare_fma(
    metadata_archive: pathlib.Path,
    audio_archive: pathlib.Path,
    extraction: pathlib.Path,
    maximum: int,
) -> tuple[pd.DataFrame, dict]:
    if digest(metadata_archive, "sha1") != FMA_METADATA_SHA1:
        raise RuntimeError("FMA metadata SHA1 mismatch")
    if digest(audio_archive, "sha1") != FMA_SMALL_SHA1:
        raise RuntimeError("FMA small SHA1 mismatch")
    metadata_root = extraction / "metadata"
    audio_root = extraction / "audio"
    extract_all(metadata_archive, metadata_root)
    extract_all(audio_archive, audio_root)
    tracks_path = next(metadata_root.rglob("tracks.csv"))
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1], low_memory=False)
    small = tracks[tracks[("set", "subset")].astype(str).eq("small")].copy()
    license_column = small[("track", "license")].astype(str)
    eligible = small[license_column.map(_allowed_fma_license)].copy()
    eligible["_license"] = license_column.loc[eligible.index]
    eligible["_artist"] = eligible[("artist", "id")].astype(str)
    eligible["_genre"] = eligible[("track", "genre_top")].astype(str)

    # Round-robin genres and cap each artist so one catalogue cannot dominate.
    selected_indices: list[int] = []
    artist_counts: dict[str, int] = {}
    grouped = {name: group.sort_index() for name, group in eligible.groupby("_genre")}
    while len(selected_indices) < maximum:
        changed = False
        for genre in sorted(grouped):
            group = grouped[genre]
            for track_id, row in group.iterrows():
                if int(track_id) in selected_indices:
                    continue
                artist = str(row["_artist"])
                if artist_counts.get(artist, 0) >= 3:
                    continue
                selected_indices.append(int(track_id))
                artist_counts[artist] = artist_counts.get(artist, 0) + 1
                changed = True
                break
            if len(selected_indices) >= maximum:
                break
        if not changed:
            break

    rows = []
    missing = []
    for track_id in selected_indices:
        row = eligible.loc[track_id]
        path = audio_root / "fma_small" / f"{track_id:06d}"[:3] / f"{track_id:06d}.mp3"
        if not path.exists():
            missing.append(str(path))
            continue
        artist = str(row["_artist"])
        rows.append(row_for_real_music(
            path, "fma_small_cc", FMA_SOURCE, "rc1",
            str(row["_license"]), f"fma_{track_id:06d}",
            f"fma::artist::{artist}", f"fma_artist_{artist}",
        ))
    report = {
        "small_tracks": int(len(small)),
        "derivatives_allowed_cc_tracks": int(len(eligible)),
        "selected_tracks": int(len(rows)),
        "selected_artists": int(len({row['speaker_id'] for row in rows})),
        "missing_selected_files": missing,
    }
    return pd.DataFrame(rows), report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", default="data/raw/public_v9/downloads")
    parser.add_argument("--extract_root", default="data/raw/public_v9/real_music")
    parser.add_argument("--output", default="data/public_v9/real_music_manifest.csv")
    parser.add_argument("--fma_max", type=int, default=500)
    parser.add_argument("--jamendo_max", type=int, default=350)
    parser.add_argument("--jamendo_repo", default="data/raw/public_v9/mtg_jamendo_repo")
    parser.add_argument("--include_fma", action="store_true")
    args = parser.parse_args()
    downloads = pathlib.Path(args.downloads)
    extraction = pathlib.Path(args.extract_root)
    guitarset = prepare_guitarset(
        downloads / "guitarset_audio_mono_mic.zip", extraction / "guitarset"
    )
    jamendo, jamendo_report = prepare_jamendo(
        downloads / "raw_30s_audio-low-00.tar", pathlib.Path(args.jamendo_repo),
        extraction / "jamendo", args.jamendo_max,
    )
    frames = [guitarset, jamendo]
    fma_report = {"status": "NOT REQUESTED"}
    if args.include_fma:
        fma, fma_report = prepare_fma(
            downloads / "fma_metadata.zip", downloads / "fma_small.zip",
            extraction / "fma", args.fma_max,
        )
        frames.append(fma)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    report = {
        "rows": int(len(combined)),
        "sources": combined.dataset.value_counts().to_dict(),
        "licenses": combined.license.value_counts().to_dict(),
        "jamendo": jamendo_report,
        "fma": fma_report,
        "final_holdout": "NOT READ / NOT RUN",
    }
    (output.parent / "real_music_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
