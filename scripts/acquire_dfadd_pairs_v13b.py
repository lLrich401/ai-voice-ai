#!/usr/bin/env python3
"""Acquire a small, content-paired DFADD/VCTK pilot outside Git.

The script intentionally downloads individual Hugging Face dataset rows instead
of the 40+ GiB DFADD archive.  Every spoof is matched to its exact VCTK
``speaker_utterance`` original and all inputs are hash-recorded.  Network/API
failures are fail-closed and never turn incomplete data into APPROVED rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile


DATASET_ID = "isjwdu/DFADD"
DATASET_URL = "https://huggingface.co/datasets/isjwdu/DFADD"
DFADD_REVISION = "dfc1eeab3cb0068db8e87a2b89a1ebd103665b1f"
VCTK_URL = "https://www.modelscope.cn/datasets/Maya23/speechfake/resolve/master/VCTK-Corpus.zip"
# Fixed offsets make the acquired pilot reproducible.  The expected family is
# checked against each returned filename, so upstream row reordering fails.
FAMILY_OFFSETS = {
    "GradTTS": 0,
    "matcha": 50_000,
    "NaturalSpeech2": 125_000,
    "StyleTTS2": 150_000,
    "pflow": 175_000,
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_json(url: str, retries: int = 10) -> dict:
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": "ai-voice-v13b/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt + 1 == retries:
                raise
            time.sleep(min(90, 8 * (attempt + 1)))
    raise AssertionError("unreachable")


def download(url: str, destination: pathlib.Path, retries: int = 5) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ai-voice-v13b/1.0"})
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            temporary.replace(destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt + 1 == retries:
                raise
            time.sleep(4 * (attempt + 1))


def family_from_name(name: str) -> str:
    stem = pathlib.Path(name).stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"spoof filename has no generator suffix: {name}")
    return "_".join(parts[2:])


def original_name(name: str) -> str:
    match = re.fullmatch(r"([a-z]\d+_\d+)(?:_.+)?\.[^.]+", name, re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot map DFADD filename to VCTK original: {name}")
    return f"{match.group(1)}.wav"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.getenv(
        "AI_VOICE_DATA_ROOT", str(pathlib.Path.home() / "ai_voice_data_v13b")))
    parser.add_argument("--per-family", type=int, default=20)
    parser.add_argument("--vctk-zip", default="")
    args = parser.parse_args()
    if not 1 <= args.per_family <= 100:
        raise ValueError("per-family must be between 1 and 100")

    root = pathlib.Path(args.data_root).resolve()
    vctk_zip = pathlib.Path(args.vctk_zip).resolve() if args.vctk_zip else (
        root / "raw/speechfake/VCTK-Corpus.zip")
    if not vctk_zip.is_file():
        raise FileNotFoundError(
            f"official VCTK archive missing: {vctk_zip}; source={VCTK_URL}")
    fake_root, real_root = root / "raw/dfadd/fake", root / "raw/dfadd/real"
    manifest_path = root / "manifests/dfadd_paired_v13b.csv"
    provenance_path = root / "provenance/dfadd_paired_v13b.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(vctk_zip) as archive:
        by_basename = {pathlib.PurePosixPath(name).name: name for name in archive.namelist()
                       if name.lower().endswith(".wav")}
        for expected_family, offset in FAMILY_OFFSETS.items():
            request_length = min(100, max(40, args.per_family * 4))
            query = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode({
                "dataset": DATASET_ID, "config": "default", "split": "train",
                "offset": offset, "length": request_length,
            })
            payload = request_json(query)
            records = payload.get("rows", [])
            if len(records) != request_length:
                raise RuntimeError(
                    f"DFADD returned {len(records)} rows at {offset}, expected {request_length}")
            accepted = 0
            for record in records:
                item = record["row"]
                name = str(item["audio_name"])
                family = family_from_name(name)
                if item.get("label") != "spoofed" or family != expected_family:
                    raise RuntimeError(
                        f"DFADD revision/order changed at row {record.get('row_idx')}: "
                        f"expected spoofed/{expected_family}, got {item.get('label')}/{family}")
                assets = item.get("audio", [])
                if len(assets) != 1 or not assets[0].get("src"):
                    raise RuntimeError(f"missing cached audio URL for DFADD row {record.get('row_idx')}")
                real_name = original_name(name)
                member = by_basename.get(real_name)
                if member is None:
                    # The SpeechFake VCTK snapshot is a documented subset.  A
                    # spoof without its exact original is not a usable pair.
                    continue
                fake_path = fake_root / family / name
                if not fake_path.is_file():
                    download(str(assets[0]["src"]), fake_path)
                speaker = real_name.split("_", 1)[0]
                real_path = real_root / speaker / real_name
                if not real_path.is_file():
                    real_path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, real_path.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)

                content = pathlib.Path(real_name).stem
                common = {
                    "source": "dfadd_vctk_paired",
                    "dataset": "DFADD/VCTK",
                    "source_url": DATASET_URL,
                    "version": DFADD_REVISION,
                    "license": "MIT (DFADD release); CC BY 4.0 (VCTK originals)",
                    "competition_use_status": "APPROVED",
                    "redistribution_allowed": "SEE_UPSTREAM_TERMS",
                    "commercial_restriction": "NONE_RECORDED",
                    "speaker_id": speaker,
                    "original_id": content,
                    "content_group": f"dfadd::{content}",
                    "near_duplicate_group": f"dfadd::{content}",
                    "split_group_id": f"dfadd::{content}",
                    "voice_present": 1,
                    "music_present": 0,
                }
                rows.append({**common, "path": str(real_path), "file_fake": 0,
                             "voice_fake": 0, "music_fake": 0, "generator": "REAL",
                             "generator_family": "REAL", "audio_sha256": sha256(real_path)})
                rows.append({**common, "path": str(fake_path), "file_fake": 1,
                             "voice_fake": 1, "music_fake": 0,
                             "generator": f"DFADD::{family}",
                             "generator_family": f"DFADD::{family}",
                             "audio_sha256": sha256(fake_path)})
                accepted += 1
                if accepted == args.per_family:
                    break
            if accepted != args.per_family:
                raise RuntimeError(
                    f"only {accepted}/{args.per_family} {expected_family} rows had exact VCTK pairs")

    # The same VCTK original appears once for each generator; retain one real
    # row per content while keeping every fake counterpart.
    real_seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for row in rows:
        if int(row["file_fake"]) == 0:
            key = str(row["content_group"])
            if key in real_seen:
                continue
            real_seen.add(key)
        unique.append(row)
    fields = list(unique[0])
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(unique)
    report = {
        "dataset": "DFADD/VCTK paired V13B pilot",
        "status": "ACQUIRED_APPROVED",
        "rows": len(unique),
        "real_rows": sum(int(row["file_fake"]) == 0 for row in unique),
        "fake_rows": sum(int(row["file_fake"]) == 1 for row in unique),
        "generator_families": sorted({str(row["generator_family"]) for row in unique
                                      if int(row["file_fake"]) == 1}),
        "content_groups": len({str(row["content_group"]) for row in unique}),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "vctk_zip": str(vctk_zip),
        "vctk_zip_sha256": sha256(vctk_zip),
        "source_urls": {"DFADD": DATASET_URL, "VCTK_archive": VCTK_URL},
        "license_evidence": {
            "DFADD": str(root / "provenance/dfadd_repo/LICENSE"),
            "DFADD_readme": str(root / "provenance/dfadd_repo/README.md"),
            "VCTK": "DFADD README records VCTK as CC-BY-4.0",
        },
    }
    provenance_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
