#!/usr/bin/env python3
"""Generate a copyright-clean, project-authored procedural fake-audio corpus."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys
from typing import Iterable

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.procedural_audio_v8 import (  # noqa: E402
    GENERATOR_VERSION, MUSIC_TRAIN_FAMILIES, MUSIC_VALID_FAMILY, SAMPLE_RATE,
    VOICE_TRAIN_FAMILIES, VOICE_VALID_FAMILY, audio_stats, content_hash,
    quality_errors, synthesize_mix, synthesize_music, synthesize_voice,
)


MANIFEST_FIELDS = (
    "path", "file_fake", "voice_fake", "music_fake", "voice_present", "music_present",
    "speaker_id", "generator", "source", "dataset", "hf_id", "original_id",
    "split_group_id", "source_url", "version", "license", "allowed_for_competition",
    "redistribution_allowed", "commercial_restriction", "dataset_name", "content_hash",
    "near_duplicate_group", "audio_sha256", "generation_seed", "generator_family",
    "recommended_split", "generated_by", "external_assets_used", "base_voice_id",
    "base_music_id", "duration_sec", "sample_rate", "peak", "rms", "dc_offset",
    "clipping_fraction", "mix_snr_db",
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duration(seed: int) -> float:
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    return float(np.round(rng.uniform(3.4, 7.2), 3))


def _base_row(path: pathlib.Path, kind: str, split: str, seed: int, duration: float,
              original_id: str, group_id: str, generator: str, wave: np.ndarray) -> dict[str, object]:
    labels = {
        "voice": (1, 1, 0, 1, 0), "music": (1, 0, 1, 0, 1),
        "mix": (1, 1, 1, 1, 1),
    }[kind]
    stats = audio_stats(wave)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "file_fake": labels[0], "voice_fake": labels[1], "music_fake": labels[2],
        "voice_present": labels[3], "music_present": labels[4],
        "speaker_id": f"proc_{split}_speaker_{seed % 97:02d}" if kind in ("voice", "mix") else "",
        "generator": generator, "source": "project_procedural_v8", "dataset": "procedural_v8",
        "hf_id": "", "original_id": original_id, "split_group_id": group_id,
        "source_url": "LOCAL:tools/procedural_audio_v8.py", "version": GENERATOR_VERSION,
        "license": "PROJECT_AUTHORED_PROCEDURAL_OUTPUT_NO_EXTERNAL_ASSETS",
        "allowed_for_competition": "YES", "redistribution_allowed": "YES_PROJECT_OWNED",
        "commercial_restriction": "NO_EXTERNAL_RESTRICTION", "dataset_name": "procedural_v8",
        "content_hash": content_hash(kind, seed, duration), "near_duplicate_group": group_id,
        "audio_sha256": _sha256(path), "generation_seed": seed, "generator_family": generator,
        "recommended_split": split, "generated_by": "numpy_scipy_project_generator",
        "external_assets_used": "NO", "base_voice_id": "", "base_music_id": "",
        "duration_sec": f"{stats.duration_sec:.6f}", "sample_rate": SAMPLE_RATE,
        "peak": f"{stats.peak:.8f}", "rms": f"{stats.rms:.8f}",
        "dc_offset": f"{stats.dc_offset:.8f}", "clipping_fraction": f"{stats.clipping_fraction:.8f}",
        "mix_snr_db": "",
    }


def _write(path: pathlib.Path, wave: np.ndarray, duration: float, force: bool) -> None:
    errors = quality_errors(wave, duration)
    if errors:
        raise RuntimeError(f"quality gate failed for {path.name}: {errors}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    sf.write(path, wave, SAMPLE_RATE, subtype="PCM_16")


def _voice_rows(audio_root: pathlib.Path, split: str, count: int, seed_base: int,
                force: bool) -> Iterable[dict[str, object]]:
    renderers = VOICE_TRAIN_FAMILIES if split == "train" else (VOICE_VALID_FAMILY,)
    for index in range(count):
        seed = seed_base + index
        duration = _duration(seed)
        group = f"proc::voice::{split}::{index:05d}"
        original = f"proc_voice_{split}_{index:05d}"
        for renderer in renderers:
            wave = synthesize_voice(seed, renderer, duration)
            path = audio_root / split / "voice" / f"{original}__{renderer}.wav"
            _write(path, wave, duration, force)
            yield _base_row(path, "voice", split, seed, duration, original, group,
                            f"proc_voice_{renderer}", wave)


def _music_rows(audio_root: pathlib.Path, split: str, count: int, seed_base: int,
                force: bool) -> Iterable[dict[str, object]]:
    renderers = MUSIC_TRAIN_FAMILIES if split == "train" else (MUSIC_VALID_FAMILY,)
    for index in range(count):
        seed = seed_base + index
        duration = _duration(seed)
        group = f"proc::music::{split}::{index:05d}"
        original = f"proc_music_{split}_{index:05d}"
        for renderer in renderers:
            wave = synthesize_music(seed, renderer, duration)
            path = audio_root / split / "music" / f"{original}__{renderer}.wav"
            _write(path, wave, duration, force)
            yield _base_row(path, "music", split, seed, duration, original, group,
                            f"proc_music_{renderer}", wave)


def _mix_rows(audio_root: pathlib.Path, split: str, count: int, seed_base: int,
              force: bool) -> Iterable[dict[str, object]]:
    for index in range(count):
        mix_seed = seed_base + index
        voice_seed = seed_base + 1_000_000 + index
        music_seed = seed_base + 2_000_000 + index
        duration = _duration(mix_seed)
        group = f"proc::mix::{split}::{index:05d}"
        original = f"proc_mix_{split}_{index:05d}"
        wave, voice_renderer, music_renderer, snr_db = synthesize_mix(
            voice_seed, music_seed, split, duration, mix_seed)
        generator = f"proc_mix_{voice_renderer}+{music_renderer}"
        path = audio_root / split / "mix" / f"{original}.wav"
        _write(path, wave, duration, force)
        row = _base_row(path, "mix", split, mix_seed, duration, original, group, generator, wave)
        row["base_voice_id"] = f"proc_mix_voice_{split}_{index:05d}"
        row["base_music_id"] = f"proc_mix_music_{split}_{index:05d}"
        row["mix_snr_db"] = f"{snr_db:.4f}"
        row["content_hash"] = hashlib.sha256(
            f"mix|{voice_seed}|{music_seed}|{duration:.3f}|{GENERATOR_VERSION}".encode()).hexdigest()
        yield row


def generate(args: argparse.Namespace) -> dict[str, object]:
    audio_root = (ROOT / args.audio_root).resolve()
    manifest = (ROOT / args.manifest).resolve()
    rows: list[dict[str, object]] = []
    rows.extend(_voice_rows(audio_root, "train", args.train_voice_contents, args.seed + 10_000, args.force))
    rows.extend(_voice_rows(audio_root, "val_unseen_generator", args.val_voice_contents,
                            args.seed + 20_000, args.force))
    rows.extend(_music_rows(audio_root, "train", args.train_music_contents, args.seed + 30_000, args.force))
    rows.extend(_music_rows(audio_root, "val_unseen_generator", args.val_music_contents,
                            args.seed + 40_000, args.force))
    rows.extend(_mix_rows(audio_root, "train", args.train_mixes, args.seed + 50_000, args.force))
    rows.extend(_mix_rows(audio_root, "val_unseen_generator", args.val_mixes,
                          args.seed + 60_000, args.force))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "generator_version": GENERATOR_VERSION, "seed": args.seed, "rows": len(rows),
        "train_rows": sum(row["recommended_split"] == "train" for row in rows),
        "validation_rows": sum(row["recommended_split"] != "train" for row in rows),
        "voice_only": sum(int(row["voice_present"]) and not int(row["music_present"]) for row in rows),
        "music_only": sum(int(row["music_present"]) and not int(row["voice_present"]) for row in rows),
        "mixed": sum(int(row["voice_present"]) and int(row["music_present"]) for row in rows),
        "external_assets_used": False, "manifest": manifest.relative_to(ROOT).as_posix(),
        "audio_root": audio_root.relative_to(ROOT).as_posix(),
    }
    summary_path = manifest.parent / "generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", default="data/generated_v8/audio")
    parser.add_argument("--manifest", default="data/generated_v8/manifest.csv")
    parser.add_argument("--seed", type=int, default=23674908)
    parser.add_argument("--train-voice-contents", type=int, default=120)
    parser.add_argument("--val-voice-contents", type=int, default=30)
    parser.add_argument("--train-music-contents", type=int, default=120)
    parser.add_argument("--val-music-contents", type=int, default=30)
    parser.add_argument("--train-mixes", type=int, default=120)
    parser.add_argument("--val-mixes", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
