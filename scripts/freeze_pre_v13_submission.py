#!/usr/bin/env python3
"""Freeze the exact pre-V13 selected submission and its external interface."""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive/pre_v13_selected"
SOURCE_ZIP = ROOT / "submit.zip"
FROZEN_ZIP = ARCHIVE / "submit.zip"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def main() -> None:
    if not SOURCE_ZIP.is_file():
        raise FileNotFoundError("submit.zip must be built and validated before V13 freeze")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(SOURCE_ZIP)
    if FROZEN_ZIP.exists():
        if sha256(FROZEN_ZIP) != source_hash:
            raise RuntimeError("pre-V13 frozen ZIP already exists with different bytes")
    else:
        shutil.copy2(SOURCE_ZIP, FROZEN_ZIP)

    selected = [
        "script.py",
        "requirements.txt",
        "model/best.pt",
        "model/music_best.pt",
        "model/fusion_weights.json",
        "model/df_arena/df_arena_1b_int8.onnx",
        "model/panns/Cnn14_16k_mAP=0.438.pth",
    ]
    artifacts = {}
    for relative in selected:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        artifacts[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    runtime_files = sorted(
        path for path in (ROOT / "model/runtime/src").rglob("*.py") if path.is_file())
    runtime = {
        path.relative_to(ROOT).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in runtime_files
    }
    with zipfile.ZipFile(FROZEN_ZIP) as archive:
        names = archive.namelist()
        top_level = sorted({name.rstrip("/").split("/")[0] for name in names if name})
        if top_level != ["model", "requirements.txt", "script.py"]:
            raise RuntimeError(f"unexpected submission top-level: {top_level}")
        members = {info.filename: info.file_size for info in archive.infolist()}

    configuration = json.loads((ROOT / "model/fusion_weights.json").read_text(
        encoding="utf-8"))
    payload = {
        "status": "FROZEN_PRE_V13_SELECTED",
        "competition": "DACON 236749 Audio Deepfake Detection",
        "actual_public_baseline": {
            "submission": "TEST5",
            "total": 0.6684394603,
            "ads": 0.6386349206,
            "cps": 0.9366803175,
            "runtime": "30m52s",
            "source": "user-reported public leaderboard result",
        },
        "git": {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current")},
        "frozen_zip": {
            "path": "archive/pre_v13_selected/submit.zip",
            "bytes": FROZEN_ZIP.stat().st_size,
            "sha256": source_hash,
            "top_level": top_level,
            "member_count": len(members),
        },
        "selected_artifacts": artifacts,
        "runtime_source": runtime,
        "fusion_configuration": configuration,
        "restore": (
            "Copy archive/pre_v13_selected/submit.zip to submit.zip and extract only when "
            "selected working-tree artifacts must be restored byte-for-byte."
        ),
        "final_holdout": "NOT READ / NOT RUN",
    }
    (ARCHIVE / "artifact_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    interface = {
        "status": "FROZEN_EXTERNAL_CONTRACT",
        "entrypoint": "python script.py --test_dir ./data/test --output ./output/submission.csv",
        "default_test_dir": "./data/test",
        "default_output": "./output/submission.csv",
        "sample_submission_mapping": "exact sample ID to audio filename stem; duplicates/missing IDs fail",
        "fallback_id_mapping": "sorted audio filename stem when sample_submission.csv is absent",
        "output_columns": [
            "id", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
            "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
        ],
        "prediction_contract": "five finite probabilities clipped to [1e-6, 1-1e-6]",
        "independence": "each file prediction uses only segments from that file; no batch rank/statistics",
        "archive_top_level": ["model", "script.py", "requirements.txt"],
        "offline": True,
        "script_sha256": artifacts["script.py"]["sha256"],
    }
    (ARCHIVE / "submission_interface.json").write_text(
        json.dumps(interface, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "archive": str(FROZEN_ZIP), "sha256": source_hash,
        "top_level": top_level, "runtime_files": len(runtime),
    }, indent=2))


if __name__ == "__main__":
    main()
