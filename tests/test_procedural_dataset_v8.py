import argparse
import ast
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import generate_procedural_v8 as generator
from tools.prepare_v8_training_manifest import prepare
from tools.procedural_audio_v8 import (
    MUSIC_TRAIN_FAMILIES,
    MUSIC_VALID_FAMILY,
    SAMPLE_RATE,
    VOICE_TRAIN_FAMILIES,
    VOICE_VALID_FAMILY,
    quality_errors,
    synthesize_mix,
    synthesize_music,
    synthesize_voice,
)


def test_procedural_audio_is_deterministic_and_passes_quality_gate():
    left = synthesize_voice(1742, "additive_formant", 3.25)
    right = synthesize_voice(1742, "additive_formant", 3.25)
    music = synthesize_music(9281, "fm_percussive", 3.25)
    mixed, _, _, _ = synthesize_mix(51, 72, "train", 3.25, 99)
    assert np.array_equal(left, right)
    assert len(left) == int(3.25 * SAMPLE_RATE)
    assert quality_errors(left, 3.25) == []
    assert quality_errors(music, 3.25) == []
    assert quality_errors(mixed, 3.25) == []


def test_unseen_generator_families_are_disjoint():
    assert VOICE_VALID_FAMILY not in VOICE_TRAIN_FAMILIES
    assert MUSIC_VALID_FAMILY not in MUSIC_TRAIN_FAMILIES


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("voice", (1, 1, 0, 1, 0)),
        ("music", (1, 0, 1, 0, 1)),
        ("mix", (1, 1, 1, 1, 1)),
    ],
)
def test_generated_label_contract(monkeypatch, tmp_path, kind, expected):
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    path = tmp_path / f"{kind}.wav"
    wave = np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE).astype(np.float32) * 0.7
    sf.write(path, wave, SAMPLE_RATE, subtype="PCM_16")
    row = generator._base_row(path, kind, "train", 11, 3.0, "original", "group", "family", wave)
    assert tuple(int(row[column]) for column in (
        "file_fake", "voice_fake", "music_fake", "voice_present", "music_present")) == expected
    assert row["external_assets_used"] == "NO"
    assert row["allowed_for_competition"] == "YES"


def test_generator_imports_no_network_model_or_media_asset_library():
    path = pathlib.Path(generator.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"requests", "urllib", "torch", "torchaudio", "librosa", "transformers"}


def _generated_frame(split: str, group: str, family: str, safe: bool = True) -> pd.DataFrame:
    return pd.DataFrame([{
        "path": f"audio/{group}.wav", "file_fake": 1, "voice_fake": 1,
        "music_fake": 0, "voice_present": 1, "music_present": 0,
        "recommended_split": split, "external_assets_used": "NO",
        "allowed_for_competition": "YES" if safe else "UNKNOWN",
        "split_group_id": group, "near_duplicate_group": group,
        "original_id": group, "speaker_id": f"speaker_{group}",
        "generator_family": family,
    }])


def test_prepare_candidate_keeps_selected_split_untouched(tmp_path):
    base_path = tmp_path / "base.csv"
    generated_path = tmp_path / "generated.csv"
    output = tmp_path / "candidate"
    base = pd.DataFrame([{"path": "real.wav", "file_fake": 0, "split_group_id": "base"}])
    base.to_csv(base_path, index=False)
    generated = pd.concat([
        _generated_frame("train", "train_group", "train_family"),
        _generated_frame("val_unseen_generator", "valid_group", "valid_family"),
    ], ignore_index=True)
    generated.to_csv(generated_path, index=False)
    before = base_path.read_bytes()
    report = prepare(base_path, generated_path, output)
    assert base_path.read_bytes() == before
    assert report["selected_v7_splits_modified"] is False
    assert len(pd.read_csv(output / "train.csv")) == 2
    assert len(pd.read_csv(output / "stress_unseen_fake.csv")) == 1


def test_prepare_rejects_unknown_license_or_split_leakage(tmp_path):
    base_path = tmp_path / "base.csv"
    base_path.write_text("path,file_fake\nreal.wav,0\n", encoding="utf-8")
    unsafe = _generated_frame("train", "unsafe", "family", safe=False)
    unsafe_path = tmp_path / "unsafe.csv"
    unsafe.to_csv(unsafe_path, index=False)
    with pytest.raises(ValueError, match="unsafe generated provenance"):
        prepare(base_path, unsafe_path, tmp_path / "unsafe_output")

    leaked = pd.concat([
        _generated_frame("train", "same_group", "train_family"),
        _generated_frame("val_unseen_generator", "same_group", "valid_family"),
    ], ignore_index=True)
    leaked_path = tmp_path / "leaked.csv"
    leaked.to_csv(leaked_path, index=False)
    with pytest.raises(ValueError, match="overlap"):
        prepare(base_path, leaked_path, tmp_path / "leaked_output")


def test_prepare_marks_high_fingerprint_risk_as_not_authorized(tmp_path):
    base_path = tmp_path / "base.csv"
    generated_path = tmp_path / "generated.csv"
    risk_path = tmp_path / "risk.json"
    pd.DataFrame([{"path": "real.wav", "file_fake": 0}]).to_csv(base_path, index=False)
    pd.concat([
        _generated_frame("train", "train_group", "train_family"),
        _generated_frame("val_unseen_generator", "valid_group", "valid_family"),
    ]).to_csv(generated_path, index=False)
    risk_path.write_text(json.dumps({"components": [{"component": "voice", "risk": "HIGH"}]}),
                         encoding="utf-8")
    report = prepare(base_path, generated_path, tmp_path / "candidate", risk_path)
    assert report["training_authorized"] is False
    assert report["adoption_status"] == "REJECT_CURRENT_DATASET_HIGH_SOURCE_FINGERPRINT"
    assert report["high_risk_components"] == ["voice"]
