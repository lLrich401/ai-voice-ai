import hashlib
import json
import pathlib
import zipfile

import pandas as pd
import pytest

from tools.v13_guards import (
    assert_final_holdout_v13_forbidden,
    assert_no_hidden_test_training_path,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPECTED_COLUMNS = [
    "id", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def split(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/splits_v13" / name)


def test_submission_interface_unchanged():
    contract = json.loads((ROOT / "archive/pre_v13_selected/submission_interface.json").read_text())
    assert contract["output_columns"] == EXPECTED_COLUMNS
    assert contract["archive_top_level"] == ["model", "script.py", "requirements.txt"]
    assert digest(ROOT / "script.py") == contract["script_sha256"]


def test_output_schema_exact():
    source = (ROOT / "script.py").read_text(encoding="utf-8")
    for column in EXPECTED_COLUMNS[1:]:
        assert column in source


def test_no_test_batch_rank():
    source = (ROOT / "script.py").read_text(encoding="utf-8").lower()
    assert "rankdata" not in source
    assert "percentile" not in source
    assert "quantile" not in source
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text())
    assert all(str(value).lower() != "rank" for value in weights.values())


def test_independent_file_prediction():
    source = (ROOT / "script.py").read_text(encoding="utf-8")
    assert "for index,(audio_path,(wave,sr)) in enumerate(zip(audio_paths,loaded))" in source
    assert "def infer_file(" in source
    assert "[pathlib.Path(audio_path)]" in source
    assert "batch percentile" not in source.lower()


def test_no_hidden_test_training_path():
    assert_no_hidden_test_training_path("data/splits_v13/train_approved.csv")
    with pytest.raises(RuntimeError, match="test data cannot be used"):
        assert_no_hidden_test_training_path("data/test/TEST_0001.wav")


def test_final_holdout_v13_guard():
    with pytest.raises(RuntimeError, match="V13 final holdout is sealed"):
        assert_final_holdout_v13_forbidden("data/splits_v13/final_holdout_v13.csv")


def test_train_val_source_disjoint():
    train, validation = split("train.csv"), split("val_source_disjoint_review.csv")
    assert set(train.source).isdisjoint(set(validation.source))


def test_train_val_generator_disjoint():
    train, validation = split("train.csv"), split("val_generator_disjoint.csv")
    train_fake = set(train.loc[train.file_fake.eq(1), "generator_family"])
    val_fake = set(validation.loc[validation.file_fake.eq(1), "generator_family"])
    assert train_fake.isdisjoint(val_fake)


def test_content_group_isolation():
    train = split("train.csv")
    for name in ("val_generator_disjoint.csv", "val_source_disjoint_review.csv", "cal_v13.csv"):
        other = split(name)
        assert set(train.content_group).isdisjoint(set(other.content_group))


def test_source_shortcut_audit_is_fail_closed():
    report = json.loads((ROOT / "experiments/v13/source_shortcut_audit_v13_pilot.json").read_text())
    assert report["threshold"] == pytest.approx(0.75)
    assert report["status"] == "FAIL_SOURCE_SHORTCUT"
    assert max(report["metadata_only"]["auc"], report["combined"]["auc"]) > report["threshold"]


def test_component_fake_not_presence_gated():
    source = (ROOT / "script.py").read_text(encoding="utf-8")
    assert "voice_fake *= voice_present" not in source
    assert "music_fake *= music_present" not in source


def test_partial_fake_file_detection_labels():
    frame = pd.read_csv(ROOT / "data/splits/fusion_calibration.csv")
    partial = frame[frame.mix_mode.astype(str).str.contains("partial|crossfade", case=False, regex=True)]
    assert len(partial) > 0
    expected = partial[["voice_fake", "music_fake"]].max(axis=1)
    assert partial.file_fake.eq(expected).all()
    assert partial.file_fake.eq(1).any()


def test_runtime_model_hashes():
    frozen = json.loads((ROOT / "archive/pre_v13_selected/artifact_manifest.json").read_text())
    for relative in ("script.py", "requirements.txt", "model/best.pt",
                     "model/music_best.pt", "model/fusion_weights.json"):
        assert digest(ROOT / relative) == frozen["selected_artifacts"][relative]["sha256"]
    for relative, metadata in frozen["runtime_source"].items():
        assert digest(ROOT / relative) == metadata["sha256"]


def test_archive_top_level_exact():
    frozen = json.loads((ROOT / "archive/pre_v13_selected/artifact_manifest.json").read_text())
    assert frozen["frozen_zip"]["top_level"] == ["model", "requirements.txt", "script.py"]
    archive = ROOT / "submit.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as handle:
            top = {name.rstrip("/").split("/")[0] for name in handle.namelist() if name}
        assert top == {"model", "requirements.txt", "script.py"}
