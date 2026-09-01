import json
import pathlib

import pandas as pd
import pytest

from src.distillation import source_balanced_weights
from tools.v13_guards import assert_final_holdout_v13b_forbidden
from scripts.prepare_dataset_v13b import apply_explicit_approval, enrich


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/splits_v13b"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(SPLITS / name)


def test_v13b_source_shortcut_gate_is_fail_closed():
    report = json.loads((ROOT / "experiments/v13b/source_shortcut_audit.json").read_text())
    assert report["hard_threshold"] == pytest.approx(0.75)
    assert report["final_holdout_v13b"] == "NOT READ / NOT RUN"
    if report["status"] == "PASS":
        assert report["metadata_only"]["auc"] <= 0.75
        assert report["acoustic_only"]["auc"] <= 0.75
        assert report["combined"]["auc"] <= 0.75
    else:
        assert report["decision"] == "MODEL_TRAINING_BLOCKED"


def test_v13b_real_fake_source_balance():
    core = read("train.csv").query("data_role == 'paired_core'")
    counts = core.groupby(["source", "file_fake"]).size().unstack(fill_value=0)
    assert set(counts.columns) == {0, 1}
    assert (counts[0] == counts[1]).all()


def test_v13b_paired_content_group():
    manifest = read("manifest.csv")
    core = manifest.query("data_role == 'paired_core'")
    labels = core.groupby(["v13b_role", "source", "content_group"]).file_fake.agg(set)
    assert labels.map(lambda value: value == {0, 1}).all()


def test_v13b_train_val_content_disjoint():
    train = read("train.csv").query("data_role == 'paired_core'")
    val = read("val_generator_disjoint.csv")
    for column in ("content_group", "split_group_id", "near_duplicate_group",
                   "audio_sha256", "source_audio_sha256"):
        assert set(train[column].dropna()) .isdisjoint(set(val[column].dropna()))


def test_v13b_train_val_generator_disjoint():
    train = read("train.csv").query("data_role == 'paired_core' and file_fake == 1")
    val = read("val_generator_disjoint.csv").query("file_fake == 1")
    assert set(train.generator_family).isdisjoint(set(val.generator_family))


def test_grouped_shortcut_cv():
    source = (ROOT / "tools/audit_source_shortcut_v13b.py").read_text(encoding="utf-8")
    assert "StratifiedGroupKFold" in source
    assert "groups=frame.content_group" in source
    assert "StratifiedKFold" not in source.replace("StratifiedGroupKFold", "")


def test_mixed_generator_ancestry_preserved():
    mixed = read("mixed_train.csv")
    required = {"voice_generator_family", "music_generator_family", "base_voice_id",
                "base_music_id", "voice_content_group", "music_content_group",
                "parent_real_id", "parent_fake_id"}
    assert required <= set(mixed.columns)
    assert mixed[list(required)].notna().all().all()
    assert (mixed.voice_content_group == mixed.base_voice_id).all()
    assert (mixed.music_content_group == mixed.base_music_id).all()


def test_base_audio_cross_split_leakage():
    report = json.loads((ROOT / "experiments/v13b/ancestry_leakage_audit.json").read_text())
    assert report["all_development_overlap_zero"]
    for audit in (report["audits"]["generator_disjoint"], report["audits"]["calibration"]):
        assert audit["pass"]
        assert all(item["count"] == 0 for item in audit["overlap"].values())


def test_final_global_history_disjoint():
    report = json.loads((ROOT / "experiments/v13b/global_history_holdout_audit.json").read_text())
    if report["status"].startswith("NOT CREATED"):
        assert report["metrics"] == "NOT READ / NOT RUN"
    else:
        assert report["status"] == "PASS"
        assert report["overlap_count"] == 0


def test_v13b_final_holdout_guard():
    with pytest.raises(RuntimeError, match="V13B final holdout is forbidden"):
        assert_final_holdout_v13b_forbidden("data/splits_v13b/final_holdout_v13b.csv")


def test_v13b_final_holdout_global_history_disjoint_fail_closed():
    report = json.loads((SPLITS / "DATASET_V13B.json").read_text())
    candidate = SPLITS / "final_holdout_v13b.csv"
    if not candidate.exists():
        assert not report["structural_gates"]["final_holdout_v13b_sealed"]
        assert report["status"] == "DATASET_NOT_READY"
    else:
        final = pd.read_csv(candidate)
        historical_sources = set()
        for path in ROOT.glob("data/splits_v*/manifest.csv"):
            if path.parent.name == "splits_v13b":
                continue
            frame = pd.read_csv(path, usecols=lambda column: column == "source")
            historical_sources.update(frame.source.dropna().astype(str))
        assert set(final.source.astype(str)).isdisjoint(historical_sources)


def test_v13b_partial_fake_generation_and_labels():
    partial = read("partial_train.csv")
    assert set(partial.partial_fake_ratio.round(2)) == {0.02, 0.05, 0.10, 0.20,
                                                        0.30, 0.50, 0.70, 1.00}
    assert set(partial.data_role) == {"partial_fake", "partial_real_control"}
    assert (partial.file_fake == partial[["voice_fake", "music_fake"]].max(axis=1)).all()
    counts = partial.groupby([partial.source, "file_fake"]).size().unstack(fill_value=0)
    assert (counts[0] == counts[1]).all()


def test_v13b_rr_rf_fr_ff_balance():
    mixed = read("mixed_train.csv")
    counts = mixed.mix_state.value_counts()
    assert set(counts.index) == {"RR", "RF", "FR", "FF"}
    assert counts.nunique() == 1
    expected = mixed[["voice_fake", "music_fake"]].max(axis=1)
    assert mixed.file_fake.eq(expected).all()


def test_v13b_same_channel_policy_both_labels():
    core = read("manifest.csv").query("data_role == 'paired_core'")
    assert set(core.channel_policy) == {"canonical16k_pcm16_label_independent_v1"}
    assert set(core.sample_rate) == {16000}
    assert set(core.codec) == {"PCM_S16LE"}
    for _, group in core.groupby("source"):
        assert set(group.file_fake) == {0, 1}


def test_v13b_production_rows_are_explicitly_approved():
    core = read("manifest.csv").query("data_role == 'paired_core'")
    assert set(core.competition_use_status) == {"APPROVED"}
    evidence = {"approval_basis", "license_source", "license_snapshot_sha256", "reviewed_at"}
    assert evidence <= set(core.columns)
    assert core[list(evidence)].fillna("").ne("").all().all()


def test_unknown_license_exclusion_and_no_yes_auto_approval():
    row = read("train.csv").query("data_role == 'paired_core'").iloc[[0]].copy()
    row["competition_use_status"] = "YES"
    assert enrich(row).competition_use_status.iloc[0] == "YES"
    registry = {"sources": {"unknown": {"status": "APPROVED"}}}
    with pytest.raises(RuntimeError, match="approval evidence incomplete"):
        apply_explicit_approval(row, "unknown", registry)


def test_every_v13b_registry_approval_has_evidence():
    registry = json.loads((ROOT / "configs/v13b/source_registry.json").read_text())
    required = {"license", "approval_basis", "license_source",
                "license_snapshot_sha256", "reviewed_at"}
    for name, entry in registry["sources"].items():
        if entry.get("status") == "APPROVED":
            assert required <= set(entry), name
            assert all(str(entry[key]).strip() for key in required), name


def test_v13b_source_generator_class_sampler():
    core = read("train.csv").query("data_role == 'paired_core'").reset_index(drop=True)
    for task in ("voice", "music"):
        weights = source_balanced_weights(core, task)
        assert weights.sum() == pytest.approx(1.0)
        assert (weights >= 0).all()


def test_v13b_model_training_blocked_when_any_data_gate_fails():
    report = json.loads((SPLITS / "DATASET_V13B.json").read_text())
    failed = [name for name, passed in report["structural_gates"].items() if not passed]
    if failed:
        assert report["status"] == "DATASET_NOT_READY"
        assert report["model_training"] == "BLOCKED_BY_DATA_GATES"


def test_v13b_stage_manager_blocks_after_failed_data_stage():
    status = json.loads((ROOT / "experiments/v13b/stage_status.json").read_text())
    assert set(status["stages"].values()) <= {"PASS", "PASS_WITH_WARNING", "FAIL", "NOT_RUN"}
    if status["stages"]["4_validation_source_acquisition"] == "FAIL":
        assert status["model_training"] == "BLOCKED_BY_DATA_GATES"
        for index in range(5, 16):
            key = next(name for name in status["stages"] if name.startswith(f"{index}_"))
            assert status["stages"][key] == "NOT_RUN"
