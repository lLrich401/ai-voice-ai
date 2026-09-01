import json
import pathlib

import pandas as pd
import pytest

from src.distillation import source_balanced_weights
from tools.v13_guards import assert_final_holdout_v13b_forbidden
from scripts.prepare_dataset_v13b import (
    apply_explicit_approval, enrich, explicit_generator_roles,
)
from scripts.manage_v13b_stages import evaluate_gates
from scripts.complete_v13b_gates import (
    validate_final, validate_paired_music, validate_source_disjoint,
)


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


def test_exploratory_training_allowed_before_adoption_gates():
    dataset = json.loads((SPLITS / "DATASET_V13B.json").read_text())
    shortcut = json.loads((ROOT / "experiments/v13b/source_shortcut_audit.json").read_text())
    policy = json.loads((ROOT / "configs/v13b/selection_policy.json").read_text())
    gates = evaluate_gates(dataset, shortcut, policy)
    assert gates["exploratory_allowed"]
    assert not gates["adoption_eligible"]


def test_shortcut_gate_is_symmetric_around_random():
    dataset = json.loads((SPLITS / "DATASET_V13B.json").read_text())
    shortcut = json.loads((ROOT / "experiments/v13b/source_shortcut_audit.json").read_text())
    policy = json.loads((ROOT / "configs/v13b/selection_policy.json").read_text())
    inverted = json.loads(json.dumps(shortcut))
    for section in ("metadata_only", "acoustic_only", "combined"):
        inverted[section]["auc"] = 1.0 - shortcut[section]["auc"]
    assert evaluate_gates(dataset, shortcut, policy) == evaluate_gates(dataset, inverted, policy)


def test_adoption_still_blocked_without_source_disjoint():
    status = json.loads((ROOT / "experiments/v13b/stage_status.json").read_text())
    assert set(status["stages"].values()) <= {"PASS", "PASS_WITH_WARNING", "FAIL", "NOT_RUN"}
    assert status["exploratory_training"]["status"] == "ALLOWED_NOT_SELECTED"
    assert status["adoption"]["status"] == "BLOCKED_BY_DATA_GATES"
    assert not status["adoption"]["checks"][
        "approved_metric_complete_source_disjoint_validation"]


def test_bootstrap_gate_at_least_0_65():
    policy = json.loads((ROOT / "configs/v13b/selection_policy.json").read_text())
    adoption = policy["adoption_gates"]
    assert adoption["bootstrap_win_rate_minimum"] >= 0.65
    assert adoption["paired_bootstrap_replicates_minimum"] >= 1000
    assert adoption["bootstrap_ads_median_delta_minimum"] >= 0.0


def test_pre_submission_runtime_gate():
    policy = json.loads((ROOT / "configs/v13b/selection_policy.json").read_text())
    adoption = policy["adoption_gates"]
    assert adoption["pre_submission_measured_runtime_minutes_maximum"] <= 55
    assert adoption["post_submission_official_runtime_minutes_maximum"] <= 60


def test_generator_split_config_stable():
    config = json.loads((ROOT / "configs/v13b/generator_split.json").read_text())
    for source, entry in config["sources"].items():
        roles = [*entry["train"], *entry["generator_val"], *entry["cal"]]
        assert len(roles) == len(set(roles)), source
        fake = pd.DataFrame({"file_fake": [1] * len(roles), "generator_family": roles})
        first = explicit_generator_roles(source, fake, config)
        second = explicit_generator_roles(source, fake.sample(frac=1, random_state=7), config)
        assert first == second


def test_unknown_generator_not_auto_assigned():
    config = json.loads((ROOT / "configs/v13b/generator_split.json").read_text())
    fake = pd.DataFrame({"file_fake": [1], "generator_family": ["new::unreviewed"]})
    with pytest.raises(RuntimeError, match="unknown generators require explicit review"):
        explicit_generator_roles("mlaad_tiny_matched", fake, config)


def test_rendered_partial_control_same_pipeline():
    partial = read("partial_train.csv")
    positive = partial.query("data_role == 'partial_fake'").sort_values("content_group")
    control = partial.query("data_role == 'partial_real_control'").sort_values("content_group")
    for column in ("partial_fake_ratio", "partial_fake_position", "partial_crossfade_sec"):
        assert positive[column].reset_index(drop=True).equals(control[column].reset_index(drop=True))
    source = (ROOT / "src/dataset.py").read_text(encoding="utf-8")
    assert source.count("wave=render_partial_fake_wave(") == 1


def test_rendered_mix_shortcut_audit():
    report = json.loads((ROOT / "experiments/v13b/rendered_training_shortcut_audit.json").read_text())
    assert report["status"] == "PASS"
    for key, raw in report["raw_auc"].items():
        assert report["effective_auc"][key] == pytest.approx(max(raw, 1.0 - raw))
        assert report["distance_from_random"][key] == pytest.approx(abs(raw - 0.5))
    assert report["effective_auc"]["partial_fake_vs_control"] <= 0.75
    assert report["effective_auc"]["mixed_rr_vs_fake_states"] <= 0.75
    assert report["final_holdout"] == "NOT READ / NOT RUN"


def _approved_rows(source: str) -> pd.DataFrame:
    labels = [(0, 0, 0, 0, 0), (0, 0, 0, 1, 0), (1, 1, 0, 1, 0),
              (0, 0, 0, 0, 1), (1, 0, 1, 0, 1)]
    rows = []
    for index, (file_fake, voice_fake, music_fake, voice_present, music_present) in enumerate(labels):
        rows.append({
            "path": f"{source}_{index}.wav", "source": source, "dataset": source,
            "file_fake": file_fake, "voice_fake": voice_fake, "music_fake": music_fake,
            "voice_present": voice_present, "music_present": music_present,
            "content_group": f"{source}_content_{index}",
            "split_group_id": f"{source}_split_{index}",
            "near_duplicate_group": f"{source}_near_{index}",
            "base_audio_id": f"{source}_base_{index}",
            "audio_sha256": f"{source}_sha_{index}",
            "source_audio_sha256": f"{source}_source_sha_{index}",
            "generator_family": f"{source}_generator_{index}" if file_fake else "REAL",
            "voice_generator_family": f"{source}_voice_gen" if voice_fake else "ABSENT",
            "music_generator_family": f"{source}_music_gen" if music_fake else "ABSENT",
            "competition_use_status": "APPROVED", "source_url": "https://example.invalid",
            "license": "test", "approval_basis": "test fixture",
            "license_source": "https://example.invalid/license",
            "license_snapshot_sha256": "a" * 64, "reviewed_at": "2026-09-01",
        })
    return pd.DataFrame(rows)


def test_second_music_source_must_be_independent():
    existing = read("train.csv")
    source = str(existing.loc[existing.music_present.eq(1), "source"].iloc[0])
    music_pair = _approved_rows(source).query("voice_present == 0 and music_present == 1")
    candidate = pd.concat([music_pair.assign(
        content_group=f"music_{index}") for index in range(10)], ignore_index=True)
    with pytest.raises(RuntimeError, match="not independent"):
        validate_paired_music(candidate, existing)


def test_source_disjoint_metric_complete():
    candidate = _approved_rows("unused_source")
    result = validate_source_disjoint(candidate, _approved_rows("development_source"))
    assert result["status"] == "PASS"
    assert all(result["metric_complete"].values())


def test_source_disjoint_train_overlap_zero():
    candidate = _approved_rows("development_source")
    with pytest.raises(RuntimeError, match="overlap detected"):
        validate_source_disjoint(candidate, _approved_rows("development_source"))


def test_final_global_history_source_disjoint():
    candidate = _approved_rows("mlaad_tiny_matched")
    with pytest.raises(RuntimeError, match="overlaps global project history"):
        validate_final(candidate)


def test_final_global_history_generator_disjoint():
    historical = read("train.csv").query("file_fake == 1").iloc[0]
    candidate = _approved_rows("globally_unused_fixture_source")
    if int(historical.voice_fake) == 1:
        candidate.loc[candidate.voice_fake.eq(1), "voice_generator_family"] = historical.voice_generator_family
    else:
        candidate.loc[candidate.music_fake.eq(1), "music_generator_family"] = historical.music_generator_family
    with pytest.raises(RuntimeError, match="overlaps global project history"):
        validate_final(candidate)


def test_final_not_written_into_git_training_splits():
    assert not (SPLITS / "final_holdout_v13b.csv").exists()
    source = (ROOT / "scripts/complete_v13b_gates.py").read_text(encoding="utf-8")
    assert 'pathlib.Path(args.data_root).resolve() / "splits/final_holdout_v13b.csv"' in source


def test_end_to_end_runtime_prediction_parity():
    report = json.loads((ROOT / "experiments/v13b/end_to_end_runtime_benchmark.json").read_text())
    assert report["status"] == "PASS"
    assert report["prediction_parity"]
    assert report["prediction_max_abs_diff"] <= report["predefined_prediction_tolerance"]
