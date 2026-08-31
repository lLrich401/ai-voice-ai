import numpy as np
import os
import pytest
import soundfile as sf
import json

import script


def test_adaptive_segment_counts():
    assert len(script.select_aux_segments(np.ones(7 * 16000, np.float32))) == 1
    assert len(script.select_aux_segments(np.ones(20 * 16000, np.float32))) == 2
    assert len(script.select_aux_segments(np.ones(40 * 16000, np.float32))) == 3


def test_task_specific_aggregation_and_df_gate_paths():
    values=np.array([0.1,0.4,0.9])
    assert script.aggregate_head_predictions(values,"voice_fake") == pytest.approx(0.65)
    assert script.aggregate_head_predictions(values,"music_fake",{"music_fake_aggregation":"max"}) == pytest.approx(0.9)
    assert script.aggregate_head_predictions(values,"voice_present") == pytest.approx(values.mean())
    outputs={"voice_present":np.array([0.2,0.4,0.8,0.9])}
    bounds=[(0,2),(2,4)]
    assert script.select_df_indices(outputs,bounds,None)==[0,1]
    assert script.select_df_indices(outputs,bounds,0.5)==[1]


def test_voice_aggregation_consistency_and_max_mean_blends():
    values = np.array([0.1, 0.4, 0.9])
    assert script.aggregate_predictions(values, "mean") == pytest.approx(values.mean())
    assert script.aggregate_predictions(values, "median") == pytest.approx(0.4)
    assert script.aggregate_predictions(values, "trimmed_mean") == pytest.approx(0.4)
    for alpha in (0.25, 0.5, 0.75):
        assert script.aggregate_predictions(values, f"max_mean_{alpha}") == pytest.approx(
            alpha * values.max() + (1.0 - alpha) * values.mean())


def test_auxiliary_segment_policies_are_deterministic_and_bounded():
    wave = np.concatenate([
        np.full(8 * 16000, 0.01, np.float32),
        np.full(8 * 16000, 0.5, np.float32),
        np.full(24 * 16000, 0.1, np.float32),
    ])
    for policy in ("high_energy", "uniform", "centered", "energy_diverse"):
        first = script.select_aux_segments(wave, policy=policy)
        second = script.select_aux_segments(wave, policy=policy)
        assert 1 <= len(first) <= 3
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
    with pytest.raises(ValueError, match="unsupported auxiliary segment policy"):
        script.select_aux_segments(wave, policy="unknown")


def test_segment_and_feature_fusion_are_identical_and_not_presence_gated():
    v = np.array([[0.4, 0.7, 0.2, 0.1, 0.3], [0.6, 0.9, 0.2, 0.2, 0.3]])
    m = np.array([[0.3, 0.2, 0.6, 0.4, 0.8], [0.5, 0.2, 0.8, 0.4, 0.9]])
    p = {"voice_present": np.array([0.05, 0.1]), "music_present": np.array([0.8, 0.9])}
    weights = {"w_voice_file": 0.0, "w_music_file": 0.5, "w_prob_or": 0.5,
               "w_df_arena": 0.25, "w_df_voice_component": 0.5,
               "w_df_music_component": 0.5, "w_panns_presence": 0.0}
    combined = script._combine_predictions(0.8, v, m, p, weights)
    direct = script.fuse_prediction_features(
        0.8, 0.8, 0.7, 0.5, 0.4, 0.15, 0.85, 0.075, 0.85, weights)
    assert np.allclose(combined, direct)

    p_low = dict(p)
    p_low["voice_present"] = np.zeros(2)
    low_presence = script._combine_predictions(0.8, v, m, p_low, weights)
    assert low_presence[1] == combined[1]  # voice-fake ranking is not gated
    assert all(script.OUTPUT_EPS <= x <= 1.0 - script.OUTPUT_EPS for x in combined)


def test_df_voice_and_music_component_weights_are_independent():
    common = dict(file_fake_df=0.9, voice_fake_model=0.1, music_fake_model=0.2,
                  file_voice=0.3, file_music=0.4, voice_present_model=0.5,
                  music_present_model=0.6, voice_present_panns=0.5,
                  music_present_panns=0.6)
    voice_only = script.fuse_prediction_features(
        **common, fusion_weights={"w_df_voice_component": 1.0,
                                  "w_df_music_component": 0.0})
    music_only = script.fuse_prediction_features(
        **common, fusion_weights={"w_df_voice_component": 0.0,
                                  "w_df_music_component": 1.0})
    assert voice_only[1] == pytest.approx(0.9)
    assert voice_only[2] == pytest.approx(0.2)
    assert music_only[1] == pytest.approx(0.1)
    assert music_only[2] == pytest.approx(0.9)


def test_skipped_df_is_neutral_for_component_fusion():
    feature = {"df": 0.99, "df_used": False, "vf": 0.1, "mf": 0.2,
               "vfile": 0.3, "mfile": 0.4, "vp_model": 0.5,
               "mp_model": 0.6, "vp_panns": 0.5, "mp_panns": 0.6}
    weights = {"w_voice_file": 1.0, "w_music_file": 0.0, "w_prob_or": 0.0,
               "w_df_arena": 0.0, "w_df_voice_component": 1.0,
               "w_df_music_component": 1.0, "w_panns_presence": 0.0}
    result = script.fuse_feature_record(feature, weights)
    assert result[0] == pytest.approx(0.3)
    assert result[1] == pytest.approx(0.1)
    assert result[2] == pytest.approx(0.2)


def test_presence_aware_file_risk_does_not_gate_component_outputs():
    common = dict(file_fake_df=0.5, voice_fake_model=0.9, music_fake_model=0.8,
                  file_voice=0.7, file_music=0.6, voice_present_model=0.0,
                  music_present_model=1.0, voice_present_panns=0.0,
                  music_present_panns=1.0)
    legacy = script.fuse_prediction_features(
        **common, fusion_weights={"file_fusion_mode":"legacy", "w_voice_file":0.5,
                                  "w_music_file":0.0, "w_prob_or":0.5,
                                  "w_df_arena":0.0, "w_panns_presence":0.5})
    aware = script.fuse_prediction_features(
        **common, fusion_weights={"file_fusion_mode":"presence_weighted",
                                  "w_voice_file":0.5, "w_music_file":0.0,
                                  "w_prob_or":0.5, "w_df_arena":0.0,
                                  "w_panns_presence":0.5})
    assert aware[1] == pytest.approx(legacy[1]) == pytest.approx(0.9)
    assert aware[2] == pytest.approx(legacy[2]) == pytest.approx(0.8)
    assert aware[0] < legacy[0]


@pytest.mark.parametrize("absent", ["voice", "music"])
def test_absent_component_cannot_affect_presence_aware_file_risk(absent):
    values = dict(file_fake_df=0.5, voice_fake_model=0.2, music_fake_model=0.3,
                  file_voice=0.4, file_music=0.5, voice_present_model=1.0,
                  music_present_model=1.0, voice_present_panns=1.0,
                  music_present_panns=1.0)
    if absent == "voice":
        values.update(voice_present_model=0.0, voice_present_panns=0.0)
        varied = ("voice_fake_model", "file_voice")
    else:
        values.update(music_present_model=0.0, music_present_panns=0.0)
        varied = ("music_fake_model", "file_music")
    weights = {"file_fusion_mode":"presence_weighted", "w_voice_file":0.25,
               "w_music_file":0.25, "w_prob_or":0.5, "w_df_arena":0.0,
               "w_panns_presence":0.5}
    first = script.fuse_prediction_features(**values, fusion_weights=weights)[0]
    values[varied[0]] = 0.99
    values[varied[1]] = 0.99
    second = script.fuse_prediction_features(**values, fusion_weights=weights)[0]
    assert second == pytest.approx(first)


def test_adaptive_df_second_crop_is_distant_and_conditioned():
    wave = np.zeros(20 * 16000, dtype=np.float32)
    wave[0:64600] = 0.5
    wave[-64600:] = 0.4
    _, second, primary_start, second_start = script._df_arena_crop_candidates(wave)
    assert second is not None
    assert abs(second_start - primary_start) >= max(script.DF_INPUT_SAMPLES // 2, 2 * 16000)
    assert script.should_use_adaptive_df_second_crop(12.0, 0.5, 0.25, 0.75)
    assert not script.should_use_adaptive_df_second_crop(11.9, 0.5, 0.25, 0.75)
    assert not script.should_use_adaptive_df_second_crop(20.0, 0.9, 0.25, 0.75)
    assert script.should_use_adaptive_df_second_crop(
        10.0,0.9,0.2,0.8,10.0,voice_fake_probability=0.5,
        music_fake_probability=0.9,trigger_mode="any_uncertain_disagreement")
    assert abs(second_start-primary_start)>=3*16000


def test_all_five_probabilities_are_finite_and_clipped():
    result = script.fuse_prediction_features(
        2.0, -1.0, 3.0, 2.0, -1.0, np.nan, 3.0, 0.5, 0.5,
        {"w_df_voice_component": 0.0, "w_df_music_component": 0.0,
         "w_panns_presence": 1.0})
    assert np.isfinite(result).all()
    assert all(script.OUTPUT_EPS <= value <= 1.0 - script.OUTPUT_EPS for value in result)


def test_exact_sample_mapping_and_output_shape():
    results = [["B", 0.1, 0.2, 0.3, 0.4, 0.5], ["A", 0.5, 0.4, 0.3, 0.2, 0.1]]
    ordered = script.order_results_by_sample(results, ["A.wav", "B"])
    assert [row[0] for row in ordered] == ["A.wav", "B"]
    assert all(len(row) == 6 for row in ordered)
    with pytest.raises(FileNotFoundError):
        script.order_results_by_sample(results, ["A", "missing"])


def test_wav_flac_and_optional_mp3_loading(tmp_path):
    wave = np.linspace(-0.1, 0.1, 1600, dtype=np.float32)
    for extension in ("wav", "flac"):
        path = tmp_path / f"audio.{extension}"
        sf.write(path, wave, 16000)
        loaded, sr = script.load_audio(path)
        assert sr == 16000 and loaded.shape == wave.shape and np.isfinite(loaded).all()
    mp3 = tmp_path / "audio.mp3"
    try:
        sf.write(mp3, wave, 16000, format="MP3")
    except Exception:
        pytest.skip("local libsndfile has no MP3 encoder")
    loaded, sr = script.load_audio(mp3)
    assert sr == 16000 and loaded.shape == wave.shape


def test_offline_flags_and_missing_models_fail_fast(tmp_path, monkeypatch):
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        script.verify_mandatory_models()


def test_stale_fusion_weights_fail(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "fusion_weights.json").write_text(json.dumps({
        "pipeline_version": "old", "voice_checkpoint_sha256": "old",
        "music_checkpoint_sha256": "old",
    }), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(script, "_sha256_file", lambda path: "current")
    with pytest.raises(RuntimeError, match="Stale fusion"):
        script.load_fusion_weights()
