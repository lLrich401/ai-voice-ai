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


def test_segment_and_feature_fusion_are_identical_and_not_presence_gated():
    v = np.array([[0.4, 0.7, 0.2, 0.1, 0.3], [0.6, 0.9, 0.2, 0.2, 0.3]])
    m = np.array([[0.3, 0.2, 0.6, 0.4, 0.8], [0.5, 0.2, 0.8, 0.4, 0.9]])
    p = {"voice_present": np.array([0.05, 0.1]), "music_present": np.array([0.8, 0.9])}
    weights = {"w_voice_file": 0.0, "w_music_file": 0.5, "w_prob_or": 0.5,
               "w_df_arena": 0.25, "w_df_component": 0.5, "w_panns_presence": 0.0}
    combined = script._combine_predictions(0.8, v, m, p, weights)
    direct = script.fuse_prediction_features(
        0.8, 0.8, 0.7, 0.5, 0.4, 0.15, 0.85, 0.075, 0.85, weights)
    assert np.allclose(combined, direct)

    p_low = dict(p)
    p_low["voice_present"] = np.zeros(2)
    low_presence = script._combine_predictions(0.8, v, m, p_low, weights)
    assert low_presence[1] == combined[1]  # voice-fake ranking is not gated
    assert all(script.OUTPUT_EPS <= x <= 1.0 - script.OUTPUT_EPS for x in combined)


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
