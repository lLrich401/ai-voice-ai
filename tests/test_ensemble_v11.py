import numpy as np
import pandas as pd
import pytest

from src.ensemble import (
    assert_final_holdout_forbidden, blend_probabilities,
    predict_head_selective_ensemble, validate_ensemble_cache_metadata,
)


def _frame():
    return pd.DataFrame({
        "df_primary": [0.2, 0.8], "vf": [0.1, 0.9], "mf": [0.3, 0.7],
        "vfile": [0.2, 0.8], "mfile": [0.4, 0.6],
        "vp_model": [0.6, 0.7], "mp_model": [0.5, 0.8],
        "vp_panns": [0.7, 0.8], "mp_panns": [0.6, 0.9],
        "v9_vfile": [0.99, 0.01], "v9_mfile": [0.98, 0.02],
    })


WEIGHTS = {
    "w_df_voice_component": 0.3, "w_df_music_component": 0.0,
    "w_panns_presence": 0.75, "w_voice_file": 0.0,
    "w_music_file": 0.5, "w_prob_or": 0.5, "w_df_arena": 0.25,
    "file_fusion_mode": "legacy",
}


@pytest.mark.parametrize("method", ["probability", "logit", "rank"])
def test_music_ensemble_alpha_zero_matches_v7(method):
    base = np.array([0.1, 0.4, 0.8])
    candidate = np.array([0.9, 0.2, 0.3])
    assert np.array_equal(blend_probabilities(base, candidate, 0.0, method), base)


@pytest.mark.parametrize("method", ["probability", "logit"])
def test_voice_ensemble_alpha_zero_matches_v7(method):
    base = np.array([0.15, 0.75])
    assert np.allclose(blend_probabilities(base, 1.0 - base, 0.0, method), base)


def test_logit_ensemble_finite():
    result = blend_probabilities([0.0, 1.0], [1.0, 0.0], 0.5, "logit")
    assert np.isfinite(result).all()
    assert ((result > 0) & (result < 1)).all()


@pytest.mark.parametrize("method", ["probability", "logit", "max"])
def test_independent_file_inference(method):
    base = np.array([0.2, 0.8, 0.35])
    candidate = np.array([0.7, 0.1, 0.65])
    batched = blend_probabilities(base, candidate, 0.3, method)
    for index in range(len(base)):
        single = blend_probabilities(
            base[index:index + 1], candidate[index:index + 1], 0.3, method)
        assert batched[index] == pytest.approx(single[0])


def test_candidate_file_head_not_used_when_disabled():
    frame = _frame()
    first = predict_head_selective_ensemble(frame, WEIGHTS)
    changed = frame.copy()
    changed[["v9_vfile", "v9_mfile"]] = [[0.01, 0.99], [0.02, 0.98]]
    second = predict_head_selective_ensemble(changed, WEIGHTS)
    assert np.array_equal(first, second)


def test_component_only_ensemble_keeps_file_prediction():
    frame = _frame()
    base = predict_head_selective_ensemble(frame, WEIGHTS)
    candidate = predict_head_selective_ensemble(
        frame, WEIGHTS, voice_fake=1.0 - frame.vf, music_fake=1.0 - frame.mf,
        voice_affects_file=False, music_affects_file=False,
    )
    assert np.array_equal(base[:, 0], candidate[:, 0])
    assert not np.array_equal(base[:, 1:3], candidate[:, 1:3])


def test_cache_checkpoint_sha_guard():
    expected = {"checkpoint_sha256": "good", "split_sha256": "same"}
    validate_ensemble_cache_metadata(dict(expected), expected)
    with pytest.raises(RuntimeError, match="stale ensemble feature cache"):
        validate_ensemble_cache_metadata(
            {"checkpoint_sha256": "bad", "split_sha256": "same"}, expected)


def test_final_holdout_guard():
    assert_final_holdout_forbidden("data/splits/val_a.csv")
    with pytest.raises(RuntimeError, match="final holdout access is forbidden"):
        assert_final_holdout_forbidden("data/splits/final_holdout.csv")
