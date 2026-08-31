"""Head-selective specialist ensembling for non-final validation and inference."""

from __future__ import annotations

import pathlib

import numpy as np

from .metrics import compute_dacon_metrics


EPSILON = 1e-6


def assert_final_holdout_forbidden(*paths) -> None:
    """Fail closed before opening any protected final-holdout artifact."""
    for value in paths:
        normalized = str(pathlib.Path(value)).replace("\\", "/").lower()
        if "final_holdout" in normalized:
            raise RuntimeError(f"final holdout access is forbidden: {value}")


def validate_ensemble_cache_metadata(actual: dict, expected: dict) -> None:
    stale = {key: (actual.get(key), value) for key, value in expected.items()
             if actual.get(key) != value}
    if stale:
        raise RuntimeError(f"stale ensemble feature cache: {stale}")


def _rank_probability(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    # Average tied ranks without depending on scipy/pandas.
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks / (len(values) + 1.0)


def blend_probabilities(base, candidate, alpha: float, method: str = "probability") -> np.ndarray:
    base = np.clip(np.asarray(base, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    candidate = np.clip(np.asarray(candidate, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    if base.shape != candidate.shape:
        raise ValueError("ensemble inputs must have identical shape")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("ensemble alpha must be in [0, 1]")
    method = str(method).lower()
    # The disabled/default configuration must reproduce v7 exactly for every
    # advertised method, including rank ensembling.
    if alpha == 0.0:
        return base.copy()
    if method == "probability":
        result = (1.0 - alpha) * base + alpha * candidate
    elif method == "logit":
        base_logit = np.log(base / (1.0 - base))
        candidate_logit = np.log(candidate / (1.0 - candidate))
        mixed = (1.0 - alpha) * base_logit + alpha * candidate_logit
        result = 1.0 / (1.0 + np.exp(-np.clip(mixed, -40.0, 40.0)))
    elif method == "rank":
        result = ((1.0 - alpha) * _rank_probability(base)
                  + alpha * _rank_probability(candidate))
    elif method == "max":
        result = np.maximum(base, candidate)
    else:
        raise ValueError(f"unsupported ensemble method: {method}")
    return np.clip(result, EPSILON, 1.0 - EPSILON)


def predict_head_selective_ensemble(
    frame,
    weights: dict,
    *,
    voice_fake=None,
    music_fake=None,
    voice_affects_file: bool = False,
    music_affects_file: bool = False,
    use_candidate_voice_file_head: bool = False,
    use_candidate_music_file_head: bool = False,
) -> np.ndarray:
    """Vectorized canonical fusion with independently selectable output heads.

    Component-only ensembling changes VOICE_FAKE/MUSIC_FAKE ranking while the
    FILE prediction remains byte-for-byte equivalent to v7. Candidate FILE
    heads are read only when their explicit switches are true.
    """
    values = lambda name: np.asarray(frame[name], dtype=np.float64)
    df_score = values("df_primary")
    base_voice_raw, base_music_raw = values("vf"), values("mf")
    voice_raw = base_voice_raw if voice_fake is None else np.asarray(voice_fake, dtype=np.float64)
    music_raw = base_music_raw if music_fake is None else np.asarray(music_fake, dtype=np.float64)
    legacy = float(weights.get("w_df_component", 0.0))
    voice_df = float(weights.get("w_df_voice_component", legacy))
    music_df = float(weights.get("w_df_music_component", legacy))
    base_voice = voice_df * df_score + (1.0 - voice_df) * base_voice_raw
    base_music = music_df * df_score + (1.0 - music_df) * base_music_raw
    output_voice = voice_df * df_score + (1.0 - voice_df) * voice_raw
    output_music = music_df * df_score + (1.0 - music_df) * music_raw
    file_voice = output_voice if voice_affects_file else base_voice
    file_music = output_music if music_affects_file else base_music
    vfile = (values("v9_vfile") if use_candidate_voice_file_head else values("vfile"))
    mfile = (values("v9_mfile") if use_candidate_music_file_head else values("mfile"))
    panns_weight = float(weights.get("w_panns_presence", 0.6))
    voice_present = panns_weight * values("vp_panns") + (1.0 - panns_weight) * values("vp_model")
    music_present = panns_weight * values("mp_panns") + (1.0 - panns_weight) * values("mp_model")
    probability_or = 1.0 - (1.0 - file_voice) * (1.0 - file_music)
    wv = float(weights.get("w_voice_file", 0.5))
    wm = float(weights.get("w_music_file", 0.3))
    wo = float(weights.get("w_prob_or", 0.2))
    mode = str(weights.get("file_fusion_mode", "legacy"))
    if mode == "legacy":
        detector = wv * vfile + wm * mfile + wo * probability_or
    else:
        voice_risk = voice_present * file_voice
        music_risk = music_present * file_music
        component_or = 1.0 - (1.0 - voice_risk) * (1.0 - music_risk)
        if mode == "presence_component_or":
            detector = component_or
        elif mode == "presence_weighted":
            detector = (wv * voice_present * vfile
                        + wm * music_present * mfile + wo * component_or)
        else:
            raise ValueError(f"unsupported file_fusion_mode={mode}")
    file_fake = float(weights.get("w_df_arena", 0.5)) * df_score + (
        1.0 - float(weights.get("w_df_arena", 0.5))) * detector
    return np.clip(np.column_stack((
        file_fake, output_voice, output_music, voice_present, music_present,
    )), EPSILON, 1.0 - EPSILON)


def score_head_selective_ensemble(frame, weights: dict, **kwargs) -> dict:
    predicted = predict_head_selective_ensemble(frame, weights, **kwargs)
    heads = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")
    truth = {head: np.asarray(frame[f"y_{head}"]) for head in heads}
    scores = {head: predicted[:, index] for index, head in enumerate(heads)}
    return compute_dacon_metrics(truth, scores)
