"""Strict DACON 236749 metric implementation.

Score = 0.9 * ADS + 0.1 * CPS. Component fake EER is conditioned on the
corresponding component being present. Production evaluation is fail-closed:
schema mismatches, non-finite values, unequal lengths, and single-class metrics
are never converted to an apparently valid 0.5 score.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


TRUTH_COLUMNS = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")
PREDICTION_COLUMNS = TRUTH_COLUMNS


class MetricUnavailableError(ValueError):
    """Raised when an official metric cannot be computed from supplied rows."""


def _validated_binary_inputs(y_true, y_score, *, allow_nonfinite: bool,
                             metric: str) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.int32).reshape(-1)
    score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if len(truth) != len(score):
        raise ValueError(f"{metric}: truth/prediction length mismatch {len(truth)} != {len(score)}")
    finite = np.isfinite(score)
    if not finite.all():
        indices = np.flatnonzero(~finite).tolist()
        if not allow_nonfinite:
            raise RuntimeError(f"{metric}: non-finite predictions at rows {indices[:10]}")
        truth, score = truth[finite], score[finite]
    if len(truth) == 0:
        raise MetricUnavailableError(f"{metric}: no finite rows")
    labels = np.unique(truth)
    if not set(labels.tolist()) <= {0, 1}:
        raise ValueError(f"{metric}: labels must be binary 0/1, got {labels.tolist()}")
    if len(labels) < 2:
        raise MetricUnavailableError(f"{metric}: single-class truth {labels.tolist()}")
    return truth, score


def compute_eer(y_true, y_score, *, allow_nonfinite: bool = False) -> float:
    truth, score = _validated_binary_inputs(
        y_true, y_score, allow_nonfinite=allow_nonfinite, metric="EER")
    fpr, tpr, _ = roc_curve(
        truth, score, pos_label=1, drop_intermediate=False)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def compute_auc(y_true, y_score, *, allow_nonfinite: bool = False) -> float:
    truth, score = _validated_binary_inputs(
        y_true, y_score, allow_nonfinite=allow_nonfinite, metric="AUC")
    return float(roc_auc_score(truth, score))


def _exact_columns(values: dict, expected: tuple[str, ...], role: str) -> dict[str, np.ndarray]:
    keys = set(values)
    missing = set(expected) - keys
    unexpected = keys - set(expected)
    if missing or unexpected:
        raise KeyError(
            f"official {role} schema mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}")
    return {name: np.asarray(values[name]) for name in expected}


def compute_dacon_metrics(y_true_dict, y_pred_dict, *,
                          allow_nonfinite: bool = False) -> dict[str, float]:
    """Compute the official metric from exact canonical column names only."""
    truth = _exact_columns(y_true_dict, TRUTH_COLUMNS, "truth")
    prediction = _exact_columns(y_pred_dict, PREDICTION_COLUMNS, "prediction")
    lengths = {name: len(np.asarray(value).reshape(-1)) for name, value in truth.items()}
    lengths.update({f"pred:{name}": len(np.asarray(value).reshape(-1))
                    for name, value in prediction.items()})
    if len(set(lengths.values())) != 1:
        raise ValueError(f"official metric columns have unequal lengths: {lengths}")

    file_eer = compute_eer(
        truth["file_fake"], prediction["file_fake"], allow_nonfinite=allow_nonfinite)
    voice_mask = np.asarray(truth["voice_present"]).astype(int) == 1
    music_mask = np.asarray(truth["music_present"]).astype(int) == 1
    voice_eer = compute_eer(
        np.asarray(truth["voice_fake"])[voice_mask],
        np.asarray(prediction["voice_fake"])[voice_mask], allow_nonfinite=allow_nonfinite)
    music_eer = compute_eer(
        np.asarray(truth["music_fake"])[music_mask],
        np.asarray(prediction["music_fake"])[music_mask], allow_nonfinite=allow_nonfinite)
    voice_auc = compute_auc(
        truth["voice_present"], prediction["voice_present"], allow_nonfinite=allow_nonfinite)
    music_auc = compute_auc(
        truth["music_present"], prediction["music_present"], allow_nonfinite=allow_nonfinite)
    ads = 1.0 - 0.5 * file_eer - 0.2 * voice_eer - 0.3 * music_eer
    cps = 0.5 * voice_auc + 0.5 * music_auc
    total = 0.9 * ads + 0.1 * cps
    return {
        "file_eer": file_eer, "voice_eer": voice_eer, "music_eer": music_eer,
        "voice_auc": voice_auc, "music_auc": music_auc,
        "ads": float(ads), "cps": float(cps), "score": float(total), "total": float(total),
    }
