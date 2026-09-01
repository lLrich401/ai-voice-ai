import numpy as np
import pytest
from sklearn.metrics import roc_curve

import src.metrics as metric_module
from src.metrics import MetricUnavailableError, compute_dacon_metrics, compute_eer


def official_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(
        y_true,
        y_score,
        pos_label=1,
        drop_intermediate=False,
    )
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def test_eer_official_noninterpolated():
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    y_score = np.array([0.05, 0.91, 0.72, 0.54, 0.54, 0.22, 0.43, 0.81])
    assert compute_eer(y_true, y_score) == official_eer(y_true, y_score)


def test_eer_drop_intermediate_false(monkeypatch):
    observed = {}
    original = roc_curve

    def checked(*args, **kwargs):
        observed.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(metric_module, "roc_curve", checked)
    compute_eer([0, 1, 0, 1], [0.1, 0.9, 0.3, 0.7])
    assert observed["drop_intermediate"] is False
    assert observed["pos_label"] == 1


def test_component_eer_is_conditioned_on_presence():
    y_true = {
        "file_fake": np.array([0, 1, 0, 1, 0, 1]),
        "voice_fake": np.array([0, 1, 0, 0, 0, 1]),
        "music_fake": np.array([0, 0, 0, 1, 0, 1]),
        "voice_present": np.array([1, 1, 0, 0, 1, 1]),
        "music_present": np.array([0, 0, 1, 1, 1, 1]),
    }
    y_pred = {
        "file_fake": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
        # Absent-source rows deliberately have misleading extreme scores.
        "voice_fake": np.array([0.1, 0.9, 1.0, 1.0, 0.2, 0.8]),
        "music_fake": np.array([1.0, 1.0, 0.1, 0.9, 0.2, 0.8]),
        "voice_present": np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.6]),
        "music_present": np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.6]),
    }
    result = compute_dacon_metrics(y_true, y_pred)
    voice_mask = y_true["voice_present"] == 1
    music_mask = y_true["music_present"] == 1
    assert result["voice_eer"] == official_eer(
        y_true["voice_fake"][voice_mask], y_pred["voice_fake"][voice_mask]
    )
    assert result["music_eer"] == official_eer(
        y_true["music_fake"][music_mask], y_pred["music_fake"][music_mask]
    )


def test_nonfinite_prediction_fails():
    with pytest.raises(RuntimeError, match="non-finite"):
        compute_eer([0, 1], [0.1, np.nan])


def test_metric_schema_exact():
    truth = {key: np.array([0, 1]) for key in metric_module.TRUTH_COLUMNS}
    prediction = dict(truth)
    prediction["FILE_FAKE_PROB"] = prediction.pop("file_fake")
    with pytest.raises(KeyError, match="schema mismatch"):
        compute_dacon_metrics(truth, prediction)


def test_single_class_explicit():
    with pytest.raises(MetricUnavailableError, match="single-class"):
        compute_eer([1, 1], [0.2, 0.8])
