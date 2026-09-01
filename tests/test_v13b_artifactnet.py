import inspect

import numpy as np

from scripts.evaluate_v13b_artifactnet import (
    ARTIFACTNET_SAMPLES,
    artifactnet_predictions,
    to_artifactnet_input,
)


def test_artifactnet_frontend_has_exact_shape_and_finite_silence_guard():
    output, adjusted = to_artifactnet_input(np.zeros(1_000, dtype=np.float32))
    assert output.shape == (ARTIFACTNET_SAMPLES,)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert np.count_nonzero(output) == 1
    assert adjusted is False


def test_artifactnet_frontend_applies_linear_gain_without_clipping():
    source = np.linspace(-1.0, 1.0, ARTIFACTNET_SAMPLES, dtype=np.float32)
    output, adjusted = to_artifactnet_input(source)
    assert adjusted is True
    assert np.isclose(np.max(np.abs(output)), 0.25)
    nonzero = source != 0
    ratios = output[nonzero] / source[nonzero]
    assert np.allclose(ratios, 0.25, rtol=1e-6, atol=1e-7)


def test_artifactnet_frontend_preserves_safe_level_audio():
    source = np.linspace(-0.1, 0.1, ARTIFACTNET_SAMPLES, dtype=np.float32)
    output, adjusted = to_artifactnet_input(source)
    assert adjusted is False
    assert np.array_equal(output, source)


def test_artifactnet_evaluation_fails_closed_on_nonfinite_by_default():
    parameter = inspect.signature(artifactnet_predictions).parameters[
        "diagnostic_skip_nonfinite"]
    assert parameter.default is False
