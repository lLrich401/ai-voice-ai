import threading

import numpy as np

import script


class IdentitySeparator:
    use_demucs = False

    @staticmethod
    def separate(wave, sr=16_000):
        return wave, wave


def _fake_outputs(groups, detector_name):
    bounds = []
    count = 0
    for segments in groups:
        left = count
        count += len(segments)
        bounds.append((left, count))
    if detector_name == "panns":
        values = {
            "voice_present": np.full(count, 0.6, dtype=np.float32),
            "music_present": np.full(count, 0.7, dtype=np.float32),
        }
    else:
        values = {
            key: np.full(count, 0.5, dtype=np.float32)
            for key in ("file_fake", "voice_fake", "music_fake",
                        "voice_present", "music_present")
        }
    return values, bounds


def test_cuda_always_on_df_overlaps_independent_torch_models(monkeypatch):
    df_started = threading.Event()
    torch_started = threading.Event()

    def fake_df(*args, **kwargs):
        df_started.set()
        assert torch_started.wait(timeout=2.0)
        return [0.7], [{
            "primary": 0.7, "second": None, "primary_start": 0,
            "second_start": None, "used_second": False,
        }]

    def fake_torch(model, groups, device, **kwargs):
        assert df_started.wait(timeout=2.0)
        torch_started.set()
        return _fake_outputs(groups, kwargs["detector_name"])

    monkeypatch.setattr(script, "df_arena_predict_batch", fake_df)
    monkeypatch.setattr(script, "_run_torch_segments", fake_torch)
    features = script.infer_wave_features_batch(
        object(), object(), object(), object(),
        [np.ones(64_000, dtype=np.float32)], "cuda",
        df_config={"enabled": False, "gpu_overlap_enabled": True}, aggregation_config={},
        separator=IdentitySeparator())
    assert len(features) == 1
    assert features[0]["df"] == 0.7
    assert features[0]["df_used"] is True


def test_cuda_overlap_is_disabled_without_explicit_measured_opt_in(monkeypatch):
    torch_finished = threading.Event()

    def fake_df(*args, **kwargs):
        assert torch_finished.is_set()
        return [0.7], [{
            "primary": 0.7, "second": None, "primary_start": 0,
            "second_start": None, "used_second": False,
        }]

    def fake_torch(model, groups, device, **kwargs):
        torch_finished.set()
        return _fake_outputs(groups, kwargs["detector_name"])

    monkeypatch.setattr(script, "df_arena_predict_batch", fake_df)
    monkeypatch.setattr(script, "_run_torch_segments", fake_torch)
    features = script.infer_wave_features_batch(
        object(), object(), object(), object(),
        [np.ones(64_000, dtype=np.float32)], "cuda",
        df_config={"enabled": False}, aggregation_config={},
        separator=IdentitySeparator())
    assert features[0]["df"] == 0.7


def test_cpu_path_remains_sequential(monkeypatch):
    torch_finished = threading.Event()

    def fake_df(*args, **kwargs):
        assert torch_finished.is_set()
        return [0.7], [{
            "primary": 0.7, "second": None, "primary_start": 0,
            "second_start": None, "used_second": False,
        }]

    def fake_torch(model, groups, device, **kwargs):
        torch_finished.set()
        return _fake_outputs(groups, kwargs["detector_name"])

    monkeypatch.setattr(script, "df_arena_predict_batch", fake_df)
    monkeypatch.setattr(script, "_run_torch_segments", fake_torch)
    features = script.infer_wave_features_batch(
        object(), object(), object(), object(),
        [np.ones(64_000, dtype=np.float32)], "cpu",
        df_config={"enabled": False}, aggregation_config={},
        separator=IdentitySeparator())
    assert features[0]["df"] == 0.7
