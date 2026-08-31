import pandas as pd
import pytest
import numpy as np
import soundfile as sf

from scripts.audit_public_audio_v9 import inspect
from scripts.prepare_public_diversity_v9 import audit_manifests
from scripts.prepare_echoes_paired_v9 import allowed_license, normalize, select_rows
from scripts.prepare_real_music_v9 import _allowed_fma_license


def _voice_row(role, group, fake, generator):
    return {
        "data_role": role, "split_group_id": group,
        "voice_fake": fake, "generator": generator,
    }


def test_public_voice_pairs_are_source_matched_and_role_disjoint():
    mlaad = pd.DataFrame([
        _voice_row("train", "a", 0, "original"),
        _voice_row("train", "a", 1, "train_generator"),
        _voice_row("val_b", "b", 0, "original"),
        _voice_row("val_b", "b", 1, "unseen_generator"),
    ])
    sonics = pd.DataFrame([{
        "generator": "SONICS::udio-120s", "voice_present": 1,
        "music_present": 1, "voice_fake": 1, "music_fake": 1,
    }])
    report = audit_manifests(mlaad, sonics)
    assert report["mlaad_cross_role_group_overlap"] == 0
    assert report["mlaad_all_groups_paired_real_fake"] is True


def test_public_voice_audit_rejects_content_leakage():
    mlaad = pd.DataFrame([
        _voice_row("train", "same", 0, "original"),
        _voice_row("train", "same", 1, "train_generator"),
        _voice_row("val_b", "same", 0, "original"),
        _voice_row("val_b", "same", 1, "unseen_generator"),
    ])
    sonics = pd.DataFrame([{
        "generator": "SONICS::udio-120s", "voice_present": 1,
        "music_present": 1, "voice_fake": 1, "music_fake": 1,
    }])
    with pytest.raises(RuntimeError, match="leakage"):
        audit_manifests(mlaad, sonics)


def test_fma_filter_excludes_unknown_and_no_derivatives():
    assert _allowed_fma_license("https://creativecommons.org/licenses/by/4.0/")
    assert _allowed_fma_license("https://creativecommons.org/licenses/by-nc-sa/3.0/")
    assert not _allowed_fma_license("https://creativecommons.org/licenses/by-nd/4.0/")
    assert not _allowed_fma_license("")


def test_echoes_license_filter_excludes_unknown_and_no_derivatives():
    assert allowed_license("https://creativecommons.org/licenses/by-sa/4.0/")
    assert allowed_license("https://creativecommons.org/licenses/by-nc/3.0/")
    assert not allowed_license("https://creativecommons.org/licenses/by-nd/4.0/")
    assert not allowed_license("nan")


def test_echoes_pair_selection_keeps_unseen_content_out_of_train():
    rows = []
    fma_index = {}
    for generator in ("diffrhythm", "songgen", "train_gen"):
        for index in range(2):
            original = f"{generator} title {index} - artist"
            rows.append({
                "generator": generator, "original_audio": original,
                "type": "generated", "path_in_dataset": f"{generator}/{index}.mp3",
            })
            fma_index[normalize(original)] = [{
                "track_id": len(fma_index) + 1,
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
            }]
    selected = select_rows(pd.DataFrame(rows), fma_index, per_generator=1)
    train_keys = {
        normalize(row["original_audio"]) for row in selected if row["data_role"] == "train"
    }
    val_keys = {
        normalize(row["original_audio"]) for row in selected if row["data_role"] != "train"
    }
    assert train_keys.isdisjoint(val_keys)
    assert {row["generator"] for row in selected if row["data_role"] != "train"} == {
        "diffrhythm", "songgen",
    }


def test_public_audio_quality_inspection_decodes_and_measures(tmp_path):
    path = tmp_path / "clean.wav"
    wave = (0.1 * np.sin(np.linspace(0, 200, 32000))).astype(np.float32)
    sf.write(path, wave, 16000)
    result = inspect(path)
    assert result["sample_rate"] == 16000
    assert result["channels"] == 1
    assert result["duration_seconds"] == pytest.approx(2.0)
    assert result["finite"] is True
    assert result["clipping_fraction"] == 0.0
