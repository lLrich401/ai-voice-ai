import json
import pathlib

import numpy as np
import pandas as pd
import pytest
import torch

from src.distillation import (
    PairedGroupBatchSampler, component_teacher_loss, freeze_teacher,
    source_balanced_weights, teacher_is_frozen, v7_retention_loss,
)
from src.ensemble import assert_final_holdout_forbidden
from src.models.beats_backbone import MusicMultitask


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_final_holdout_forbidden_v12():
    with pytest.raises(RuntimeError, match="final holdout access is forbidden"):
        assert_final_holdout_forbidden("experiments/v12/final_holdout.csv")


def test_distillation_teacher_frozen():
    teacher = freeze_teacher(torch.nn.Linear(3, 1))
    assert teacher_is_frozen(teacher)


def test_v7_retention_loss_student_only_gradient():
    student = {name: torch.tensor([0.2, -0.3], requires_grad=True)
               for name in ("file_fake", "voice_fake", "voice_present")}
    teacher = {name: torch.tensor([0.4, -0.1], requires_grad=False)
               for name in student}
    labels = torch.tensor([[1, 1, 0, 1, 0], [0, 0, 0, 1, 0]], dtype=torch.float32)
    loss = v7_retention_loss(student, teacher, labels, "voice")
    assert torch.isfinite(loss)
    loss.backward()
    assert all(value.grad is not None for value in student.values())
    assert all(value.grad is None for value in teacher.values())


@pytest.mark.parametrize("task,component", [("music", "music_fake"), ("voice", "voice_fake")])
def test_candidate_teacher_file_head_not_used(task, component):
    student = {component: torch.tensor([0.1, -0.2], requires_grad=True)}
    teacher = {component: torch.tensor([0.3, -0.4])}
    labels = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.float32)
    loss = component_teacher_loss(student, teacher, labels, task)
    assert torch.isfinite(loss)


def test_source_balanced_sampler():
    frame = pd.DataFrame({
        "voice_present": [1, 1, 1, 0], "voice_fake": [0, 1, 1, 0],
        "source": ["a", "b", "b", "c"], "generator": ["r", "f", "f", "none"],
    })
    weights = source_balanced_weights(frame, "voice")
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] == pytest.approx(0.4)
    assert weights[1] + weights[2] == pytest.approx(0.4)
    assert weights[3] == pytest.approx(0.2)


def test_paired_group_sampler_no_cross_split():
    frame = pd.DataFrame({
        "data_role": ["train_v12", "train_v12"],
        "voice_present": [1, 1], "voice_fake": [0, 1],
        "source": ["a", "a"], "generator": ["real", "fake"],
        "split_group_id": ["g", "g"],
    })
    sampler = PairedGroupBatchSampler(frame, "voice", 2)
    assert len(next(iter(sampler))) == 2
    bad = frame.copy(); bad.loc[1, "data_role"] = "val_b"
    with pytest.raises(ValueError, match="TRAIN rows only"):
        PairedGroupBatchSampler(bad, "voice", 2)


def test_cal_v12_no_train_or_val_overlap():
    report = json.loads((ROOT / "experiments/v12/cal_v12_report.json").read_text())
    for role in ("train", "validation", "expanded_unseen"):
        assert all(value == 0 for value in report["overlap"][role].values())


def test_v12_cache_checkpoint_and_split_sha():
    meta = json.loads((ROOT / "experiments/v12/cache/v7_canonical/cal_old.csv.meta.json").read_text())
    assert len(meta["checkpoint_sha256"]["voice"]) == 64
    assert len(meta["split_sha256"]) == 64
    assert meta["final_holdout"] == "NOT RUN"


def test_student_prediction_not_batch_dependent():
    model = MusicMultitask(base_channels=4).eval()
    first = torch.zeros(1, 64000)
    second = torch.ones(1, 64000) * 0.01
    with torch.no_grad():
        alone = torch.sigmoid(model(first)["music_fake"])[0]
        batched = torch.sigmoid(model(torch.cat([first, second]))["music_fake"])[0]
    # oneDNN/XPU can select different float32 convolution kernels for N=1 and
    # N=2.  The observed numerical drift is small (<2e-4) and no module uses
    # batch statistics in eval mode, so this threshold detects semantic
    # cross-file normalization without pretending kernels are bit-identical.
    assert all(not module.training for module in model.modules())
    assert float(alone) == pytest.approx(float(batched), abs=5e-4)
