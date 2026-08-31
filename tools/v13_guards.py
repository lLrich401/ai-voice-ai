"""Fail-closed guards for V13 development and training tooling."""

from __future__ import annotations

import os
import pathlib


def assert_final_holdout_v13_forbidden(*paths: object) -> None:
    """Reject any V13 development access to the sealed final holdout."""
    if os.getenv("V13_FINAL_LOCKED", "1") != "1":
        raise RuntimeError("V13_FINAL_LOCKED must remain 1 during development")
    for value in paths:
        text = str(pathlib.Path(value)).replace("\\", "/").lower()
        if "final_holdout_v13" in text or "final-v13" in text:
            raise RuntimeError(f"V13 final holdout is sealed: {value}")


def assert_no_hidden_test_training_path(*paths: object) -> None:
    """Reject test/hidden/evaluation paths as training input."""
    forbidden = ("/data/test/", "/hidden_test/", "/private_test/", "/evaluation_test/")
    for value in paths:
        text = "/" + str(pathlib.Path(value)).replace("\\", "/").lower().strip("/") + "/"
        if any(token in text for token in forbidden):
            raise RuntimeError(f"test data cannot be used for training: {value}")
