#!/usr/bin/env python3
"""Refresh only dynamic V13B branch/commit metadata from the local Git state."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEVELOPMENT_BASE_COMMIT = "e8434b9c368ee5de3368d8b0b04559cf19c3ffaa"
JSON_TARGETS = (
    "experiments/latest_results.json",
    "experiments/v13b/stage_status.json",
    "experiments/v13b/correctness_report.json",
)
REPORT = ROOT / "experiments/v13b/final_report.md"


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def expected() -> dict[str, str]:
    return {
        "branch": git_value("branch", "--show-current"),
        "development_base_commit": DEVELOPMENT_BASE_COMMIT,
        "current_git_commit": git_value("rev-parse", "HEAD"),
    }


def update_json(path: pathlib.Path, values: dict[str, str], *, write: bool) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for key, value in values.items():
        if payload.get(key) != value:
            payload[key] = value
            changed = True
    if write and changed:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return changed


def update_report(values: dict[str, str], *, write: bool) -> bool:
    text = REPORT.read_text(encoding="utf-8")
    replacements = {
        r"(?m)^- Branch: .*$": f"- Branch: `{values['branch']}`",
        r"(?m)^- development_base_commit: .*$": (
            f"- development_base_commit: `{values['development_base_commit']}`"),
        r"(?m)^- current_git_commit: .*$": (
            f"- current_git_commit: `{values['current_git_commit']}`"),
    }
    updated = text
    for pattern, replacement in replacements.items():
        updated, count = re.subn(pattern, replacement, updated, count=1)
        if count != 1:
            raise RuntimeError(f"source-of-truth report field missing: {pattern}")
    changed = updated != text
    if write and changed:
        REPORT.write_text(updated, encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail rather than write stale metadata")
    args = parser.parse_args()
    values = expected()
    stale = []
    for relative in JSON_TARGETS:
        path = ROOT / relative
        if update_json(path, values, write=not args.check):
            stale.append(relative)
    if update_report(values, write=not args.check):
        stale.append(REPORT.relative_to(ROOT).as_posix())
    if args.check and stale:
        raise RuntimeError(f"stale V13B source-of-truth metadata: {stale}")
    print(json.dumps({"status": "PASS", "updated": stale if not args.check else [], **values}, indent=2))


if __name__ == "__main__":
    main()
