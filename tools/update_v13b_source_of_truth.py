#!/usr/bin/env python3
"""Refresh report-generation metadata without self-referential commit claims."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys


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
        # This is deliberately the HEAD that generated the report, not a claim
        # that a committed report contains its own future commit SHA.
        "report_generated_from_commit": git_value("rev-parse", "HEAD"),
    }


def update_json(path: pathlib.Path, values: dict[str, str], *, write: bool) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for key, value in values.items():
        if payload.get(key) != value:
            payload[key] = value
            changed = True
    if "current_git_commit" in payload:
        payload.pop("current_git_commit")
        payload["current_git_commit_deprecated"] = (
            "Runtime actual HEAD must be read with: git rev-parse HEAD")
        changed = True
    if write and changed:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return changed


def generated_report_state(latest: dict) -> str:
    development = latest.get("development", {})
    artifact = development.get("v13b_artifactnet_research", {})
    f2 = development.get("v13b_f2_generator_confirmation", {})
    history = development.get("v13b_global_history", {})
    generator = artifact.get("generator_confirmation", {}) if isinstance(artifact, dict) else {}
    gate = artifact.get("selective_gate", {}) if isinstance(artifact, dict) else {}
    return "\n".join([
        "<!-- GENERATED_STATE_START -->",
        "## GENERATED CURRENT STATE",
        "",
        f"- ArtifactNet current one-crop generator Music EER: `{generator.get('music_eer', 'NOT RUN')}` (GENERATOR_CONFIRMATION)",
        f"- ArtifactNet frozen gate threshold: `{gate.get('threshold', 'NOT RUN')}`",
        f"- ArtifactNet added runtime: `{gate.get('projected_added_runtime_minutes', 'NOT RUN')} min` (PROJECTED)",
        f"- ArtifactNet license state: `{artifact.get('license_and_competition_approval', 'NOT RUN')}`",
        f"- F2 frozen weight: `{f2.get('f2_weight_frozen', 'NOT RUN')}`; generator FILE EER: `{f2.get('f2_blend_file_eer', 'NOT RUN')}`",
        f"- Global history: files `{history.get('history_files_scanned', 'NOT RUN')}`, entries `{history.get('history_entries', 'NOT RUN')}`, external root `{history.get('external_data_root', 'NOT RUN')}`",
        "- Source-disjoint full result: `NOT RUN`; FINAL: `NOT ACQUIRED / NOT RUN`.",
        "<!-- GENERATED_STATE_END -->",
    ])


def update_report(values: dict[str, str], latest: dict, *, write: bool) -> bool:
    text = REPORT.read_text(encoding="utf-8")
    replacements = {
        r"(?m)^- Branch: .*$": f"- Branch: `{values['branch']}`",
        r"(?m)^- development_base_commit: .*$": (
            f"- development_base_commit: `{values['development_base_commit']}`"),
        r"(?m)^- (?:current_git_commit|report_generated_from_commit): .*$": (
            f"- report_generated_from_commit: `{values['report_generated_from_commit']}`"),
    }
    updated = text
    for pattern, replacement in replacements.items():
        updated, count = re.subn(pattern, replacement, updated, count=1)
        if count != 1:
            raise RuntimeError(f"source-of-truth report field missing: {pattern}")
    changed = updated != text
    state_pattern = r"(?s)<!-- GENERATED_STATE_START -->.*?<!-- GENERATED_STATE_END -->"
    # Use a callback so Windows paths in generated state are not interpreted as
    # backreferences (for example ``\\U``) by ``re.sub``.
    updated, count = re.subn(
        state_pattern,
        lambda _match: generated_report_state(latest),
        updated,
        count=1,
    )
    if count != 1:
        raise RuntimeError("source-of-truth report generated-state block missing")
    changed = updated != text
    if write and changed:
        REPORT.write_text(updated, encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail rather than write stale metadata")
    args = parser.parse_args()
    # ``latest_results.json`` is the machine-readable canonical report.
    # Refresh its measured research/gate projections before applying the
    # runtime Git metadata.  Check mode is read-only by design.
    if not args.check:
        subprocess.check_call([sys.executable, str(ROOT / "scripts/manage_v13b_stages.py")], cwd=ROOT)
    values = expected()
    stale = []
    for relative in JSON_TARGETS:
        path = ROOT / relative
        if update_json(path, values, write=not args.check):
            stale.append(relative)
    latest = json.loads((ROOT / "experiments/latest_results.json").read_text(encoding="utf-8"))
    if update_report(values, latest, write=not args.check):
        stale.append(REPORT.relative_to(ROOT).as_posix())
    if args.check and stale:
        raise RuntimeError(f"stale V13B source-of-truth metadata: {stale}")
    print(json.dumps({"status": "PASS", "updated": stale if not args.check else [], **values}, indent=2))


if __name__ == "__main__":
    main()
