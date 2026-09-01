#!/usr/bin/env python3
"""Create a hash inventory of all local changes before further V13B work."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/v13b/worktree_snapshot.json"
DEVELOPMENT_BASE_COMMIT = "e8434b9c368ee5de3368d8b0b04559cf19c3ffaa"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    raw = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"], cwd=ROOT)
    entries = []
    for record in raw.decode("utf-8", errors="surrogateescape").split("\0"):
        if not record:
            continue
        status, relative = record[:2], record[3:]
        if " -> " in relative:
            relative = relative.split(" -> ", 1)[1]
        path = ROOT / relative
        if path.resolve() == OUTPUT.resolve():
            continue
        entries.append({
            "status": status,
            "path": pathlib.PurePath(relative).as_posix(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": digest(path) if path.is_file() else None,
        })
    report = {
        "schema_version": 1,
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "development_base_commit": DEVELOPMENT_BASE_COMMIT,
        "current_git_commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "modified_files": [item for item in entries if item["status"] != "??"],
        "untracked_files": [item for item in entries if item["status"] == "??"],
        "entry_count": len(entries),
        "note": "Current x1 worktree inventory; output file excludes itself. No reset, clean, or stash used.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "entries": len(entries),
                      "head": report["current_git_commit"]}, indent=2))


if __name__ == "__main__":
    main()
