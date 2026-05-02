"""Regression test for the `make clean` target.

Background: an earlier version of `clean` ran `rm -rf data/processed`, which
ignored .gitignore negation rules and deleted the committed mimic-draft cache
(real LLM-generated text). The fix uses `git clean -fdX` which respects the
ignore rules and never touches tracked files. This test guards against
regression.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)


@pytest.fixture
def fresh_clone(tmp_path: Path) -> Path:
    """Clone the current checkout into a tmpdir so we can safely run destructive
    Makefile targets without touching the developer's working tree.
    """
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    if shutil.which("git") is None or shutil.which("make") is None:
        pytest.skip("git or make not available")
    clone = tmp_path / "clone"
    _run(["git", "clone", "--quiet", str(REPO_ROOT), str(clone)], cwd=tmp_path)
    return clone


def test_make_clean_does_not_remove_tracked_files(fresh_clone: Path) -> None:
    """Tracked files in data/processed, figures, results must survive `make clean`.

    The most important of these is data/processed/mimics/claude-opus-4-7.json
    (and any other generator caches under that directory) -- these contain real
    LLM-generated drafts that should never be deleted by a clean target.
    """
    tracked = subprocess.check_output(
        ["git", "ls-files", "data/processed", "figures", "results"],
        cwd=fresh_clone, text=True,
    ).splitlines()
    tracked_paths = [fresh_clone / p for p in tracked if p.strip()]
    # If none of these dirs has tracked files in this branch state, the test
    # is uninformative; assert the precondition we actually care about.
    assert tracked_paths, "expected at least one tracked file under data/processed/figures/results"

    # Sanity: all tracked files exist before clean.
    for p in tracked_paths:
        assert p.exists(), f"precondition: {p} should exist in fresh clone"

    _run(["make", "clean"], cwd=fresh_clone)

    missing = [p for p in tracked_paths if not p.exists()]
    assert not missing, (
        "make clean removed tracked files (regression of the destructive "
        f"`rm -rf data/processed` bug): {[str(p.relative_to(fresh_clone)) for p in missing]}"
    )


def test_make_clean_removes_ignored_artifacts(fresh_clone: Path) -> None:
    """Generated/ignored artifacts in data/processed/ should be removed by `make clean`.

    `data/processed/observations.parquet` is a generated file matched by the
    `data/processed/*` ignore rule; it is the canonical example of something
    that *should* be cleaned.
    """
    fake = fresh_clone / "data" / "processed" / "fake_artifact.parquet"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_bytes(b"junk")

    # Confirm the file is actually ignored, not tracked, in this checkout.
    is_ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(fake.relative_to(fresh_clone))],
        cwd=fresh_clone,
    ).returncode == 0
    assert is_ignored, "test setup error: fake_artifact.parquet must be ignored"

    _run(["make", "clean"], cwd=fresh_clone)

    assert not fake.exists(), "make clean did not remove an ignored artifact"
