"""Regression test for the figure-link path in REPRODUCTION_REPORT.md.

Background: an earlier version of `scripts/06_build_report.py` emitted image
links like `![Fig. 3](../figures/fig3_before_after.png)`. Because the report
is written to the repository root, the `../` prefix navigates one directory
*above* the repo, so every figure link rendered as a broken image on GitHub.
This test rebuilds the report and asserts the links resolve to existing
files relative to the report's own directory.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_report() -> Path:
    """Run scripts/06_build_report.py and return the path to the generated report."""
    out = REPO_ROOT / "REPRODUCTION_REPORT.md"
    # Need the result CSVs the report reads; if they aren't there, skip.
    needed = [
        REPO_ROOT / "results" / "hypothesis_tests.csv",
        REPO_ROOT / "results" / "h3_rmcorr.csv",
        REPO_ROOT / "results" / "perception_treatment_vs_control.csv",
    ]
    if not all(p.exists() for p in needed):
        pytest.skip("results/*.csv not present; run `make tests figures` first")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "06_build_report.py")],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )
    return out


_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def test_report_image_links_resolve_relative_to_report() -> None:
    report = _build_report()
    text = report.read_text(encoding="utf-8")
    links = _IMG_RE.findall(text)
    assert links, "expected at least one ![alt](path) image link in the report"
    report_dir = report.parent
    bad: list[tuple[str, Path]] = []
    for link in links:
        # Skip absolute paths and external URLs - the test is about repo-relative links.
        if link.startswith(("http://", "https://", "/")):
            continue
        # The link must NOT escape the repo root with `..`. The committed report
        # lives at the repo root, so `../figures/...` resolves outside the repo
        # and breaks on GitHub. Fail loudly if anyone reintroduces that pattern.
        target = (report_dir / link).resolve()
        try:
            target.relative_to(REPO_ROOT.resolve())
        except ValueError:
            bad.append((link, target))
            continue
        if not target.exists():
            bad.append((link, target))
    assert not bad, (
        "Some figure links in REPRODUCTION_REPORT.md don't resolve to a file "
        "inside the repo (the `../figures/` regression):\n  "
        + "\n  ".join(f"{link} -> {target}" for link, target in bad)
    )
