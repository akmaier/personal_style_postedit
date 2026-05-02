"""The frozen adversarial-rewriting SVM must have been trained on a set
of authors disjoint from the test-fold authors that the sub-agents
optimise against. Otherwise the result is meaningless.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK = REPO_ROOT / "data" / "processed" / "adversarial_pack.json"
SVM = REPO_ROOT / "data" / "processed" / "adversarial_svm.npz"


@pytest.mark.skipif(not (PACK.exists() and SVM.exists()),
                    reason="run scripts/14_adversarial_rewriting.py prepare first")
def test_train_and_target_pids_are_disjoint() -> None:
    z = np.load(SVM)
    train_pids = set(z["train_pids"].astype(str).tolist())
    test_pids = set(z["test_pids"].astype(str).tolist())
    assert not (train_pids & test_pids), \
        "fold-1 train and test pids overlap"

    pack = json.loads(PACK.read_text())
    target_pids = {t["pid"] for t in pack["targets"]}
    assert target_pids <= test_pids, \
        "adversarial targets contain pids not in the SVM's test set"
    assert not (target_pids & train_pids), \
        "adversarial targets contain pids that the SVM was trained on -- LEAK"


@pytest.mark.skipif(not (PACK.exists() and SVM.exists()),
                    reason="run scripts/14_adversarial_rewriting.py prepare first")
def test_initial_margins_are_decisively_positive() -> None:
    """We picked the most-confidently-AI test mimics on purpose; their
    initial margins should all be well above zero so that 'flipping' is
    a non-trivial optimisation."""
    pack = json.loads(PACK.read_text())
    margins = [t["initial_margin"] for t in pack["targets"]]
    assert all(m > 1.0 for m in margins), \
        f"initial margins not decisively positive: {margins}"
    assert pack["baseline_test_auc"] > 0.85, \
        f"baseline AUC unexpectedly low: {pack['baseline_test_auc']}"
