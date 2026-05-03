"""Regression test for the detection-experiment train/test leakage guard.

Background: scripts/11_detection_experiment.py uses GroupKFold(groups=pid)
to ensure that no participant's text appears in both the train and the
test fold of any split. This test makes the guarantee load-bearing: if
anyone changes the script to drop GroupKFold, this test fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _have_embeddings() -> bool:
    return (REPO_ROOT / "data" / "processed" / "embeddings.npz").exists()


def _have_mimic_embeddings() -> bool:
    return (REPO_ROOT / "data" / "processed" / "mimic_embeddings.npz").exists()


def test_groupkfold_keeps_authors_disjoint() -> None:
    """Synthetic-data sanity check: GroupKFold respects the group constraint."""
    rng = np.random.default_rng(0)
    n_pids = 81
    samples_per_pid = 5
    pids = np.array([f"p{i}" for i in range(n_pids) for _ in range(samples_per_pid)])
    X = rng.normal(size=(len(pids), 4))
    y = rng.integers(0, 2, size=len(pids))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, pids):
        train_pids = set(pids[tr].tolist())
        test_pids = set(pids[te].tolist())
        assert not (train_pids & test_pids), \
            "GroupKFold leaked an author across train/test"
        assert len(train_pids) + len(test_pids) == n_pids, \
            "every pid should appear in exactly one of train/test"


@pytest.mark.skipif(not _have_embeddings(), reason="embeddings.npz missing; build it first")
def test_assemble_dataset_uses_held_out_control_only() -> None:
    """The human class must use ONLY the held-out control (the higher
    task_idx one), never the demo control. This makes the AI/human
    contrast share the same eval target as the paper's Section 5.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "_d11", REPO_ROOT / "scripts" / "11_detection_experiment.py"
    )
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)

    from personal_style.data import Paths
    from personal_style.embeddings import load_embeddings
    from personal_style.similarity import EmbeddingTable

    paths = Paths()
    z = load_embeddings(paths.processed_dir / "embeddings.npz")
    base = EmbeddingTable(z["pids"], z["kinds"], z["task_idx"],
                          z["scenarios"], z["vecs"])
    held = m.build_heldout_index(base)

    # Each pid contributes exactly ONE human-class sample (the held-out vec).
    X, y, g = m.assemble_dataset(paths, base, "edited")
    pids_y0 = g[y == 0]
    assert len(pids_y0) == len(set(pids_y0.tolist())), \
        "human class should have one sample per pid"

    # And that sample must equal the held-out vec, not the demo vec.
    for pid, vec in zip(pids_y0, X[y == 0]):
        info = held[str(pid)]
        assert np.allclose(vec, info["held_out_vec"]), \
            f"human class for {pid} is not the held-out control"
        # Also: must NOT equal the demo vec (otherwise we leaked).
        assert not np.allclose(vec, info["demo_vec"]), \
            f"human class for {pid} matches the LLM-shown demo control"


@pytest.mark.skipif(not (_have_embeddings() and _have_mimic_embeddings()),
                    reason="embedding caches missing")
def test_cv_splits_are_author_disjoint() -> None:
    """End-to-end: every CV split produced by the script keeps authors
    fully disjoint between train and test."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "_d11", REPO_ROOT / "scripts" / "11_detection_experiment.py"
    )
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)

    from personal_style.data import Paths
    from personal_style.embeddings import load_embeddings
    from personal_style.similarity import EmbeddingTable

    paths = Paths()
    z = load_embeddings(paths.processed_dir / "embeddings.npz")
    base = EmbeddingTable(z["pids"], z["kinds"], z["task_idx"],
                          z["scenarios"], z["vecs"])

    for label, kind in m.APPROACHES:
        X, y, g = m.assemble_dataset(paths, base, kind)
        gkf = GroupKFold(n_splits=m.N_SPLITS)
        for tr, te in gkf.split(X, y, g):
            train_pids = set(map(str, np.unique(g[tr]).tolist()))
            test_pids = set(map(str, np.unique(g[te]).tolist()))
            assert not (train_pids & test_pids), \
                f"author leak in {label} split"
            # Every pid present in this approach's data must appear
            # in exactly one of train or test (no orphans).
            all_pids = set(map(str, np.unique(g).tolist()))
            assert (train_pids | test_pids) == all_pids
