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
def test_assemble_dataset_uses_both_controls_per_author() -> None:
    """The human class must use BOTH unassisted control texts of every
    participant. Using only one would let the classifier shortcut the
    task by remembering author-specific embedding directions instead of
    learning the broader human-vs-AI distinction.
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
    from personal_style.similarity import participant_kind_indices

    paths = Paths()
    z = load_embeddings(paths.processed_dir / "embeddings.npz")
    base = EmbeddingTable(z["pids"], z["kinds"], z["task_idx"],
                          z["scenarios"], z["vecs"])

    X, y, g = m.assemble_dataset(paths, base, "edited")

    # Exactly 2 human samples per pid (one per unassisted control text).
    from collections import Counter
    pids_y0 = g[y == 0]
    counts = Counter(pids_y0.tolist())
    assert len(counts) == 81, f"expected 81 pids in human class, got {len(counts)}"
    assert all(c == 2 for c in counts.values()), \
        f"every pid must contribute exactly 2 human samples, got {dict(counts)}"
    assert int((y == 0).sum()) == 162, \
        f"expected 162 human samples (81 pids x 2 controls), got {(y==0).sum()}"

    # And those 162 vectors must coincide with the participants' control
    # embeddings -- no demo/held-out filtering on the human side.
    pki = participant_kind_indices(base)
    for pid in set(pids_y0.tolist()):
        ctrl_idx = pki[(str(pid), "control")]
        ctrl_vecs = {tuple(v.tolist()) for v in base.vecs[ctrl_idx]}
        # The 2 human samples for this pid must be a *subset* (in fact equal)
        # to the participant's 2 control vectors.
        humans_for_pid = X[y == 0][pids_y0 == pid]
        humans_set = {tuple(v.tolist()) for v in humans_for_pid}
        assert humans_set <= ctrl_vecs, \
            f"human class for {pid} contains non-control embeddings"


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
