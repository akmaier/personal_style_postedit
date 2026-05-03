"""Stage 11: AI-text detection on LUAR embeddings.

For each "AI" approach we train a binary linear-SVM that classifies a
LUAR-MUD embedding as either *human* or *that approach's output*. We then
report the area under the ROC curve under leave-authors-out 5-fold
cross-validation. The hypothesis is that detector AUC drops as the
approach gets closer to the participant's own style:

    o4-mini  >  human-edit  >  Opus 4.7  ~  GPT-5.5

i.e. unconditioned LLM drafts are easy to spot, post-edited drafts are
harder, and frontier-LLM mimics shown a single style sample are hardest
of all. The classifier is intentionally simple (linear SVM in the
512-d LUAR space) so the result is about the embeddings, not about the
classifier.

# Leakage protocol (matches the paper's Section 5 held-out protocol)

The released study has exactly 2 unassisted control texts per
participant. We re-use the same lower-task-idx demo / higher-task-idx
held-out split as in scripts/07. Concretely, for each binary classifier
the *human* class is each participant's HELD-OUT control text -- the
one that the LLM was *not* shown during generation. The *AI* class is
that approach's output for one of the participant's 4 treatment tasks.

Cross-validation uses GroupKFold(groups=pid). This guarantees that no
author appears in both the train and the test fold of any split, which
in turn guarantees the AUC is genuinely measuring "can a held-out
participant's style be distinguished from this approach's output", not
"does the classifier remember this participant".

Two test asserts (in tests/test_detection_no_leakage.py) make the
no-leakage protocol load-bearing:
    1. for every (train_idx, test_idx) split: the set of pids in train
       and the set of pids in test must be disjoint; and
    2. each AI approach must be on the SAME 81 participants.

# Outputs

  results/detection_aucs.csv          one row per approach
  results/detection_per_fold.csv      one row per (approach, fold)
  figures/fig11_detection.{pdf,png}   AUC bar chart with bootstrap CIs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from personal_style.data import Paths, load_processed  # noqa: E402
from personal_style.embeddings import load_embeddings  # noqa: E402
from personal_style.similarity import EmbeddingTable, participant_kind_indices  # noqa: E402

SEED = 20260101
N_SPLITS = 5

# Approach labels (used as classifier names + plotted labels)
APPROACHES = [
    ("o4-mini", "llm"),         # uses the cached LLM-draft embeddings
    ("Human post-edit", "edited"),
    ("Claude Opus 4.7", "claude-opus-4-7"),
    ("GPT-5.5", "gpt-5.5"),
]
PALETTE = {
    "o4-mini": "#d95f02",
    "Human post-edit": "#1b9e77",
    "Claude Opus 4.7": "#e7298a",
    "GPT-5.5": "#7570b3",
}


def build_heldout_index(base: EmbeddingTable):
    """Per pid: dict with held_out_idx and held_out_vec, matching scripts/08."""
    pki = participant_kind_indices(base)
    out = {}
    for pid in sorted(set(base.pids.tolist())):
        ctrl = pki.get((pid, "control"), [])
        if len(ctrl) < 2:
            continue
        rows = sorted([(int(base.task_idx[i]), i) for i in ctrl], key=lambda x: x[0])
        # rows[0] = demo (LOWER task_idx, shown to LLM); rows[1] = held-out
        out[pid] = {
            "demo_idx": rows[0][0],
            "demo_vec": base.vecs[rows[0][1]],
            "held_out_idx": rows[1][0],
            "held_out_vec": base.vecs[rows[1][1]],
        }
    return out


def load_mimic_vectors(paths: Paths, generator: str) -> dict[tuple[str, int], np.ndarray]:
    """Return {(pid, task_idx): vec} for a given mimic generator from the
    same on-disk cache scripts/08 produces."""
    cache = paths.processed_dir / "mimic_embeddings.npz"
    z = np.load(cache)
    pids = z["pids"].astype(str)
    tasks = z["task_idx"].astype(int)
    gens = z["generators"].astype(str)
    vecs = z["vecs"].astype(np.float32)
    out = {}
    for p, t, g, v in zip(pids, tasks, gens, vecs):
        if g == generator:
            out[(str(p), int(t))] = v
    return out


def assemble_dataset(paths: Paths, base: EmbeddingTable, approach_kind: str
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a given approach, return X (vectors), y (1=AI, 0=human), groups (pid).

    Human class: each participant's held-out control vector (1 per pid).
    AI class: that approach's output for each treatment task (4 per pid for
        edited / opus / gpt55; 4 per pid for o4-mini draft).

    No participant's *demo* control is ever used. No participant appears
    twice in the human class. The human class is therefore exactly the
    same vector set used as the eval target in Section 5/6 of the paper,
    which is what makes the detection result an honest companion to those
    statistics rather than a separately-confounded measurement.
    """
    held = build_heldout_index(base)
    pki = participant_kind_indices(base)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    g_list: list[str] = []

    if approach_kind in {"llm", "edited"}:
        # The base embeddings cache has these directly under their kind.
        for pid, info in held.items():
            # Human (label 0) -- held-out control vector, exactly once per pid.
            X_list.append(info["held_out_vec"])
            y_list.append(0)
            g_list.append(pid)
            # AI (label 1) -- one entry per treatment task for this pid.
            for emb_i in pki.get((pid, approach_kind), []):
                X_list.append(base.vecs[emb_i])
                y_list.append(1)
                g_list.append(pid)
    else:
        # claude-opus-4-7 / gpt-5.5 -- come from the mimic cache.
        mimic = load_mimic_vectors(paths, approach_kind)
        for pid, info in held.items():
            X_list.append(info["held_out_vec"])
            y_list.append(0)
            g_list.append(pid)
            # one AI vector per treatment task for this pid
            for (p, t), v in mimic.items():
                if p != pid:
                    continue
                X_list.append(v)
                y_list.append(1)
                g_list.append(pid)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    g = np.array(g_list, dtype=object)
    return X, y, g


def cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
           seed: int = SEED) -> tuple[list[dict], dict]:
    """Run GroupKFold(n_splits=5) and return per-fold + summary AUC."""
    rng = np.random.default_rng(seed)
    gkf = GroupKFold(n_splits=N_SPLITS)
    rows = []
    aucs = []
    for fold_i, (tr, te) in enumerate(gkf.split(X, y, groups)):
        # SVM hyper-params kept tiny + fixed: this is the embedding test, not
        # an SVM hyper-search.
        clf = LinearSVC(C=1.0, dual="auto", random_state=seed + fold_i,
                        max_iter=5000)
        clf.fit(X[tr], y[tr])
        # decision_function gives a continuous score; that's what
        # roc_auc_score wants.
        scores = clf.decision_function(X[te])
        auc = float(roc_auc_score(y[te], scores))
        # Sanity check: train and test pids must be disjoint.
        train_pids = set(map(str, np.unique(groups[tr]).tolist()))
        test_pids = set(map(str, np.unique(groups[te]).tolist()))
        assert not (train_pids & test_pids), "AUTHOR LEAK"
        rows.append({
            "fold": fold_i,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_train_pids": len(train_pids),
            "n_test_pids": len(test_pids),
            "auc": auc,
        })
        aucs.append(auc)
    aucs_arr = np.array(aucs)
    # Bootstrap CI on the *fold mean* using simple percentile bootstrap.
    n_boot = 2000
    boot = rng.choice(aucs_arr, size=(n_boot, N_SPLITS), replace=True).mean(axis=1)
    summary = {
        "auc_mean": float(aucs_arr.mean()),
        "auc_std": float(aucs_arr.std(ddof=1)),
        "auc_ci_low": float(np.percentile(boot, 2.5)),
        "auc_ci_high": float(np.percentile(boot, 97.5)),
    }
    return rows, summary


def main() -> None:
    paths = Paths()
    res_dir = REPO_ROOT / "results"
    fig_dir = REPO_ROOT / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load base embeddings (control / llm / edited).
    z = load_embeddings(paths.processed_dir / "embeddings.npz")
    base = EmbeddingTable(z["pids"], z["kinds"], z["task_idx"],
                          z["scenarios"], z["vecs"])

    summary_rows = []
    fold_rows = []
    for label, kind in APPROACHES:
        X, y, g = assemble_dataset(paths, base, kind)
        n_pids = len(set(g.tolist()))
        n_human = int((y == 0).sum())
        n_ai = int((y == 1).sum())
        print(f"{label:20s}  pids={n_pids}  human={n_human}  ai={n_ai}")
        rows, summ = cv_auc(X, y, g)
        for r in rows:
            r["approach"] = label
            r["kind"] = kind
            fold_rows.append(r)
        summary_rows.append({
            "approach": label,
            "kind": kind,
            "n_pids": n_pids,
            "n_human": n_human,
            "n_ai": n_ai,
            **summ,
        })

    sdf = pd.DataFrame(summary_rows)
    fdf = pd.DataFrame(fold_rows)
    sdf.to_csv(res_dir / "detection_aucs.csv", index=False)
    fdf.to_csv(res_dir / "detection_per_fold.csv", index=False)
    print()
    print(sdf.to_string(index=False))

    # ---- Figure 11
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = sdf["approach"].tolist()
    means = sdf["auc_mean"].to_numpy()
    los = sdf["auc_ci_low"].to_numpy()
    his = sdf["auc_ci_high"].to_numpy()
    ys = np.arange(len(labels))[::-1]
    colors = [PALETTE[a] for a in labels]
    ax.barh(ys, means, color=colors, alpha=0.7, edgecolor="black")
    ax.errorbar(means, ys, xerr=[means - los, his - means],
                fmt="none", ecolor="black", capsize=3, lw=1.0)
    for i, (y, m, lo, hi) in enumerate(zip(ys, means, los, his)):
        ax.text(m + 0.012, y, f"AUC = {m:.3f}\n[{lo:.3f}, {hi:.3f}]",
                va="center", fontsize=8)
    ax.axvline(0.5, color="#888", ls=":", lw=0.9)
    ax.text(0.5, ys.min() - 0.4, "chance (0.5)",
            color="#666", ha="center", fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.4, 1.05)
    ax.set_xlabel("Detection AUC under leave-authors-out 5-fold CV\n(LinearSVC on LUAR-MUD embeddings)")
    ax.set_title("Figure 11: detecting AI text gets harder as the approach\nmoves toward the participant's natural style")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig11_detection.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig11_detection.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\nfigures/fig11_detection.{{pdf,png}} written")


if __name__ == "__main__":
    main()
