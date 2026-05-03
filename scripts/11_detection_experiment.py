r"""Stage 11: AI-text detection on LUAR embeddings.

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

# Detection task framing and the no-leakage protocol

The classifier must learn the question "is this text **human** or **AI**?",
\emph{not} the question "is this text by **this specific author**?".
Two design choices make this concrete:

  1. The human class uses *both* unassisted control texts of each
     participant (162 human samples total: 81 pids x 2 controls). If we
     used only one (e.g. only the held-out control), the classifier
     would only ever see one human prototype per author and could
     trivially solve the task by remembering author-specific embedding
     directions. With both controls in, the human class is a proper
     stylistic distribution drawn from 81 different authors.

  2. The CV is GroupKFold(groups=pid, n_splits=5): every author appears
     in *exactly one* of train or test in each split. The test fold
     therefore contains entirely unseen authors, on both the human
     side and the AI side, so a high test-fold AUC genuinely measures
     "can the classifier discriminate human writing from this approach
     on authors it has never seen before".

These two invariants are both load-bearing. tests/test_detection_no_leakage.py
asserts them on every CV split for every approach.

Note that the AI samples are still the embeddings of the LLM's
generated drafts; the demo control text influences the *content* of
the generated draft, but the LLM's embedding vector is not a function
of the demo control alone. So including the demo control on the human
side does not bleed information into the AI side -- the two halves of
the contrast remain genuinely contrastive.

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

    Human class (label 0): BOTH unassisted control texts of each participant
        (162 samples total = 81 pids x 2 controls). Using both controls makes
        the human class a proper distribution rather than 81 author-specific
        prototypes -- otherwise the classifier could shortcut the task by
        remembering author-specific embedding directions instead of learning
        the broader human-vs-AI distinction.

    AI class (label 1): that approach's output for each treatment task
        (4 per pid; 324 samples).

    Group label is the participant id, so GroupKFold can guarantee that no
    author appears in both train and test in any CV split.
    """
    pki = participant_kind_indices(base)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    g_list: list[str] = []

    pids_with_both = []
    for pid in sorted(set(base.pids.tolist())):
        ctrl_idx = pki.get((pid, "control"), [])
        if len(ctrl_idx) < 2:
            continue
        pids_with_both.append(pid)
        # Human samples -- BOTH unassisted controls.
        for ei in ctrl_idx:
            X_list.append(base.vecs[ei])
            y_list.append(0)
            g_list.append(pid)

    if approach_kind in {"llm", "edited"}:
        for pid in pids_with_both:
            for emb_i in pki.get((pid, approach_kind), []):
                X_list.append(base.vecs[emb_i])
                y_list.append(1)
                g_list.append(pid)
    else:
        mimic = load_mimic_vectors(paths, approach_kind)
        for pid in pids_with_both:
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
        # Class weight = balanced because the human class has 2 samples per
        # author (162 total) while the AI class has 4 samples per author
        # (324 total); without re-weighting the optimum decision threshold
        # would drift toward the majority class and bias the AUC.
        # AUC is computed on the decision function so it is threshold-free,
        # but balanced weighting also stabilises the SVM optimisation.
        clf = LinearSVC(C=1.0, dual="auto", class_weight="balanced",
                        random_state=seed + fold_i, max_iter=5000)
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
