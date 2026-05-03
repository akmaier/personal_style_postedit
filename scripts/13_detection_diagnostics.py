"""Stage 13: sanity-check the detection result of stage 11.

The user looked at the t-SNE plot in figures/fig12b_embedding_tsne.png and
noted that human controls and Claude Opus 4.7 mimics seem to overlap
visually, which is hard to square with the reported AUC of 0.952. This
script does six sanity checks that, taken together, distinguish three
possibilities:

  (i)   The AUC is honest -- LUAR has 512 dimensions and t-SNE only shows 2,
        so the linear SVM may just be picking up structure that the 2-D
        projection flattens out.

  (ii)  The AUC is inflated by an underdetermined-regression effect: with
        ~390 training samples and 513 parameters (512 weights + 1 bias),
        a hard-margin LinearSVC can fit a separating hyperplane for almost
        ANY binary labelling, including a nonsense one. AUC on the test
        fold is then mostly noise around 0.5 only if the labels are random;
        but if there is even a weak true signal, the SVM will find a
        separating hyperplane in 512-d that captures it AND a lot of
        irrelevant author-correlated variation.

  (iii) The AUC is inflated by a length/format confound -- LLM mimics tend
        to write 150-word polished paragraphs while human controls have
        more variable length.

The script runs all six diagnostics, prints them, and writes them to
results/detection_diagnostics.csv. The conclusion is then drawn in the
paper based on what these numbers actually show, not on what we hoped
they would show.

Outputs:
  results/detection_diagnostics.csv  -- one row per diagnostic
  results/detection_diagnostics.md   -- human-readable report
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import importlib.util as _ilu
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

from personal_style.data import Paths, load_processed
from personal_style.embeddings import load_embeddings
from personal_style.similarity import EmbeddingTable

# Re-use the stage-11 dataset builder so the diagnostics speak to the same data.
_spec = _ilu.spec_from_file_location(
    "_d11", REPO_ROOT / "scripts" / "11_detection_experiment.py"
)
_d11 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_d11)

SEED = 20260101
N_SPLITS = 5

APPROACHES = list(_d11.APPROACHES)


def author_overlap_audit(X, y, g):
    """Print the train/test pid sets for every fold so we can see with our
    own eyes that no author leaks across."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    rows = []
    for fold_i, (tr, te) in enumerate(gkf.split(X, y, g)):
        train_pids = set(map(str, np.unique(g[tr]).tolist()))
        test_pids = set(map(str, np.unique(g[te]).tolist()))
        overlap = train_pids & test_pids
        rows.append({
            "fold": fold_i,
            "n_train_samples": int(len(tr)),
            "n_test_samples": int(len(te)),
            "n_train_pids": len(train_pids),
            "n_test_pids": len(test_pids),
            "n_overlap_pids": len(overlap),
        })
    return pd.DataFrame(rows)


def cv_with_clf(X, y, g, clf_factory, seed=SEED):
    gkf = GroupKFold(n_splits=N_SPLITS)
    aucs = []
    for fold_i, (tr, te) in enumerate(gkf.split(X, y, g)):
        clf = clf_factory(seed + fold_i)
        clf.fit(X[tr], y[tr])
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(X[te])
        else:
            scores = clf.predict_proba(X[te])[:, 1]
        aucs.append(float(roc_auc_score(y[te], scores)))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1)), aucs


def diag_param_count(X, y, g, label):
    n_train_per_fold = []
    gkf = GroupKFold(n_splits=N_SPLITS)
    for tr, te in gkf.split(X, y, g):
        n_train_per_fold.append(int(len(tr)))
    n_params = X.shape[1] + 1
    return {
        "diagnostic": "A. param/sample count",
        "approach": label,
        "n_features": int(X.shape[1]),
        "n_params_LinearSVC": n_params,
        "mean_n_train_per_fold": float(np.mean(n_train_per_fold)),
        "min_n_train_per_fold": int(min(n_train_per_fold)),
        "max_n_train_per_fold": int(max(n_train_per_fold)),
        "underdetermined": bool(n_params > min(n_train_per_fold)),
        "auc_mean": np.nan, "auc_std": np.nan,
    }


def diag_shuffled(X, y, g, label, seed=SEED):
    """Train the same LinearSVC but on PERMUTED labels.
    With proper GroupKFold and no author leak, the test AUC should hover
    around 0.5. Anything substantially above 0.5 means something is bleeding
    into the supposedly-random labels."""
    rng = np.random.default_rng(seed)
    aucs = []
    gkf = GroupKFold(n_splits=N_SPLITS)
    for tr, te in gkf.split(X, y, g):
        y_shuf = y.copy()
        rng.shuffle(y_shuf)
        clf = LinearSVC(C=1.0, class_weight="balanced", dual="auto",
                        max_iter=5000, random_state=seed)
        clf.fit(X[tr], y_shuf[tr])
        scores = clf.decision_function(X[te])
        aucs.append(float(roc_auc_score(y_shuf[te], scores)))
    m, s, _ = float(np.mean(aucs)), float(np.std(aucs, ddof=1)), aucs
    return {
        "diagnostic": "B. shuffled labels",
        "approach": label, "n_features": int(X.shape[1]),
        "auc_mean": m, "auc_std": s,
    }


def diag_length_only(paths, base, kind, label):
    """Single-feature baseline: word count of the text. If LLM-vs-human is
    largely a length game, this should already give a high AUC."""
    obs, _ = load_processed(paths)
    obs["pid"] = obs["pid"].astype(str)
    pki_idx = _d11.participant_kind_indices(base)
    pids = sorted({pid for pid, _ in pki_idx if _ == "control"})

    X_list, y_list, g_list = [], [], []
    for pid in pids:
        # human: word counts of both controls
        ctrl_rows = obs[(obs["pid"] == pid) & (obs["condition"] == "control")]
        for _, r in ctrl_rows.iterrows():
            X_list.append([len(str(r["final_text"]).split())])
            y_list.append(0); g_list.append(pid)
        # ai: word counts depend on kind
        tr_rows = obs[(obs["pid"] == pid) & (obs["condition"] == "treatment")]
        if kind == "llm":
            texts = tr_rows["llm_draft"]
        elif kind == "edited":
            texts = tr_rows["final_text"]
        else:
            # Use the cached mimic generator JSON for length info
            mimic = _d11.load_mimic_vectors(paths, kind)  # not text -- skip
            texts = []
        if kind in {"claude-opus-4-7", "gpt-5.5"}:
            # Read mimic JSON for word counts.
            import json
            mimic_path = paths.processed_dir / "mimics" / f"{kind}.json"
            cache = json.loads(mimic_path.read_text())
            entries = [v for v in cache.values() if v["pid"] == pid]
            for e in entries:
                X_list.append([len(str(e["text"]).split())])
                y_list.append(1); g_list.append(pid)
        else:
            for t in texts:
                X_list.append([len(str(t).split())])
                y_list.append(1); g_list.append(pid)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    g = np.array(g_list, dtype=object)

    def make(seed):
        return LinearSVC(C=1.0, class_weight="balanced", dual="auto",
                         max_iter=5000, random_state=seed)

    m, s, _ = cv_with_clf(X, y, g, make)
    return {
        "diagnostic": "C. length-only baseline",
        "approach": label, "n_features": 1,
        "auc_mean": m, "auc_std": s,
    }


def diag_pca32(X, y, g, label, n_comp=32):
    """LinearSVC after PCA(32). Cuts param count from 513 to 33,
    well below the per-fold sample count, so the regime is now
    properly determined."""
    pca = PCA(n_components=n_comp, random_state=SEED)
    Xp = pca.fit_transform(X)

    def make(seed):
        return LinearSVC(C=1.0, class_weight="balanced", dual="auto",
                         max_iter=5000, random_state=seed)

    m, s, _ = cv_with_clf(Xp, y, g, make)
    return {
        "diagnostic": f"E. LinearSVC after PCA({n_comp})",
        "approach": label, "n_features": int(Xp.shape[1]),
        "auc_mean": m, "auc_std": s,
    }


def diag_strong_regularization(X, y, g, label):
    """Heavily regularised L2-logistic regression. C=0.001 forces almost
    all weights toward zero, so any AUC above 0.5 is signal that even
    a low-capacity model can pick up."""

    def make(seed):
        return LogisticRegression(C=1e-3, class_weight="balanced",
                                  max_iter=5000, random_state=seed)

    m, s, _ = cv_with_clf(X, y, g, make)
    return {
        "diagnostic": "F. L2-logreg, C=1e-3 (heavy regularisation)",
        "approach": label, "n_features": int(X.shape[1]),
        "auc_mean": m, "auc_std": s,
    }


def diag_cross_generator(paths, base):
    """D. Train detector on Opus mimics; test on GPT-5.5 mimics
    (and vice versa). If the AUC stays >> 0.5, what we're picking up
    is a generic AI signature, not a model-specific pattern."""
    rows = []
    for src, tgt in [("claude-opus-4-7", "gpt-5.5"),
                     ("gpt-5.5", "claude-opus-4-7")]:
        Xs, ys, gs = _d11.assemble_dataset(paths, base, src)
        Xt, yt, gt = _d11.assemble_dataset(paths, base, tgt)
        # Use ALL of source as training (no CV needed, separate-set evaluation).
        clf = LinearSVC(C=1.0, class_weight="balanced", dual="auto",
                        max_iter=5000, random_state=SEED)
        clf.fit(Xs, ys)
        # Evaluate on target's full data.
        scores = clf.decision_function(Xt)
        auc = float(roc_auc_score(yt, scores))
        rows.append({
            "diagnostic": "D. cross-model generalisation",
            "approach": f"train={src}, test={tgt}",
            "n_features": int(Xs.shape[1]),
            "auc_mean": auc, "auc_std": np.nan,
        })
    return rows


def main() -> None:
    paths = Paths()
    res_dir = REPO_ROOT / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    z = load_embeddings(paths.processed_dir / "embeddings.npz")
    base = EmbeddingTable(z["pids"], z["kinds"], z["task_idx"],
                          z["scenarios"], z["vecs"])

    print("=" * 70)
    print("DIAGNOSTIC A: parameter count vs. training-set size; author audit")
    print("=" * 70)
    rows = []
    audit_rows = []
    for label, kind in APPROACHES:
        X, y, g = _d11.assemble_dataset(paths, base, kind)
        # Author overlap audit per fold
        audit = author_overlap_audit(X, y, g)
        audit["approach"] = label
        audit_rows.append(audit)
        # Param count
        rows.append(diag_param_count(X, y, g, label))
        print(f"\n  {label}:")
        print(f"    n_features = {X.shape[1]}, n_params(LinearSVC) = {X.shape[1]+1}")
        print(f"    n_train per fold = {audit['n_train_samples'].tolist()}")
        print(f"    n_test  per fold = {audit['n_test_samples'].tolist()}")
        print(f"    n_train_pids per fold = {audit['n_train_pids'].tolist()}")
        print(f"    n_test_pids  per fold = {audit['n_test_pids'].tolist()}")
        print(f"    n_overlap_pids per fold = {audit['n_overlap_pids'].tolist()}  "
              f"(should be all 0)")
        if audit["n_overlap_pids"].max() > 0:
            print(f"    ** AUTHOR LEAK DETECTED **")
        if X.shape[1] + 1 > audit["n_train_samples"].min():
            print(f"    ** UNDERDETERMINED: more SVM params than train samples **")

    # Save audit
    pd.concat(audit_rows, ignore_index=True).to_csv(
        res_dir / "detection_fold_audit.csv", index=False)

    print()
    print("=" * 70)
    print("DIAGNOSTIC B: shuffle-labels baseline (must give AUC ~0.5)")
    print("=" * 70)
    for label, kind in APPROACHES:
        X, y, g = _d11.assemble_dataset(paths, base, kind)
        r = diag_shuffled(X, y, g, label)
        rows.append(r)
        print(f"  {label:20s}  shuffled-AUC = {r['auc_mean']:.3f} +/- {r['auc_std']:.3f}")

    print()
    print("=" * 70)
    print("DIAGNOSTIC C: length-only baseline (single feature)")
    print("=" * 70)
    for label, kind in APPROACHES:
        try:
            r = diag_length_only(paths, base, kind, label)
            rows.append(r)
            print(f"  {label:20s}  length-only AUC = {r['auc_mean']:.3f} +/- {r['auc_std']:.3f}")
        except Exception as e:
            print(f"  {label}: SKIPPED ({e})")

    print()
    print("=" * 70)
    print("DIAGNOSTIC D: cross-model generalisation (train on one LLM, test on other)")
    print("=" * 70)
    for r in diag_cross_generator(paths, base):
        rows.append(r)
        print(f"  {r['approach']:50s}  AUC = {r['auc_mean']:.3f}")

    print()
    print("=" * 70)
    print("DIAGNOSTIC E: LinearSVC after PCA(32) (no longer underdetermined)")
    print("=" * 70)
    for label, kind in APPROACHES:
        X, y, g = _d11.assemble_dataset(paths, base, kind)
        r = diag_pca32(X, y, g, label, n_comp=32)
        rows.append(r)
        print(f"  {label:20s}  PCA(32) + LinearSVC AUC = {r['auc_mean']:.3f} +/- {r['auc_std']:.3f}")

    print()
    print("=" * 70)
    print("DIAGNOSTIC F: heavily-regularised LR (C=1e-3) on raw 512-d LUAR")
    print("=" * 70)
    for label, kind in APPROACHES:
        X, y, g = _d11.assemble_dataset(paths, base, kind)
        r = diag_strong_regularization(X, y, g, label)
        rows.append(r)
        print(f"  {label:20s}  LR(C=1e-3) AUC = {r['auc_mean']:.3f} +/- {r['auc_std']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(res_dir / "detection_diagnostics.csv", index=False)
    print(f"\nresults/detection_diagnostics.csv written ({len(df)} rows)")
    print(f"results/detection_fold_audit.csv written")


if __name__ == "__main__":
    main()
