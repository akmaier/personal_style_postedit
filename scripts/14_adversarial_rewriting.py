"""Stage 14: agentic adversarial rewriting against the frozen detector.

Karpathy-style autoresearch on the detector from Section 7:

  1. Train the LinearSVC for the Claude Opus 4.7 detector on FOLD-1's
     training half (~65 authors). Freeze it. The held-out test half
     (~16 authors) is the universe of texts the SVM has never seen, on
     either the human or the AI side.

  2. Pick a small set of test-fold Opus mimics. For each one, run a
     T-iteration loop in which a fresh Opus 4.7 sub-agent receives:
        - the planning details for that task
        - the SVM's current AI-confidence score for the current draft
        - the trajectory of previous drafts and their scores
     and is asked to produce a new variation that minimises the AI
     score, *without* changing the substantive content of the
     planning details and *without* dropping below the paper-required
     100-200 word range.

  3. Each new variation is embedded with LUAR-MUD and scored by the
     same frozen SVM. After T iterations we report (a) the score
     trajectory, (b) the final detection AUC over the optimised drafts,
     (c) whether any individual draft crosses the SVM decision boundary
     into the human side.

The *target* is to drive the SVM's signed margin from positive (AI) to
negative (human) on every test draft, i.e. flip the classifier's vote.
That is the operationalisation of "Opus beats the classifier".

This script provides three sub-commands:

  python scripts/14_adversarial_rewriting.py prepare
      Trains and freezes the SVM, picks the fold-1 test mimics, writes
      data/processed/adversarial_pack.json (one entry per target text).

  python scripts/14_adversarial_rewriting.py score TEXT_FILE
      Embeds the text in TEXT_FILE with LUAR, scores it with the frozen
      SVM, prints {auc-equivalent margin, decision_function, predicted
      class} as JSON. This is the call sub-agents make in their loop.

  python scripts/14_adversarial_rewriting.py aggregate
      Reads all the per-target trajectories that the sub-agents wrote
      to /tmp/adversarial_runs/*.json and produces results/
      adversarial_trajectories.csv plus figures/fig13_adversarial.{pdf,png}.

The protocol is no-leakage by construction: the SVM is trained on a
disjoint set of authors from the targets, and only ever sees the
test-fold authors when it scores a draft. The human-side embeddings
the SVM was trained on are not exposed to Opus.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import importlib.util as _ilu
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

from personal_style.data import Paths, load_processed
from personal_style.embeddings import embed_texts, load_embeddings, load_luar
from personal_style.similarity import EmbeddingTable, participant_kind_indices

# Reuse the stage-11 dataset builder so the SVM sees exactly the same data.
_spec = _ilu.spec_from_file_location(
    "_d11", REPO_ROOT / "scripts" / "11_detection_experiment.py"
)
_d11 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_d11)


SEED = 20260101
ADV_DIR = Path("/tmp/adversarial_runs")
PACK_PATH = REPO_ROOT / "data" / "processed" / "adversarial_pack.json"
SVM_VEC_PATH = REPO_ROOT / "data" / "processed" / "adversarial_svm.npz"


# ---------------------------------------------------------------------------
# prepare: train fold-1 SVM, pick targets, write pack file

def prepare() -> None:
    paths = Paths()
    z = load_embeddings(paths.processed_dir / "embeddings.npz")
    base = EmbeddingTable(z["pids"], z["kinds"], z["task_idx"],
                          z["scenarios"], z["vecs"])

    X, y, g = _d11.assemble_dataset(paths, base, "claude-opus-4-7")

    gkf = GroupKFold(n_splits=_d11.N_SPLITS)
    splits = list(gkf.split(X, y, g))
    tr_idx, te_idx = splits[0]  # FOLD 1
    train_pids = sorted(set(map(str, np.unique(g[tr_idx]).tolist())))
    test_pids = sorted(set(map(str, np.unique(g[te_idx]).tolist())))
    assert not (set(train_pids) & set(test_pids)), "AUTHOR LEAK"

    clf = LinearSVC(C=1.0, class_weight="balanced", dual="auto",
                    max_iter=5000, random_state=SEED)
    clf.fit(X[tr_idx], y[tr_idx])
    test_scores = clf.decision_function(X[te_idx])
    auc_before = float(roc_auc_score(y[te_idx], test_scores))
    print(f"fold-1 trained on {len(train_pids)} pids, "
          f"test on {len(test_pids)} pids, baseline test AUC = {auc_before:.3f}")

    # Save the SVM weights (we re-load them in score-mode without re-training)
    np.savez_compressed(
        SVM_VEC_PATH,
        coef=clf.coef_.astype(np.float32),
        intercept=clf.intercept_.astype(np.float32),
        train_pids=np.array(train_pids, dtype="U64"),
        test_pids=np.array(test_pids, dtype="U64"),
    )

    # ---- Pick 5 fold-1 test Opus mimics, one per scenario where possible.
    obs, _ = load_processed(paths)
    obs["pid"] = obs["pid"].astype(str)
    treat = obs[obs["condition"] == "treatment"]

    # Map (pid, task_idx) -> AI margin against the frozen SVM
    ai_mask = (y[te_idx] == 1)
    ai_g = g[te_idx][ai_mask]
    ai_scores = test_scores[ai_mask]
    # Need to back out which draft each row came from. The dataset builder
    # appends mimic vectors in (pid, task_idx) order; reconstruct:
    mimic_index = []  # list of (pid, task_idx) parallel to ai_g
    for pid in test_pids:
        tasks = sorted(treat[treat["pid"] == pid]["task_idx"].tolist())
        for t in tasks:
            mimic_index.append((pid, int(t)))
    assert len(mimic_index) == len(ai_g), \
        f"mismatch {len(mimic_index)} vs {len(ai_g)}"

    candidates = []
    seen_scenarios = set()
    mimics_path = paths.processed_dir / "mimics" / "claude-opus-4-7.json"
    mimics = json.loads(mimics_path.read_text())
    for (pid, task_idx), score in sorted(zip(mimic_index, ai_scores),
                                          key=lambda x: -x[1]):
        # Sort by descending margin (most-confidently-AI first), pick most
        # decisive cases. Diversify by scenario.
        row = treat[(treat["pid"] == pid) & (treat["task_idx"] == task_idx)]
        if row.empty:
            continue
        scenario = str(row.iloc[0]["scenario"])
        if scenario in seen_scenarios and len(candidates) < 5:
            continue
        seen_scenarios.add(scenario)
        # Find the original Opus draft text
        draft_text = None
        for v in mimics.values():
            if v["pid"] == pid and int(v["task_idx"]) == task_idx:
                draft_text = v["text"]
                break
        if draft_text is None:
            continue
        candidates.append({
            "pid": pid, "task_idx": task_idx,
            "scenario": scenario,
            "details": str(row.iloc[0]["details"] or ""),
            "initial_text": draft_text,
            "initial_margin": float(score),
        })
        if len(candidates) >= 5:
            break

    pack = {
        "fold": 0, "n_train_pids": len(train_pids),
        "n_test_pids": len(test_pids),
        "baseline_test_auc": auc_before,
        "targets": candidates,
    }
    PACK_PATH.write_text(json.dumps(pack, indent=2, ensure_ascii=False))
    print(f"wrote {PACK_PATH} with {len(candidates)} targets")
    for c in candidates:
        print(f"  {c['scenario']:12s}  pid={c['pid'][:8]}  task={c['task_idx']}  "
              f"initial_margin={c['initial_margin']:+.3f}")


# ---------------------------------------------------------------------------
# score: embed text + return frozen SVM margin (called by sub-agents)

def score(text_path: str) -> None:
    """Embed the text in TEXT_PATH with LUAR-MUD and score it against the
    frozen SVM. Prints a JSON line so a sub-agent can parse it."""
    z = np.load(SVM_VEC_PATH)
    coef = z["coef"]            # shape (1, 512)
    intercept = z["intercept"]  # shape (1,)

    text = Path(text_path).read_text(encoding="utf-8")
    word_count = len(text.split())

    tokenizer, model, device = load_luar()
    vec = embed_texts([text], tokenizer, model, device,
                      batch_size=1, show_progress=False)[0]
    margin = float((vec @ coef.T).flatten()[0] + intercept[0])
    out = {
        "decision_function": margin,
        "predicted_class": "AI" if margin > 0 else "Human",
        "ai_score_higher_is_more_aiish": margin,
        "word_count": int(word_count),
    }
    print(json.dumps(out))


# ---------------------------------------------------------------------------
# aggregate: collect all sub-agent trajectories and produce Figure 13

def aggregate() -> None:
    res_dir = REPO_ROOT / "results"
    fig_dir = REPO_ROOT / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(ADV_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"no trajectories found in {ADV_DIR}")
    rows = []
    for f in files:
        traj = json.loads(f.read_text())
        target_id = traj.get("target_id") or f.stem
        for i, r in enumerate(traj["iterations"]):
            # Sub-agents used slightly different field names; accept any.
            margin = r.get("margin", r.get("decision_function",
                                           r.get("ai_score_higher_is_more_aiish")))
            if margin is None:
                raise KeyError(f"no margin field in {f.name} iter {i}: {list(r.keys())}")
            rows.append({
                "target": target_id,
                "scenario": traj["scenario"],
                "pid": traj["pid"],
                "iteration": i,
                "margin": float(margin),
                "word_count": int(r.get("word_count", -1)),
                "predicted_class": r["predicted_class"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(res_dir / "adversarial_trajectories.csv", index=False)
    print(f"loaded {len(files)} trajectories -> "
          f"{len(df)} (target, iter) rows")

    # Key summary statistics
    summary = df.groupby("target").agg(
        scenario=("scenario", "first"),
        n_iter=("iteration", "max"),
        margin_initial=("margin", "first"),
        margin_final=("margin", "last"),
        margin_min=("margin", "min"),
        flipped_to_human_min=("margin", lambda s: bool((s < 0).any())),
        flipped_to_human_final=("margin", lambda s: bool(s.iloc[-1] < 0)),
    ).reset_index()
    summary.to_csv(res_dir / "adversarial_summary.csv", index=False)
    print("\nper-target summary:")
    print(summary.to_string(index=False))

    # ---- Figure 13: trajectories
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [3, 2]})

    ax = axes[0]
    cmap = plt.colormaps.get_cmap("tab10")
    for i, (target, sub) in enumerate(df.groupby("target")):
        sub = sub.sort_values("iteration")
        ax.plot(sub["iteration"], sub["margin"], "-o",
                color=cmap(i), markersize=3, lw=1.0,
                label=f"{sub['scenario'].iloc[0]} (pid {sub['pid'].iloc[0][:8]})")
    ax.axhline(0, color="black", lw=1.0, ls="--")
    ax.text(0.02, 0.04, "AI side", transform=ax.transAxes, color="#7570b3",
            fontsize=9)
    ax.text(0.02, -0.04, "Human side (decision flipped)", transform=ax.transAxes,
            color="#1b9e77", va="top", fontsize=9)
    ax.set_xlabel("iteration")
    ax.set_ylabel("frozen-SVM margin\n(positive = AI)")
    ax.set_title("Adversarial rewriting trajectories")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    init = summary["margin_initial"].to_numpy()
    final = summary["margin_final"].to_numpy()
    ax.scatter(init, final, s=80, color="#e7298a", edgecolor="black", zorder=3)
    for _, r in summary.iterrows():
        ax.annotate(r["scenario"], (r["margin_initial"], r["margin_final"]),
                    xytext=(4, 2), textcoords="offset points", fontsize=8)
    lo = min(init.min(), final.min()) - 0.2
    hi = max(init.max(), final.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], color="#888", ls=":", lw=1.0)
    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.axvline(0, color="black", lw=0.7, ls="--")
    ax.set_xlabel("initial margin")
    ax.set_ylabel("final margin")
    ax.set_title("Initial vs.\u00a0final (after $T$ iterations)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig13_adversarial.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig13_adversarial.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\nfigures/fig13_adversarial.{{pdf,png}} written")


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sc = sub.add_parser("score")
    sc.add_argument("text_file")
    sub.add_parser("aggregate")
    args = p.parse_args()

    if args.cmd == "prepare":
        prepare()
    elif args.cmd == "score":
        score(args.text_file)
    elif args.cmd == "aggregate":
        aggregate()


if __name__ == "__main__":
    main()
