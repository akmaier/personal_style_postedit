"""Stage 10: comprehensive final statistical assessment across all 4 approaches.

For every one of the 324 treatment tasks we have, on the same held-out target:

  - sim_o4_mini       : o4-mini draft (no style sample shown)
  - sim_human_edit    : human post-edit of that o4-mini draft
  - sim_opus          : Claude Opus 4.7 mimic (1 style sample, held-out target unseen)
  - sim_gpt55         : GPT-5.5 mimic            (1 style sample, held-out target unseen)

This script does:

  1. Build a tidy 324-row wide table.
  2. Friedman omnibus across all 4 approaches.
  3. Posthoc Wilcoxon signed-rank for every pair (6 pairs), with paired
     permutation Hedges' g and 1000-bootstrap 95 % CIs, BH-FDR over the 6.
  4. "Did it beat the human?" win-rate per approach, with binomial 95 % CI
     and an exact two-sided binomial test against 0.5.
  5. Per-scenario means + bootstrap 95 % CIs for each approach.
  6. Figure 10: forest plot of all-pairs Hedges' g, bar chart of win-vs-human
     rate, and per-scenario means.

Outputs:
  - results/final_pairwise_tests.csv
  - results/final_win_vs_human.csv
  - results/final_friedman.csv
  - results/final_per_scenario.csv
  - figures/fig10_final_assessment.{pdf,png}
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest, friedmanchisquare, wilcoxon  # noqa: E402

from personal_style.data import Paths  # noqa: E402
from personal_style.stats import benjamini_hochberg, run_test  # noqa: E402

APPROACHES = ["sim_o4_mini", "sim_human_edit", "sim_opus", "sim_gpt55"]
LABELS = {
    "sim_o4_mini": "o4-mini draft",
    "sim_human_edit": "Human post-edit",
    "sim_opus": "Claude Opus 4.7",
    "sim_gpt55": "GPT-5.5",
}
COLORS = {
    "sim_o4_mini": "#d95f02",
    "sim_human_edit": "#1b9e77",
    "sim_opus": "#e7298a",
    "sim_gpt55": "#7570b3",
}


def build_wide_table(paths: Paths) -> pd.DataFrame:
    df = pd.read_parquet(paths.processed_dir / "mimic_similarities.parquet")
    # The o4-mini and human columns are duplicated across generators; dedupe to one
    # row per (pid, task_idx).
    base = df[["pid", "task_idx", "scenario", "sim_llm_heldout", "sim_edited_heldout"]].drop_duplicates(
        subset=["pid", "task_idx"]
    )
    base = base.rename(columns={
        "sim_llm_heldout": "sim_o4_mini",
        "sim_edited_heldout": "sim_human_edit",
    })
    pivot = df.pivot_table(index=["pid", "task_idx"], columns="generator",
                           values="sim_mimic_heldout").reset_index()
    pivot = pivot.rename(columns={"claude-opus-4-7": "sim_opus", "gpt-5.5": "sim_gpt55"})
    out = base.merge(pivot, on=["pid", "task_idx"], how="left")
    out = out.dropna(subset=APPROACHES).reset_index(drop=True)
    return out


def friedman_omnibus(df: pd.DataFrame) -> dict:
    cols = [df[c].to_numpy() for c in APPROACHES]
    stat, p = friedmanchisquare(*cols)
    return {"chi2": float(stat), "p": float(p), "k": len(APPROACHES), "n": len(df)}


def all_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = list(combinations(APPROACHES, 2))
    for a, b in pairs:
        x = df[a].to_numpy()
        y = df[b].to_numpy()
        wstat, wp = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        # Paired permutation Hedges' g + 1000-boot CI
        res = run_test(f"{a} vs {b}", x, y, paired=True, n_perm=10000, n_boot=1000,
                       seed=hash((a, b)) & 0xFFFFFFFF)
        rows.append({
            "a": a, "b": b,
            "label_a": LABELS[a], "label_b": LABELS[b],
            "n": len(df),
            "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "median_diff": float(np.median(x - y)),
            "g": res.g, "ci_low": res.ci_low, "ci_high": res.ci_high,
            "p_perm": res.p,
            "p_wilcoxon": float(wp), "wilcoxon_stat": float(wstat),
        })
    out = pd.DataFrame(rows)
    p_bh, reject = benjamini_hochberg(out["p_perm"].to_numpy(), q=0.05)
    out["p_perm_bh"] = p_bh
    out["bh_reject"] = reject
    return out


def win_vs_human(df: pd.DataFrame) -> pd.DataFrame:
    """For each approach, fraction of tasks where it >= human-edit on held-out sim."""
    rows = []
    h = df["sim_human_edit"].to_numpy()
    for a in APPROACHES:
        x = df[a].to_numpy()
        if a == "sim_human_edit":
            continue
        wins = int((x > h).sum())
        ties = int((x == h).sum())
        n = len(df)
        # Exact two-sided binomial test against 0.5 (treat ties as half-wins).
        wins_for_test = int(wins + 0.5 * ties)
        bt = binomtest(k=wins_for_test, n=n, p=0.5, alternative="two-sided")
        rows.append({
            "approach": a,
            "label": LABELS[a],
            "wins": wins,
            "ties": ties,
            "n": n,
            "win_rate": (wins + 0.5 * ties) / n,
            "ci_low": bt.proportion_ci().low,
            "ci_high": bt.proportion_ci().high,
            "p_binomial": float(bt.pvalue),
        })
    return pd.DataFrame(rows)


def per_scenario(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(20260101)
    rows = []
    for scen, sub in df.groupby("scenario"):
        for a in APPROACHES:
            x = sub[a].to_numpy()
            if len(x) == 0:
                continue
            mean = float(x.mean())
            # Bootstrap 95 % CI of the mean
            boots = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, len(x), size=len(x))
                boots[b] = x[idx].mean()
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({
                "scenario": scen,
                "approach": a,
                "label": LABELS[a],
                "n": len(sub),
                "mean": mean,
                "ci_low": float(lo),
                "ci_high": float(hi),
            })
    return pd.DataFrame(rows)


def figure_10(wide: pd.DataFrame, pairs: pd.DataFrame, wins: pd.DataFrame,
              per_scen: pd.DataFrame, upper_bound: float, fig_dir: Path) -> None:
    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[3.0, 2.6, 3.0], hspace=0.55, wspace=0.3)

    # (a) Violin/strip + dashed human + dashed upper-bound
    ax = fig.add_subplot(gs[0, :])
    series = [wide[c].to_numpy() for c in APPROACHES]
    parts = ax.violinplot(series, positions=range(len(series)), widths=0.85,
                          showextrema=False)
    for pc, c in zip(parts["bodies"], [COLORS[k] for k in APPROACHES]):
        pc.set_facecolor(c); pc.set_alpha(0.5); pc.set_edgecolor("black")
    rng = np.random.RandomState(0)
    for i, arr in enumerate(series):
        ax.scatter(np.full_like(arr, i, dtype=float) + rng.uniform(-0.07, 0.07, size=len(arr)),
                   arr, s=5, color="black", alpha=0.25)
        ax.hlines(arr.mean(), i - 0.32, i + 0.32, color="black", lw=1.6)
        ax.text(i, arr.mean() + 0.018, f"{arr.mean():.3f}",
                ha="center", fontsize=9, fontweight="bold", color="black")
    h_mean = wide["sim_human_edit"].mean()
    ax.axhline(h_mean, color=COLORS["sim_human_edit"], ls=":", lw=1.2, alpha=0.7)
    ax.text(len(series) - 0.5, h_mean - 0.022,
            f"human post-edit threshold = {h_mean:.3f}",
            ha="right", va="top", fontsize=9, color=COLORS["sim_human_edit"])
    ax.axhline(upper_bound, color="#444", ls="--", lw=1.0)
    ax.text(len(series) - 0.5, upper_bound + 0.005,
            f"same-author control \u2194 control upper bound = {upper_bound:.3f}",
            ha="right", va="bottom", fontsize=9, color="#333")
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([LABELS[k] for k in APPROACHES])
    ax.set_ylabel("LUAR cosine sim. to held-out control text")
    ax.set_title("(a) Distributions across all 4 approaches (n = 324 tasks)")

    # (b) Forest plot of all 6 pairwise Hedges' g
    ax = fig.add_subplot(gs[1, 0])
    pairs_sorted = pairs.copy().sort_values("g")
    ys = np.arange(len(pairs_sorted))
    for y, (_, r) in zip(ys, pairs_sorted.iterrows()):
        sig = r["bh_reject"]
        color = "#222" if sig else "#888"
        ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=color, lw=1.6)
        ax.plot(r["g"], y, "o", color=color, ms=6)
        ax.text(r["ci_high"] + 0.05, y, f"g={r['g']:+.2f}", va="center", fontsize=8)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['label_a']}  vs  {r['label_b']}"
                        for _, r in pairs_sorted.iterrows()], fontsize=9)
    ax.set_xlabel("Hedges' g (paired)")
    ax.set_title("(b) All pairwise effect sizes\n(black = BH-FDR significant at q=0.05)")

    # (c) Win rate vs human-edit
    ax = fig.add_subplot(gs[1, 1])
    ws = wins.sort_values("win_rate")
    bx = np.arange(len(ws))
    bar_colors = [COLORS[a] for a in ws["approach"]]
    ax.barh(bx, ws["win_rate"], color=bar_colors, alpha=0.7, edgecolor="black")
    for i, (_, r) in enumerate(ws.iterrows()):
        ax.errorbar(r["win_rate"], i,
                    xerr=[[r["win_rate"] - r["ci_low"]], [r["ci_high"] - r["win_rate"]]],
                    fmt="none", ecolor="black", capsize=3, lw=1.2)
        ax.text(r["win_rate"] + 0.012, i, f"{r['win_rate']*100:.1f}% (n={r['wins']}/{r['n']})",
                va="center", fontsize=8)
    ax.axvline(0.5, color="#888", ls=":", lw=1.0)
    ax.text(0.5, len(ws) - 0.4, "no-effect (50 %)", color="#888", ha="center", fontsize=8)
    ax.set_yticks(bx)
    ax.set_yticklabels(ws["label"], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(approach beats human post-edit on held-out sim)")
    ax.set_title("(c) Per-task win rate vs. human post-edit")

    # (d) Per-scenario means with bootstrap CIs
    ax = fig.add_subplot(gs[2, :])
    scens = sorted(per_scen["scenario"].unique())
    width = 0.2
    xpos = np.arange(len(scens))
    for k, a in enumerate(APPROACHES):
        means = []
        los = []
        his = []
        for s in scens:
            row = per_scen[(per_scen["scenario"] == s) & (per_scen["approach"] == a)]
            if row.empty:
                means.append(np.nan); los.append(np.nan); his.append(np.nan)
            else:
                r = row.iloc[0]
                means.append(r["mean"]); los.append(r["ci_low"]); his.append(r["ci_high"])
        means = np.array(means); los = np.array(los); his = np.array(his)
        offset = (k - 1.5) * width
        ax.bar(xpos + offset, means, width=width, color=COLORS[a],
               alpha=0.7, edgecolor="black", label=LABELS[a])
        ax.errorbar(xpos + offset, means,
                    yerr=[means - los, his - means], fmt="none",
                    ecolor="black", capsize=2, lw=1.0)
    ax.axhline(upper_bound, color="#444", ls="--", lw=0.9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(scens, rotation=15)
    ax.set_ylabel("Mean LUAR sim to held-out\n(95 % bootstrap CI)")
    ax.set_title("(d) Per-scenario means")
    ax.legend(loc="lower right", ncol=4, fontsize=8, frameon=False)
    ax.set_ylim(0.30, 0.78)

    fig.suptitle("Figure 10: Final assessment \u2014 LLM mimics vs human post-editing on the leakage-free metric",
                 fontsize=13, y=0.995)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fig10_final_assessment.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig10_final_assessment.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    paths = Paths()
    res_dir = REPO_ROOT / "results"
    fig_dir = REPO_ROOT / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)

    wide = build_wide_table(paths)
    print(f"wide table: {len(wide)} tasks ({wide['scenario'].value_counts().to_dict()})")

    fr = friedman_omnibus(wide)
    pd.DataFrame([fr]).to_csv(res_dir / "final_friedman.csv", index=False)
    print(f"Friedman: chi2={fr['chi2']:.2f}, k={fr['k']}, n={fr['n']}, p={fr['p']:.3g}")

    pairs = all_pairs(wide)
    pairs.to_csv(res_dir / "final_pairwise_tests.csv", index=False)
    print("\nPairwise tests:")
    print(pairs[["label_a", "label_b", "n", "mean_a", "mean_b", "g", "ci_low", "ci_high",
                  "p_perm", "p_wilcoxon", "p_perm_bh", "bh_reject"]].to_string(index=False))

    wins = win_vs_human(wide)
    wins.to_csv(res_dir / "final_win_vs_human.csv", index=False)
    print("\nWin rate vs human post-edit:")
    print(wins.to_string(index=False))

    per_scen = per_scenario(wide)
    per_scen.to_csv(res_dir / "final_per_scenario.csv", index=False)
    print(f"\nper-scenario: {len(per_scen)} rows")

    # Same-author upper bound (for the figure)
    from personal_style.embeddings import load_embeddings
    from personal_style.similarity import EmbeddingTable, cosine, participant_kind_indices
    z = load_embeddings(paths.processed_dir / "embeddings.npz")
    t = EmbeddingTable(z["pids"], z["kinds"], z["task_idx"], z["scenarios"], z["vecs"])
    pi = participant_kind_indices(t)
    sims = []
    for pid in sorted(set(t.pids.tolist())):
        ctrl = pi.get((pid, "control"), [])
        if len(ctrl) < 2:
            continue
        rows = sorted([(int(t.task_idx[i]), i) for i in ctrl], key=lambda x: x[0])
        sims.append(cosine(t.vecs[rows[0][1]], t.vecs[rows[1][1]]))
    upper_bound = float(np.mean(sims))
    print(f"\nupper bound (same-author control \u2194 control): {upper_bound:.3f}")

    figure_10(wide, pairs, wins, per_scen, upper_bound, fig_dir)
    print(f"\nfigures/fig10_final_assessment.{{pdf,png}} written")
    print(f"results/final_*.csv written")


if __name__ == "__main__":
    main()
