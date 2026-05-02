"""Stage 05: produce Figures 3-8 (PDF + PNG). H3 (rmcorr) is also computed and saved."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pingouin as pg  # noqa: E402

from personal_style.data import Paths, load_processed  # noqa: E402
from personal_style.plots import figure3, figure4, figure5, figure6, figure7, figure8  # noqa: E402


def main() -> None:
    paths = Paths()
    fig_dir = REPO_ROOT / "figures"
    res_dir = REPO_ROOT / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = load_processed(paths)
    treat = pd.read_parquet(paths.processed_dir / "sim_treatment.parquet")
    pool_ctrl = np.load(paths.processed_dir / "pool_pairwise_control.npy")
    pool_edit = np.load(paths.processed_dir / "pool_pairwise_edited.npy")
    pool_llm = np.load(paths.processed_dir / "pool_pairwise_llm.npy")

    # ---- Figure 3 / 4 / 5 / 6 from similarity tables ----
    figure3(treat, fig_dir)
    figure4(treat, fig_dir)
    figure5(treat, fig_dir)
    figure6(pool_ctrl, pool_edit, pool_llm, fig_dir)

    # ---- Figure 7: perceived self-similarity per condition ----
    # Treatment: average likert_edited_capture + likert_edited_friend (perceived self-sim of post-edited)
    # and likert_original_capture + likert_original_friend (perceived self-sim of LLM draft).
    # Control: likert_edited_capture + likert_edited_friend (only "edited" Likerts asked) -- per repo README.
    tr = obs[obs["condition"] == "treatment"].copy()
    cr = obs[obs["condition"] == "control"].copy()

    perc = {
        "llm": tr[["likert_original_capture", "likert_original_friend"]].mean(axis=1).rename("llm"),
        "edited": tr[["likert_edited_capture", "likert_edited_friend"]].mean(axis=1).rename("edited"),
        "control": cr[["likert_edited_capture", "likert_edited_friend"]].mean(axis=1).rename("control"),
    }
    perc_df = pd.concat(
        [perc["llm"].reset_index(drop=True), perc["edited"].reset_index(drop=True), perc["control"].reset_index(drop=True)],
        axis=1,
    )
    figure7(perc_df, fig_dir)

    # ---- H3: repeated-measures correlation (Figure 8) ----
    # Paper: "After writing or post-editing, participants were asked questions pertaining
    # to how well each text captures their style which we average into a single score."
    # The two relevant Likert items are *_capture (does it capture your style) and
    # *_friend (would a friend recognize this as you), asked for both the LLM draft and
    # the post-edited text in the treatment condition. Each treatment row therefore
    # contributes TWO data points (LLM draft + edited) to the correlation.
    tr_sim = tr.merge(treat, on=["pid", "task_idx"], how="inner")
    h3_rows = []
    for _, r in tr_sim.iterrows():
        perc_llm = np.nanmean([r["likert_original_capture"], r["likert_original_friend"]])
        perc_edit = np.nanmean([r["likert_edited_capture"], r["likert_edited_friend"]])
        h3_rows.append({"pid": r["pid"], "luar": r["sim_llm_control_self"], "perceived_self": perc_llm, "kind": "llm"})
        h3_rows.append({"pid": r["pid"], "luar": r["sim_edited_control_self"], "perceived_self": perc_edit, "kind": "edited"})
    h3 = pd.DataFrame(h3_rows).dropna(subset=["luar", "perceived_self"])

    rmc = pg.rm_corr(data=h3, x="luar", y="perceived_self", subject="pid")
    rmcorr_r = float(rmc["r"].iloc[0])
    rmcorr_p = float(rmc["pval"].iloc[0])
    cols = list(rmc.columns)
    ci_col = "CI95%" if "CI95%" in cols else "CI95"
    rmcorr_ci = list(rmc[ci_col].iloc[0])
    rmcorr_dof = int(rmc["dof"].iloc[0])
    pd.DataFrame(
        [{"r": rmcorr_r, "p": rmcorr_p, "ci_low": rmcorr_ci[0], "ci_high": rmcorr_ci[1], "dof": rmcorr_dof, "n": len(h3),
          "paper_r": 0.244, "paper_se": 0.076, "paper_p": "<.0001"}]
    ).to_csv(res_dir / "h3_rmcorr.csv", index=False)

    merged = h3[["pid", "luar", "perceived_self"]].copy()
    merged["perc_scaled"] = (merged["perceived_self"] - 1) / 4
    figure8(merged, fig_dir, rmcorr_r, rmcorr_p)

    # ---- Figure 7 also gets a paired comparison: control vs. edited perceived self-similarity ----
    # (the paper's exploratory finding: g ≈ 0.01, p ≈ .9062). Save as CSV.
    from personal_style.stats import run_test
    pids_with_both = set(tr["pid"]).intersection(set(cr["pid"]))
    edited_per_pid = (
        tr.assign(perc=tr[["likert_edited_capture", "likert_edited_friend"]].mean(axis=1))
        .groupby("pid")["perc"].mean()
    )
    control_per_pid = (
        cr.assign(perc=cr[["likert_edited_capture", "likert_edited_friend"]].mean(axis=1))
        .groupby("pid")["perc"].mean()
    )
    paired_pids = sorted(pids_with_both)
    x = np.array([edited_per_pid.loc[p] for p in paired_pids if p in edited_per_pid.index])
    y = np.array([control_per_pid.loc[p] for p in paired_pids if p in control_per_pid.index])
    n = min(len(x), len(y))
    res = run_test("perceived_edited_vs_control", x[:n], y[:n], paired=True, n_perm=10000, n_boot=1000, seed=7777)
    pd.DataFrame(
        [{"name": res.name, "n": res.n, "g": res.g, "ci_low": res.ci_low, "ci_high": res.ci_high, "p": res.p,
          "paper_g": 0.01, "paper_ci_low": -0.17, "paper_ci_high": 0.19, "paper_p": 0.9062}]
    ).to_csv(res_dir / "perception_treatment_vs_control.csv", index=False)

    print("All figures written to", fig_dir)
    print(f"H3 rmcorr: r={rmcorr_r:+.3f}, p={rmcorr_p:.4g}, CI95={rmcorr_ci}, dof={rmcorr_dof}")


if __name__ == "__main__":
    main()
