"""Generate the LaTeX tables used in the RRPR 2026 paper.

Reads the canonical CSVs under `results/` and writes booktabs-style tables
to `paper/tables/`. Every numerical claim in the paper that appears in a
table is therefore traceable to a specific row of a specific CSV, per
requirement [D.2] in `paper/paper_requirements.md`.

Output files (overwritten on every run):
  paper/tables/tab_reproduction.tex   -- side-by-side H1/H2/H3 reproduction
  paper/tables/tab_pairwise.tex       -- all-pairs Hedges' g + BH-FDR
  paper/tables/tab_winrate.tex        -- per-task win rate vs human-edit
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
OUT = Path(__file__).resolve().parents[1] / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def _fmt_g(g: float) -> str:
    return f"{g:+.2f}".replace("-", r"$-$").replace("+", r"$+$")


def _fmt_ci(lo: float, hi: float) -> str:
    return rf"[{_fmt_g(lo)}, {_fmt_g(hi)}]"


def _fmt_p_perm(p: float) -> str:
    # n_perm = 10000 floor is 1e-4
    if p < 1.5e-4:
        return r"$<10^{-4}$"
    return f"{p:.3g}"


# ---------------------------------------------------------------------------
# Table 1: reproduction (paper number vs ours, BH-FDR survives)
# ---------------------------------------------------------------------------

def build_reproduction_table(df: pd.DataFrame) -> str:
    short = {
        "H1a (sim_edited_control_self vs sim_llm_control_self)": "H1a",
        "H1b (sim_edited_llm_other vs sim_llm_llm_other)": "H1b",
        "H1a' (sim_edited_control_other vs sim_edited_control_self)": r"H1a$'$",
        "H1c (sim_edited_control_self vs sim_edited_llm_other)": "H1c",
        "H2a (pool_edited vs pool_control)": "H2a",
        "H2b (pool_edited vs pool_llm)": "H2b",
        "H2c (sim_edited_edited_other vs sim_edited_control_self)": "H2c",
    }
    lines = [
        r"\begin{tabular}{l@{\quad}r@{\quad}r@{\quad}r@{\quad}r}",
        r"\toprule",
        r"Hyp. & Paper $g$ & Ours $g$ & 95\,\% CI (ours) & BH-FDR \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        h = short.get(r["name"])
        if h is None:
            continue
        lines.append(
            rf"{h} & {_fmt_g(r['paper_g'])} & {_fmt_g(r['g'])} & "
            rf"{_fmt_ci(r['ci_low'], r['ci_high'])} & "
            rf"{r'$\checkmark$' if r['bh_reject'] else r'$\times$'} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table 2: all-pairs paired permutation tests + BH-FDR + Wilcoxon p
# ---------------------------------------------------------------------------

def build_pairwise_table(df: pd.DataFrame) -> str:
    nice = {
        "sim_o4_mini": r"o4-mini",
        "sim_human_edit": r"Human",
        "sim_opus": r"Opus 4.7",
        "sim_gpt55": r"GPT-5.5",
    }
    lines = [
        r"\begin{tabular}{l@{\;}c@{\;}l r r l r r}",
        r"\toprule",
        r"\multicolumn{3}{c}{Pair (A vs.\ B)} & "
        r"$\bar{x}_A$ & $\bar{x}_B$ & "
        r"$g$ \,[95\,\% CI] & $p$ (perm) & $p$ (BH) \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        a = nice.get(r["a"], r["label_a"])
        b = nice.get(r["b"], r["label_b"])
        lines.append(
            rf"{a} & vs. & {b} & "
            rf"{r['mean_a']:.3f} & {r['mean_b']:.3f} & "
            rf"{_fmt_g(r['g'])}\,{_fmt_ci(r['ci_low'], r['ci_high'])} & "
            rf"{_fmt_p_perm(r['p_perm'])} & {_fmt_p_perm(r['p_perm_bh'])} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table 3: per-task win rate vs human post-edit (binomial CI)
# ---------------------------------------------------------------------------

def build_winrate_table(df: pd.DataFrame) -> str:
    nice = {
        "sim_o4_mini": "o4-mini draft",
        "sim_opus": "Claude Opus 4.7 mimic",
        "sim_gpt55": "GPT-5.5 mimic",
    }
    lines = [
        r"\begin{tabular}{l r r r}",
        r"\toprule",
        r"Approach (vs.\ human post-edit) & Wins/$n$ & Win rate (95\,\% CI) "
        r"& $p$ (vs.\ 0.5) \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        appr = nice.get(r["approach"], r["label"])
        wins_str = f"{int(r['wins'])}/{int(r['n'])}"
        if r["ties"] > 0:
            wins_str += rf" \,(+{int(r['ties'])} ties)"
        rate = f"{r['win_rate']*100:.1f}\\%"
        ci = f"[{r['ci_low']*100:.1f}\\%, {r['ci_high']*100:.1f}\\%]"
        p = r["p_binomial"]
        if p < 1e-12:
            p_str = rf"$<10^{{-12}}$"
        else:
            p_str = f"{p:.2g}"
        lines.append(rf"{appr} & {wins_str} & {rate}\,{ci} & {p_str} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    repro = pd.read_csv(RESULTS / "hypothesis_tests.csv")
    pairs = pd.read_csv(RESULTS / "final_pairwise_tests.csv")
    wins = pd.read_csv(RESULTS / "final_win_vs_human.csv")

    (OUT / "tab_reproduction.tex").write_text(build_reproduction_table(repro))
    (OUT / "tab_pairwise.tex").write_text(build_pairwise_table(pairs))
    (OUT / "tab_winrate.tex").write_text(build_winrate_table(wins))

    print("wrote:")
    for p in OUT.glob("*.tex"):
        print(f"  {p}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
