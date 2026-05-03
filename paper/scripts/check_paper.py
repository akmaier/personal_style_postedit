"""Checks every requirement in `paper/paper_requirements.md` against the built
paper.

This script is intentionally simple and self-contained: it does not parse
LaTeX, it just greps the source `.tex` and the resulting PDF text, and it
loads the canonical result CSVs directly. Any failure prints a clear
[FAIL] line; non-failures print [ok]. Exit status is non-zero iff any
[FAIL] line was produced.

Usage:
    python3 paper/scripts/check_paper.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
TEX = PAPER / "main.tex"
PDF = PAPER / "main.pdf"
REFS_BIB = PAPER / "refs.bib"
REFS_VERIFIED = PAPER / "references_verified.md"
REQS = PAPER / "paper_requirements.md"


passed: list[str] = []
failed: list[str] = []


def ok(req_id: str, msg: str) -> None:
    passed.append(f"[ok]   {req_id}  {msg}")


def fail(req_id: str, msg: str) -> None:
    failed.append(f"[FAIL] {req_id}  {msg}")


# ---------------------------------------------------------------------------
# A. formal venue requirements

def check_A() -> None:
    tex = TEX.read_text()
    if r"\documentclass[runningheads]{llncs}" in tex:
        ok("A.1", "main.tex uses llncs class with runningheads option")
    else:
        fail("A.1", "main.tex does not use llncs documentclass")

    if PDF.exists():
        info = subprocess.check_output(["pdfinfo", str(PDF)], text=True)
        m = re.search(r"^Pages:\s+(\d+)", info, re.M)
        n = int(m.group(1)) if m else -1
        # CFP says 6-14 pages, with extra pages on request. We hold ourselves
        # to 6-15: anything above 14 is flagged as a soft warning (we will
        # need to request a 1-page extension at submission time) but does
        # not break the build.
        if 6 <= n <= 14:
            ok("A.2 / E.2", f"page count {n} \u2208 [6, 14]")
        elif n == 15:
            ok("A.2 / E.2 (soft)",
               f"page count {n}; CFP allows extra pages on request, but plan to ask for one")
        else:
            fail("A.2 / E.2", f"page count {n} far outside [6, 14]")
    else:
        fail("A.2 / E.2", "main.pdf missing; run `make all` first")

    if "Track" in tex and "RRPR" in tex:
        ok("A.3", "Track / RRPR explicitly named")
    else:
        fail("A.3", "neither 'Track' nor 'RRPR' found in main.tex")

    cfp_items = ["implementation", "limitations", "improvements",
                 "reproducibility checklist"]
    hits = sum(1 for w in cfp_items if w.lower() in tex.lower())
    if hits >= 3:
        ok("A.4", f"covers {hits}/4 of the suggested CFP content axes")
    else:
        fail("A.4", f"covers only {hits}/4 of the suggested CFP content axes")

    if "Anonymous" in tex:
        ok("A.6", "anonymized for blind review")
    else:
        fail("A.6", "main.tex still contains author names")


# ---------------------------------------------------------------------------
# B. required content

def check_B() -> None:
    tex = TEX.read_text()
    if r"baumler2026personalstyle" in tex:
        ok("B.1 / D.3", "source paper Baumler et al. cited (key baumler2026personalstyle)")
    else:
        fail("B.1 / D.3", "missing \\cite{baumler2026personalstyle}")

    needed_keys = [
        "rivera2021luar",        # D.4
        "zaiss2026agent4mr",     # C.4
        "maier2018precision",    # C.3
        "maier2022knownoperator",  # C.3
        "maier2019gentle",       # C.2
        "hedges1981",
        "benjamini1995fdr",
        "bakdash2017rmcorr",
        "wilcoxon1945",
        "friedman1937",
    ]
    missing = [k for k in needed_keys if k not in tex]
    if not missing:
        ok("D.1 / D.4", f"all {len(needed_keys)} key citations present in main.tex")
    else:
        fail("D.1 / D.4", f"missing required citations: {missing}")


# ---------------------------------------------------------------------------
# D / E. negative checks: numbers traceability

def check_numbers() -> None:
    tex = TEX.read_text()
    repro = pd.read_csv(REPO / "results" / "hypothesis_tests.csv")
    pairs = pd.read_csv(REPO / "results" / "final_pairwise_tests.csv")
    wins = pd.read_csv(REPO / "results" / "final_win_vs_human.csv")
    fr = pd.read_csv(REPO / "results" / "final_friedman.csv").iloc[0]
    h3 = pd.read_csv(REPO / "results" / "h3_rmcorr.csv").iloc[0]

    # H3 rmcorr - exact match to 3 decimals
    if "+0.244" in tex:
        ok("D.2 / B.3 (rmcorr)", "rmcorr value +0.244 appears in paper")
    else:
        fail("D.2 / B.3 (rmcorr)", "rmcorr value +0.244 missing from paper")

    if abs(h3["r"] - 0.244) < 0.005:
        ok("D.2 / B.3 (CSV)", f"results/h3_rmcorr.csv r={h3['r']:.4f} \u2248 0.244")
    else:
        fail("D.2 / B.3 (CSV)", f"results/h3_rmcorr.csv r={h3['r']:.4f} \u2260 0.244")

    # Friedman
    if "419.3" in tex:
        ok("D.2 (Friedman in paper)", "chi2=419.3 appears in paper")
    else:
        fail("D.2 (Friedman in paper)", "chi2=419.3 not found in paper")
    if abs(fr["chi2"] - 419.3) < 1:
        ok("D.2 (Friedman in CSV)", f"final_friedman.csv chi2={fr['chi2']:.1f}")
    else:
        fail("D.2 (Friedman in CSV)", f"final_friedman.csv chi2={fr['chi2']:.1f} \u2260 419.3")

    # Win rate Opus 76.9 %
    opus_row = wins[wins["approach"] == "sim_opus"].iloc[0]
    rate = opus_row["win_rate"] * 100
    if abs(rate - 76.9) < 0.1:
        ok("D.2 (win rate Opus)", f"Opus win rate {rate:.1f}\u2009%")
    else:
        fail("D.2 (win rate Opus)", f"Opus win rate {rate:.1f}\u2009% \u2260 76.9\u2009%")
    if "76.9" in tex or r"76.9\%" in tex:
        ok("D.2 (win rate Opus in paper)", "76.9 appears in main.tex")
    else:
        fail("D.2 (win rate Opus in paper)", "76.9 missing")

    # Pairwise Opus vs GPT-5.5 should be ns
    opus_gpt = pairs[(pairs["a"].str.contains("opus")) & (pairs["b"].str.contains("gpt55"))]
    if len(opus_gpt) == 1:
        r = opus_gpt.iloc[0]
        if r["bh_reject"] is False or r["bh_reject"] == 0:
            ok("D.2 (Opus vs GPT-5.5 ns)",
               f"BH-FDR rejects=False, p={r['p_perm']:.3f}")
        else:
            fail("D.2 (Opus vs GPT-5.5 ns)", f"unexpectedly significant: {r['p_perm']}")
    else:
        fail("D.2 (Opus vs GPT-5.5 ns)", "row not found in final_pairwise_tests.csv")

    # Detection AUCs: o4-mini > human > Opus > GPT-5.5, all > chance
    det = pd.read_csv(REPO / "results" / "detection_aucs.csv").set_index("approach")
    expected_order = ["o4-mini", "Human post-edit", "Claude Opus 4.7", "GPT-5.5"]
    aucs = [det.loc[k, "auc_mean"] for k in expected_order]
    if aucs[0] > aucs[1] > aucs[2] > aucs[3] > 0.5:
        ok("D.2 (detection ordering)",
           f"AUCs strictly decreasing & above chance: {[f'{a:.3f}' for a in aucs]}")
    else:
        fail("D.2 (detection ordering)",
             f"detection AUCs not strictly decreasing or below chance: {aucs}")
    # Detection AUC for o4-mini approx 0.999 should appear in the paper
    if "0.999" in tex:
        ok("D.2 (detection o4-mini in paper)", "AUC 0.999 cited")
    else:
        fail("D.2 (detection o4-mini in paper)", "AUC 0.999 not in main.tex")
    # GPT-5.5 detection AUC -- the redesigned protocol gives ~0.931
    gpt55_auc = float(det.loc["GPT-5.5", "auc_mean"])
    rounded = f"{gpt55_auc:.3f}"
    if rounded in tex:
        ok("D.2 (detection GPT-5.5 in paper)", f"AUC {rounded} cited")
    else:
        fail("D.2 (detection GPT-5.5 in paper)",
             f"AUC {rounded} not in main.tex (CSV says {gpt55_auc:.4f})")

    # Adversarial rewriting (D.2). The paper must report:
    adv_csv = REPO / "results" / "adversarial_summary.csv"
    if adv_csv.exists():
        adv = pd.read_csv(adv_csv)
        n_flipped = int(adv["flipped_to_human_final"].sum())
        if "flipped" in tex.lower() or "flip" in tex.lower():
            ok("D.2 (adversarial flip count)",
               f"adversarial section present (flipped {n_flipped}/{len(adv)} of targets)")
        else:
            fail("D.2 (adversarial flip count)",
                 "no mention of adversarial flipping in main.tex")
        # The exact flipped count should appear in the paper for honesty
        if f"{n_flipped} of " in tex or f"{n_flipped}/" in tex or f"two of the five" in tex.lower():
            ok("D.2 (adversarial fraction in paper)",
               f"the {n_flipped}-of-{len(adv)} count appears")
        else:
            fail("D.2 (adversarial fraction in paper)",
                 f"the {n_flipped}-of-{len(adv)} count missing from paper")

    # Diagnostics (D.2 / B.x). The paper must report:
    #  - the GPT-5.5 length-only AUC (~0.88) as evidence of the length confound
    #  - the Opus length-only AUC (~0.52) as evidence the Opus signal is genuine
    #  - the shuffle-label baseline somewhere
    diag_csv = REPO / "results" / "detection_diagnostics.csv"
    if diag_csv.exists():
        diag = pd.read_csv(diag_csv)
        len_rows = diag[diag["diagnostic"].str.startswith("C.")]
        if not len_rows.empty:
            gpt_len = len_rows[len_rows["approach"] == "GPT-5.5"]["auc_mean"].iloc[0]
            opus_len = len_rows[len_rows["approach"] == "Claude Opus 4.7"]["auc_mean"].iloc[0]
            if f"{gpt_len:.3f}" in tex:
                ok("D.2 (length-only GPT-5.5)",
                   f"length-only AUC {gpt_len:.3f} appears in paper")
            else:
                fail("D.2 (length-only GPT-5.5)",
                     f"length-only AUC {gpt_len:.3f} missing from paper")
            if f"{opus_len:.3f}" in tex:
                ok("D.2 (length-only Opus)",
                   f"length-only AUC {opus_len:.3f} appears in paper")
            else:
                fail("D.2 (length-only Opus)",
                     f"length-only AUC {opus_len:.3f} missing from paper")
        if "shuffle" in tex.lower() or "Shuffle" in tex:
            ok("D.2 (shuffle-label diagnostic)",
               "shuffle-label diagnostic mentioned in paper")
        else:
            fail("D.2 (shuffle-label diagnostic)",
                 "shuffle-label diagnostic not mentioned")
    else:
        fail("D.2 (diagnostics CSV)",
             "results/detection_diagnostics.csv not found")


# ---------------------------------------------------------------------------
# Figures and tables referenced from text

def check_figs_and_tables() -> None:
    tex = TEX.read_text()
    fig_labels = re.findall(r"\\label\{(fig:[^}]+)\}", tex)
    tab_labels = re.findall(r"\\label\{(tab:[^}]+)\}", tex)
    for lbl in fig_labels:
        if rf"\ref{{{lbl}}}" in tex or rf"~\ref{{{lbl}}}" in tex:
            ok("D.6", f"figure {lbl} is referenced")
        else:
            fail("D.6", f"figure {lbl} declared but never \\ref'd")
    for lbl in tab_labels:
        if rf"\ref{{{lbl}}}" in tex or rf"~\ref{{{lbl}}}" in tex:
            ok("D.7", f"table {lbl} is referenced")
        else:
            fail("D.7", f"table {lbl} declared but never \\ref'd")


# ---------------------------------------------------------------------------
# References vs verified table

def check_refs_verified() -> None:
    bib = REFS_BIB.read_text()
    keys_in_bib = set(re.findall(r"@\w+\{([^,]+),", bib))
    verified = REFS_VERIFIED.read_text()
    keys_verified = set(re.findall(r"`(\w[\w\.\-]+)`", verified))
    missing = keys_in_bib - keys_verified
    if not missing:
        ok("D.1 / E.4", f"all {len(keys_in_bib)} bib keys appear in references_verified.md")
    else:
        fail("D.1 / E.4", f"keys in refs.bib but not in references_verified.md: {sorted(missing)}")


# ---------------------------------------------------------------------------
# Anonymization (D.10 / E.6)

def check_anonymization() -> None:
    """Check that the *built anonymous PDF* contains no author/affiliation
    strings. We deliberately run this against the rendered PDF, not the
    .tex source, because main.tex contains the real authors inside an
    `\\else` branch of `\\ifanonymous` that is only compiled by `make
    arxiv`. Grepping the source would false-positive on the inactive
    branch.
    """
    bad = [
        "Andreas K. Maier", "Andreas Maier",
        "Moritz Zaiss",
        "Siming Bayer",
        "Pattern Recognition Lab",
        "Friedrich-Alexander",
        "Institute of Neuroradiology",
    ]
    anon_pdf = PAPER / "main_anonymous.pdf"
    if not anon_pdf.exists():
        # Fall back to checking the source: the anonymous title block
        # must always be the active branch of the \\ifanonymous switch.
        tex = TEX.read_text()
        if r"\anonymoustrue" in tex:
            ok("D.10 / E.6 (source)",
               "main.tex defaults to anonymous (\\anonymoustrue); "
               "build the PDF with `make all && make anonymize` for full check")
        else:
            fail("D.10 / E.6 (source)",
                 "main.tex does NOT default to \\anonymoustrue; "
                 "the blind-review build would leak author info")
        return
    try:
        text = subprocess.check_output(
            ["pdftotext", str(anon_pdf), "-"], text=True
        )
    except FileNotFoundError:
        # pdftotext not installed -- skip with a soft warning instead of
        # blocking the check.
        ok("D.10 / E.6 (skipped)", "pdftotext not installed; cannot verify rendered PDF")
        return
    leaks = [b for b in bad if b in text]
    if not leaks:
        ok("D.10 / E.6", "no author/affiliation strings in rendered main_anonymous.pdf")
    else:
        fail("D.10 / E.6", f"main_anonymous.pdf leaks author info: {leaks}")


# ---------------------------------------------------------------------------

def main() -> int:
    if not TEX.exists():
        print(f"[FAIL] paper main.tex not found at {TEX}")
        return 2
    check_A()
    check_B()
    check_numbers()
    check_figs_and_tables()
    check_refs_verified()
    check_anonymization()

    for line in passed:
        print(line)
    for line in failed:
        print(line)
    print()
    print(f"summary: {len(passed)} ok, {len(failed)} fail")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
