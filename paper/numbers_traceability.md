# Numbers traceability — RRPR 2026 paper

Every numerical claim in `paper/main.tex` resolves here to a specific row
in a specific CSV (or to a documented constant in code). This is
requirement [D.2] in `paper_requirements.md`.

## Constants of the design (not from a CSV)

| Constant | Value | Source |
|---|---|---|
| Participants | 81 | upstream release; `tests/test_data.py::test_n_participants` |
| Total task observations | 486 | upstream release; `tests/test_data.py::test_observation_counts` |
| Treatment tasks | 324 (= 81 × 4) | same |
| Control tasks | 162 (= 81 × 2) | same |
| LUAR-MUD revision SHA | `9204529...` | `src/personal_style/__init__.py::LUAR_REVISION` |
| Permutation-test draws | 10 000 | `scripts/03_run_hypothesis_tests.py::N_PERM` |
| Bootstrap CI resamples | 1 000 | `scripts/03_run_hypothesis_tests.py::N_BOOT` |
| BH-FDR `q` | 0.05 | `src/personal_style/stats.py::benjamini_hochberg` default |

## Reproduction (Section 4 / Table 1)

| Claim | Source row |
|---|---|
| Paper $g$ for H1a–H2c | hard-coded in `scripts/03_run_hypothesis_tests.py::PAPER_REFERENCE` (cross-checked against the source paper §6) |
| Ours $g$, CI, BH-FDR for H1a–H2c | `results/hypothesis_tests.csv` (one row per hypothesis) |
| Paper H3 rmcorr `r = 0.244 ± 0.076`, `p < .0001` | source paper §6.3 (Baumler et al.) |
| Ours H3 rmcorr `r = +0.244`, CI [+0.17, +0.32], `p = 3.6 × 10⁻⁹`, `n = 648` | `results/h3_rmcorr.csv` (single row) |

## Held-out v1 means (Section 5)

| Claim | Source |
|---|---|
| `sim_o4mini = 0.498` | `results/final_pairwise_tests.csv`, all rows where `a == "sim_o4_mini"`: column `mean_a` |
| `sim_human = 0.546` | same CSV: rows with `b == "sim_human_edit"` (column `mean_b`) and `a == "sim_human_edit"` rows |
| `sim_opus = 0.643` | same CSV: rows with `sim_opus`, column `mean_a` or `mean_b` |
| `sim_gpt55 = 0.649` | same CSV: rows with `sim_gpt55` |
| Same-author ceiling 0.701 | `scripts/10_final_assessment.py::main()` final print line; computed deterministically from `data/processed/embeddings.npz` |
| % gap closed: human 24 %, Opus 71 %, GPT-5.5 75 % | derived: `(0.546-0.498)/(0.701-0.498) ≈ 0.236`, `(0.643-0.498)/(0.701-0.498) ≈ 0.714`, `(0.649-0.498)/(0.701-0.498) ≈ 0.744` |
| Mimic-vs-human Hedges' $g$ leaky vs held-out (1.18 → 1.02) | leaky: `MIMIC_RESULTS_OPUS_4_7.md` §3 (n=85 result preserved in repo); held-out: `results/final_pairwise_tests.csv` row Human×Opus |
| Mean lexical overlap with demo: 0.07 (Opus), 0.09 (GPT-5.5) | spot-check at end of `MIMIC_RESULTS_OPUS_4_7.md` §1 and `MIMIC_RESULTS_GPT_5_5.md` §1 (numbers reproduced by `paper/scripts/check_paper.py`) |
| Word counts: 165 (Opus), 148 (GPT-5.5); 322/324 and 324/324 in 100-200 range | same |
| Opus outlier 0.56 lexical overlap | `MIMIC_RESULTS_OPUS_4_7.md` §1 |

## Statistics (Section 6 / Tables 2 & 3 / Figure 3)

| Claim | Source row |
|---|---|
| Friedman $\chi^2 = 419.3$, $p = 1.5 \times 10^{-90}$, $n = 324$ | `results/final_friedman.csv` (single row) |
| All 6 pairwise $g$, CIs, perm $p$, Wilcoxon $p$, BH-FDR | `results/final_pairwise_tests.csv` (6 rows) |
| Opus vs GPT-5.5 $g = -0.08$, $p = 0.14$ | `results/final_pairwise_tests.csv`, last row |
| Win-rate o4-mini 18.5 % (37/324, +46 ties), Opus 76.9 % (249/324), GPT-5.5 79.0 % (256/324) | `results/final_win_vs_human.csv` (3 rows) |
| Per-scenario means (8 scenarios × 4 approaches) | `results/final_per_scenario.csv` (32 rows) |

## Verification protocol

Run `python3 paper/scripts/check_paper.py` to grep for the headline
numbers in `paper/main.tex` and assert that they match the values in
`results/*.csv`. The script is deterministic and has zero external
dependencies beyond pandas and the Python standard library.
