# Reproduction Plan: *Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style*

**Paper:** Baumler, Bao, Nghiem, Yang, Carpuat, Daumé III (UMD), to appear at ACL 26.
**arXiv:** [2604.24444](https://arxiv.org/abs/2604.24444)
**Upstream data repo:** https://github.com/ctbaumler/personal_style_postedit
**This fork:** `akmaier/personal_style_postedit` (branch `cursor/reproduction-plan-85a9`)

This document is a concrete, executable plan to reproduce **all main figures and statistical
results** of the paper from the writing logs that ship with the upstream repository
(`logs/*.json`, n = 81 participants, 6 tasks each).

The paper does **not** publish analysis code – only the raw study logs – so reproduction
means re-implementing the analysis pipeline end-to-end from scratch, faithful to the
paper's described methodology.

---

## 1. What "reproduce" means here

Goal: regenerate every published quantitative result and every figure that depends on the
released data, in a self-contained, deterministic, scripted pipeline that lives entirely
in this repo.

Concretely the deliverables are:

| Paper artefact | Source / data needed | Output file in this repo |
|---|---|---|
| **Fig. 3** – sim(LLM, treatment) and sim(control, treatment) before vs. after post-edit | LUAR(LLM draft, edited), LUAR(control, edited) | `figures/fig3_before_after.{pdf,png}` |
| **Fig. 4** – treatment text similarity to control vs. LLM, before & after | same as Fig. 3 | `figures/fig4_treatment_vs_control_llm.{pdf,png}` |
| **Fig. 5** – self vs. other-participant similarity (control & post-edit pools) | LUAR cross-participant matrices | `figures/fig5_self_vs_other.{pdf,png}` |
| **Fig. 6** – within-pool homogeneity (LLM / post-edit / control) | pairwise LUAR similarities | `figures/fig6_homogeneity.{pdf,png}` |
| **Fig. 7** – perceived self-similarity by condition | task-level Likert items | `figures/fig7_perceived.{pdf,png}` |
| **Fig. 8** – perceived vs. LUAR similarity (rmcorr) | LUAR sim + Likert | `figures/fig8_rmcorr.{pdf,png}` |
| **H1a, H1a′, H1b, H1c, H2a, H2b, H2c, H3** test statistics (perm. p, Hedges g + 95 % CI) | as above | `results/hypothesis_tests.csv`, `results/hypothesis_tests.md` |
| **Pangram H1b replication** (g ≈ −0.45) | Pangram AI-detector scores on each text | `results/pangram_tests.csv` |
| **Appendix B figures / tables** that depend only on logs (B.1, B.4, B.5, B.7.1–B.7.3) | logs + LUAR + simple stylometry | `figures/appendix_*.{pdf,png}`, `results/appendix_*.csv` |
| Final write-up with side-by-side comparisons to paper-reported values | all of the above | `REPRODUCTION_REPORT.md` |

Things we will **not** be able to reproduce exactly:

- The original LLM drafts: they are already stored in each log under
  `responses[i].model_generation`, generated with `gpt-o4-mini`. We re-use those exact
  drafts; we do not re-query the API. So § 5.1 of the paper does not need re-running.
- §§ 6.4 / B.7 *qualitative* coding: the paper's open-coding of ~20 % of treatment
  documents was done by humans. We will reproduce only the quantitative analyses around
  it (e.g., word-error-rate distribution, edit density).
- Pangram scores: Pangram is a paid hosted API. We will reproduce H1b through Pangram
  only if a `PANGRAM_API_KEY` is provided as a secret; otherwise that single test is
  skipped and explicitly marked as "not reproduced (no API key)" in the report.

---

## 2. Repository layout we will build

```
.
├── REPRODUCTION_PLAN.md          # this file
├── REPRODUCTION_REPORT.md        # written at the end; paper vs. ours, side-by-side
├── README.md                     # existing data README, lightly updated to point at the plan
├── logs/                         # existing 81 JSON logs, untouched
├── pyproject.toml                # pinned deps
├── requirements.txt              # mirror of pyproject for pip-only users
├── Makefile                      # `make all` runs the whole pipeline end-to-end
├── src/personal_style/
│   ├── __init__.py
│   ├── data.py                   # loads logs/*.json -> tidy long-format DataFrames
│   ├── embeddings.py             # LUAR-MUD wrapper (HF model rrivera1849/LUAR-MUD)
│   ├── similarity.py             # cosine sim helpers + caching layer
│   ├── stats.py                  # permutation tests, Hedges' g + bootstrap CI, BH-FDR, rmcorr
│   ├── pangram.py                # optional Pangram API client (skipped if no key)
│   ├── plots.py                  # all figure-generating functions
│   └── cli.py                    # `python -m personal_style.cli <stage>`
├── scripts/
│   ├── 00_build_dataframe.py     # logs/ -> data/processed/observations.parquet
│   ├── 01_compute_embeddings.py  # -> data/processed/embeddings.npz
│   ├── 02_compute_similarities.py# -> data/processed/sim_*.parquet
│   ├── 03_run_hypothesis_tests.py# -> results/*.csv
│   ├── 04_pangram_scores.py      # optional
│   ├── 05_make_figures.py        # -> figures/*.{pdf,png}
│   └── 06_build_report.py        # -> REPRODUCTION_REPORT.md
├── data/processed/               # gitignored, regenerated by scripts
├── figures/                      # committed, small PDFs/PNGs
├── results/                      # committed CSV/MD result tables
└── tests/
    ├── test_data.py              # schema + invariants on logs (e.g. n=81, 6 tasks)
    ├── test_stats.py             # permutation test & Hedges g sanity checks
    └── test_smoke_pipeline.py    # ~3-participant subset runs end-to-end
```

A `Makefile` will chain everything:

```
make data        # 00
make embeddings  # 01 (cached, GPU-friendly)
make sims        # 02
make tests       # 03
make figures     # 05
make report      # 06
make all         # everything above + lint + tests
```

---

## 3. Methodological mapping (paper → code)

### 3.1 Data model (§ 4 + repo README)

For each of the 81 `logs/*.json` files we extract one row per (participant × task), giving
**486 task observations** (81 × 6) split as 324 treatment + 162 control. Tidy columns:

```
pid, task_idx (0..5), scenario, condition (treatment|control),
details (str), llm_draft (str), final_text (str), n_edits, edit_duration_s,
likert_original_useable, likert_original_capture, likert_original_friend,
likert_original_content, likert_edited_useable, likert_edited_capture,
likert_edited_friend, likert_written_before, likert_voice_important,
edit_type (multi-label list)
```

Demographic / survey columns (`gender`, `race`, `loe`, `age`, `*_table`,
`pre_*`, `post_*`, mid-survey TLX) live in a separate participant-level table joined on
`pid`. Validated by `tests/test_data.py` (n = 81; conditions sum to 4 per participant; etc.).

### 3.2 Style embeddings (§ 5.2)

- **Model:** `rrivera1849/LUAR-MUD` from Hugging Face (`AutoModel.from_pretrained(..., trust_remote_code=True)`),
  exactly the LUAR variant the paper says it picked in § B.2 because it best identified
  authors on the control data.
- **Episodes:** the paper computes one LUAR vector per text. We follow the LUAR-MUD card:
  treat each text as a single "episode" with a single document, max_length 512, mean-pool
  via the model's own forward (`out` shape `[batch, 512]`).
- **Output cache:** `data/processed/embeddings.npz` with keys `pids`, `kinds`
  (`control` | `llm` | `edited`), `task_idx`, `scenario`, `vecs (N×512)`. Re-run is
  idempotent; embeddings cached on disk so the rest of the pipeline is CPU-only.
- **Sanity check:** reproduce the authorship-identification result of § B.2 on the 162
  control texts (leave-one-task-out: rank a participant's other 2 control texts among
  all other participants' control texts; report top-1 / MRR). If our number is in the
  same ballpark as the paper's reported authorship accuracy, we trust the embedding step.

### 3.3 Cosine-similarity tables (§ 6.1, § 6.2)

For each treatment observation we compute cosines between:

1. `edited` ↔ `llm`           (its own LLM draft)
2. `edited` ↔ `control_self`  (mean of the same participant's 2 control texts)
3. `edited` ↔ `control_other` (mean over each *other* participant's control texts)
4. `edited` ↔ `edited_other`  (mean over each *other* participant's edited texts)
5. `llm`    ↔ `control_self`  (the "before post-edit" baseline used in Figs. 3–4)
6. `llm`    ↔ `llm_other`     (within-LLM homogeneity for Fig. 6)
7. `control_self` ↔ `control_other`     (within-control homogeneity for Fig. 6)
8. `edited_self`  ↔ `edited_other`      (within-edited homogeneity for Fig. 6)

These exactly mirror Figures 3–6. Stored as long-format Parquet so plotting is trivial.

### 3.4 Statistical tests (§ 6, § A.2)

Implemented in `src/personal_style/stats.py`:

- **Permutation test**: paired or unpaired depending on the hypothesis, `n_permutations =
  10_000`. The paper reports `p = .0002`, the Monte-Carlo floor for 5 000 permutations of
  a paired test (`(0+1)/(5000+1)`). We use 10 000 to match or exceed their resolution.
- **Hedges' g**: standard small-sample-corrected Cohen's d (`g = d * (1 - 3/(4n-9))`).
  95 % CI from 1 000 bootstrap resamples (paper's own protocol).
- **rmcorr** for H3: `pingouin.rm_corr` (or a from-scratch implementation —
  `lme4`-style ANCOVA per participant — so we don't depend on R).
- **Benjamini–Hochberg FDR** at q = 0.05 across the preregistered set of tests; raw
  p-values are reported alongside, exactly as the paper does.

Each hypothesis becomes one row in `results/hypothesis_tests.csv` with columns:
`hypothesis, n_pairs, statistic, p_raw, p_bh, hedges_g, ci_low, ci_high, paper_g, paper_ci_low, paper_ci_high, agreement`.

### 3.5 Pangram H1b (§ 6.1, optional)

`src/personal_style/pangram.py` calls Pangram's REST API (`POST /predict` with API key in
`PANGRAM_API_KEY`) for every `llm` and `edited` text, caches scores on disk, and runs the
same paired permutation + Hedges' g machinery. If the secret is missing the script logs
a single `SKIPPED: no PANGRAM_API_KEY` line and the report records "not reproduced".

### 3.6 Plots (Figures 3–8 + Appendix)

`src/personal_style/plots.py` produces matplotlib figures that visually match the paper's
KDE / strip / boxplot / scatter style. Each plot function takes a tidy DataFrame so the
plotting code stays small. All figures are written to both PDF (vector) and PNG (raster
for the report and walkthrough video).

### 3.7 Appendix items we will also reproduce

- **B.1** – test for stylistic personalisation in `details` text (LUAR sim of `details`
  to participant's own control vs. others' control).
- **B.4** – post-editing effort by demographic groups (edit count, edit duration,
  word-error-rate) crossed with `gender`, `race`, `loe`, `age` (with optional ones
  collapsed to "given / not given").
- **B.5** – H1 effects split by per-task `voice-important` Likert.
- **B.7.1** – nonstandard / less-acceptable edits effect, using a simple LanguageTool or
  spell-check pass to flag potential typos / non-standard punctuation.
- **B.7.2** – frequency of em-dashes / known LLM lexicon ("dive", "delve", "tapestry",
  …) before vs. after edits.
- **B.7.3** – edit *density* (mean edit position dispersion / WER) vs. ΔLUAR similarity.

Anything that is purely qualitative (open-coded examples in § 6.4) is summarised as
counts only.

---

## 4. Engineering plan

### 4.1 Environment & dependencies

`pyproject.toml` will pin (latest stable as of 2026):

- `python>=3.11`
- `numpy`, `pandas`, `pyarrow`
- `scipy`, `statsmodels`, `pingouin` (rmcorr, BH-FDR)
- `torch`, `transformers`, `accelerate`, `huggingface_hub` (for LUAR)
- `matplotlib`, `seaborn`
- `tqdm`, `python-dotenv`, `requests` (Pangram)
- dev: `pytest`, `ruff`, `black`, `mypy`

Install: `pip install -e ".[dev]"`. A `requirements.txt` is generated from `pyproject.toml`
so users without `pip install -e` still work.

### 4.2 Determinism

- All numpy / torch RNGs seeded from a single `SEED=20260101` constant.
- Permutation tests use a `numpy.random.Generator(PCG64(seed))` per test, seeded from
  the test name.
- Embedding step is deterministic on a given CUDA device but tolerates ±1e-6 on CPU.
- The pipeline is idempotent: re-running `make all` on a fresh checkout reproduces every
  artefact bit-for-bit, modulo the embedding numerics noted above.

### 4.3 Compute requirements

LUAR-MUD is 82 M params, FP32. Embedding ~1 100 short documents (≈ 162 control + 324
edited + 324 LLM + a few aux) is well under a minute on any modern GPU and a few minutes
on CPU. A single CPU run of the whole pipeline finishes in roughly a few minutes —
fits easily into a Cursor Cloud agent VM.

### 4.4 CI

A small `.github/workflows/reproduce.yml`:

1. Checkout, set up Python 3.11, cache `~/.cache/huggingface`.
2. `pip install -e ".[dev]"`
3. `pytest -q` (smoke tests + stats unit tests, *no* full embedding run).
4. On `main` only: run `make all` on the full data, upload `figures/` and
   `results/` as workflow artefacts. Expected wall time < 10 min.

### 4.5 Validation strategy

- Each hypothesis row carries our number **and** the paper's number. The report
  highlights any |Δg| > 0.1 or sign disagreement so reviewers can spot regressions.
- We also reproduce one simple ground-truth check: re-embed the same text twice and
  assert the cosine similarity is `1.0 ± 1e-6`.
- `tests/test_smoke_pipeline.py` exercises the full pipeline on 3 participants so the
  whole chain stays green in CI without downloading the full LUAR weights every time
  (it monkey-patches the embedding model with a tiny random projection).

---

## 5. Step-by-step execution order

| Step | Branch | Output |
|---|---|---|
| 1. Land this plan | `cursor/reproduction-plan-85a9` (this PR) | `REPRODUCTION_PLAN.md` |
| 2. Scaffolding + `data.py` + tests | `cursor/scaffold-…` | `src/`, `tests/`, parquet of observations |
| 3. LUAR embedding pipeline + auth-ID sanity check | `cursor/embeddings-…` | `embeddings.npz`, B.2 number |
| 4. Similarity tables + § 6.1 tests (H1*) + Fig. 3 / 4 | `cursor/h1-…` | first real figures |
| 5. Homogeneity tests (H2*) + Fig. 5 / 6 | `cursor/h2-…` | |
| 6. Perception (H3) + Fig. 7 / 8 | `cursor/h3-…` | |
| 7. Pangram (gated by secret) | `cursor/pangram-…` | optional CSV |
| 8. Appendix B.1 / B.4 / B.5 / B.7.* | `cursor/appendix-…` | |
| 9. `REPRODUCTION_REPORT.md` and CI | `cursor/report-…` | side-by-side report |

Each step is its own short PR off `main`, with a passing test suite, so review stays
incremental and the full reproduction can be re-run after any single step.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| LUAR-MUD weights move / vanish from HF | Pin a specific revision (`AutoModel.from_pretrained(..., revision="<sha>")`) and document the SHA in the report. |
| Numerical mismatch with paper (different LUAR version, no GPU determinism) | Report Hedges' g to two decimals and tolerate ±0.05 vs. paper; flag bigger gaps. |
| Pangram API unavailable | Step 7 is gated; report explicitly marks H1b-Pangram as "not reproduced". |
| Permutation-test floor (`p=.0002`) not matching at lower n_perm | Use ≥ 5 000 permutations for every reported test; document `n_perm` in CSV. |
| rmcorr implementation drift between `pingouin` versions | Pin `pingouin>=0.5,<0.6` and ship a 50-line numpy reference implementation in `tests/`. |
| Paper expects FDR-corrected significance — easy to forget | `stats.py` always returns both `p_raw` and `p_bh`; tests assert that the published preregistered set is what gets BH-corrected. |

---

## 7. What's in this PR

Just this plan file. No code yet. Subsequent PRs will implement steps 2–9.
