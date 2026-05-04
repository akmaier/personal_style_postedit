# Paper plan — RRPR 2026 companion paper

## 1. Working title

> **Reproducing and Extending *Can You Make It Sound Like You?*: A
> Companion Paper on Style-Embedding Reproducibility and a
> Leakage-Free LLM-vs-Human Style-Mimicry Benchmark**

Alternative shorter forms (we'll pick during drafting):

- "On Reproducing Personal-Style Post-Editing Studies in NLP — A
  Pattern-Recognition-Workshop Companion to Baumler et al. (ACL 26)"
- "From Reproduction to Comparison: A Track-2 Companion Paper for the
  ACL 26 Personal-Style Post-Editing Study"

## 2. Audience and venue framing

The reader is an **ICPR/RRPR-style pattern-recognition** researcher,
typically working in image/signal/medical applications, who knows
embedding-based similarity metrics conceptually. They probably have not
read the Baumler et al. NLP paper. They care about: reproducibility
methodology, statistical rigor, hybrid evaluation protocols, and
hands-on RR practice.

This shapes the paper structure: heavy on protocol clarity, light on
dataset minutiae, and using **medical-imaging analogies** (anchored in
the cited Maier-group works) wherever they make a point land faster.

## 3. Story arc (one paragraph each)

1. **Hook.** Personal-style post-editing of LLM drafts is becoming a
   common workflow. Baumler et al. (ACL 26, arXiv:2604.24444) released a
   data-only study showing that humans *can* re-personalize an LLM
   draft, but only partially. Their dataset is public; their analysis
   code is not. We use this gap as a Track-2 RR exercise.
2. **What we did.** First, we re-implement and reproduce every
   preregistered hypothesis from the released logs, including the most
   independently checkable number (rmcorr `r = 0.244`). Second, we
   extend the protocol with a leakage-free comparison against two
   frontier LLMs (GPT-5.5 and Claude Opus 4.7) given a single
   unassisted writing sample, on the same 324 paired tasks.
3. **Why it matters for RR.** The whole pipeline is committed,
   tested, deterministic, and re-runs in <5 minutes on a CPU-only
   machine without any LLM API calls (mimic drafts are committed under
   `data/processed/mimics/`).
4. **What's new beyond reproduction.** A held-out v1 protocol that
   fixes a memorisation leak we initially had; an
   omnibus-then-pairwise statistical battery (Friedman + 6 paired
   permutation tests + Wilcoxon + BH-FDR + binomial win-rate); a
   per-scenario breakdown across the 8 writing-task types.
5. **Headline finding.** Both LLM mimics close ~71-75 % of the gap
   between an unconditioned o4-mini draft and the natural same-author
   ceiling on this LUAR metric, vs ~24 % for human post-editing.
   They beat the human post-edit on ~4 of 5 tasks (binomial p ≈ 1e-23).
6. **Caveats.** Self-experiment, n_demos=1, workflow-not-skill,
   style-fidelity-not-quality, the perception-vs-LUAR gap that the
   original paper itself reports.

## 4. Section outline (~12 pages target, 8 body + 4 refs)

| # | Section | Pages | Notes |
|---|---|---:|---|
| 1 | Introduction | 1.0 | Hook, what we contribute, the 3 key numbers |
| 2 | Background | 1.5 | The original paper in 2 paragraphs; LUAR-MUD in one paragraph; pattern-recognition framing of style embeddings as known-operator-style hybrid models |
| 3 | Reproduction methodology | 1.5 | Pipeline diagram (Fig. 1); LUAR pinning; permutation/bootstrap protocol; what's deterministic and what isn't |
| 4 | Reproduction results | 1.5 | Side-by-side table for H1a, H1a', H1b, H1c, H2a, H2b, H2c, H3 (Table 1); reproduction of Figs. 3-8 (consolidated as our Fig. 2 montage) |
| 5 | Beyond reproduction: held-out LLM mimic | 2.0 | Held-out v1 protocol; cache-key salt; sub-agent generation; 4-way comparison (Fig. 3 = our Fig. 10) |
| 6 | Statistical assessment | 1.0 | Friedman omnibus; all 6 paired permutation tests with BH-FDR (Table 2); Wilcoxon agreement; per-task binomial win-rate (Fig. 4 = our fig10 panel c) |
| 7 | Discussion | 1.0 | Caveats; perception-vs-LUAR gap; what an external 3rd-generator confirmation would need |
| 8 | Reproducibility checklist | 0.5 | The 11-item checklist matching our `FINAL_ASSESSMENT.md` |
| - | References | 1.0 | ~25-30 entries |

## 5. Figures and tables

We re-use figures we already have in `figures/`. None of them were made
for this paper specifically, but they were made by the committed
pipeline that this paper is *about*, which is exactly what an RR
companion paper should do.

| ID | Source file | Caption (draft) |
|---|---|---|
| Fig. 1 | new (draw with TikZ in `paper/`) | Pipeline overview: logs → tidy → LUAR → similarity → tests → figures |
| Fig. 2 | montage of `figures/fig3_before_after.png` … `fig8_rmcorr.png` | Reproduction of paper Figs. 3-8 from the released logs |
| Fig. 3 | `figures/fig9_llm_mimic_vs_human.png` | Held-out 3-way comparison (one panel per generator) |
| Fig. 4 | `figures/fig10_final_assessment.png` | Final 4-way assessment: distributions, forest plot, win-rate, per-scenario |
| Tab. 1 | derived from `results/hypothesis_tests.csv` + paper numbers | Side-by-side reproduction of H1-H3 |
| Tab. 2 | derived from `results/final_pairwise_tests.csv` | All 6 pairwise paired tests with Hedges *g*, 95% CI, BH-FDR |
| Tab. 3 | derived from `results/final_win_vs_human.csv` | Per-task win-rate vs human post-edit |

## 6. Citation list (target — verified during planning)

This is the seed bibliography we'll use. Every entry was *read or
bibliographically resolved* during this planning turn (see
`paper/references_verified.md` after the build step).

### 6.1 The source paper (cite often)

- **Baumler, Bao, Nghiem, Yang, Carpuat, Daumé III.** *Can You Make It
  Sound Like You? Post-Editing LLM-Generated Text for Personal Style.*
  arXiv:2604.24444. To appear at ACL 26.

### 6.2 Style embeddings and LUAR

- **Rivera-Soto, Miano, Ordonez, Chen, Khan, Bishop, Andrews.** *Learning
  Universal Authorship Representations.* EMNLP 2021. **Cite at [B.4]
  whenever LUAR is mentioned.**
- (Optional) **Wegmann, Schraagen, Nguyen.** *Same Author or Just Same
  Topic? Towards Content-Independent Style Representations.* RepL4NLP
  2022 — only if we discuss alternative style models.

### 6.3 Style-inspiration anchor papers (Maier / Zaiss group)

- **Karpathy.** *autoresearch.* GitHub repository (2026).
  *Cite alongside [C.4]: fixed-budget GPT-style pretraining loop with
  `val_bpb` as the automatic objective; useful contrast to our
  inference-time, closed-weight setting.*
- **Zaiss, Aly, Endres, Dornstetter, Weinmüller, Maier.** *Agentic MR
  sequence development: leveraging LLMs with MR skills for automatic
  physics-informed sequence development.* arXiv:2604.13282 (2026).
  *Cite at [C.4]: LLM-vs-human evaluation framing; Table 1 is a model
  for our own LLM-vs-human cost/quality table.*
- **Maier, Schebesch, Syben, Würfl, Steidl, Choi, Fahrig.** *Precision
  Learning: Towards Use of Known Operators in Neural Networks.*
  arXiv:1712.00374 / ICPR 2018, pp. 183-188. *Cite at [C.3]: framing the
  LLM mimic prompt as a hybrid / known-operator-style approach.*
- **Maier, Köstler, Heisig, Krauß, Yang.** *Known Operator Learning and
  Hybrid Machine Learning in Medical Imaging — A Review.* arXiv:2108.04543
  / Prog. Biomed. Eng. 4 022002 (2022). *Cite at [C.3]: hybrid-models
  review for the same point.*
- **Maier, Syben, Lasser, Riess.** *A gentle introduction to deep
  learning in medical image processing.* Z. Med. Phys. 29(2):86-101
  (2019). DOI 10.1016/j.zemedi.2018.12.003. *Cite at [C.2]: tone /
  structure anchor; "didactic but technical" model for our writing.*
  **Note:** the user supplied the ScienceDirect link; this is the
  paper that link points to (it is *not* a Pattern Recognition Letters
  paper, despite the URL pattern).

### 6.4 Statistical methodology (cite once each, in §3 or §6)

- **Hedges (1981)** for the small-sample correction.
- **Benjamini & Hochberg (1995)** for FDR.
- **Bakdash & Marusich (2017)** for repeated-measures correlation
  (rmcorr; needed for the H3 reproduction).
- **Wilcoxon (1945)** for the signed-rank test.
- **Friedman (1937)** for the omnibus test.

### 6.5 Software & datasets

- **HuggingFace Transformers** (Wolf et al., EMNLP demo 2020).
- **PyTorch** (Paszke et al., NeurIPS 2019).
- **pingouin** (Vallat, JOSS 2018).
- **scipy.stats** (Virtanen et al., Nat. Methods 2020).

We will resolve full BibTeX entries during the drafting turn and add
each one to `paper/references_verified.md` with its URL/DOI and the date
we verified it.

## 7. Build plan (sequential turns)

1. **This turn (planning):** template extracted, `paper_requirements.md`
   and `paper_plan.md` written. **No paper text yet.**
2. **Drafting turn:** `paper/main.tex`, `paper/refs.bib`, populated bib
   with verified entries, all 4 figures dropped in, 3 tables generated
   from CSVs by a small `paper/scripts/build_tables.py`.
3. **Self-review turn:** generate `paper/paper_check.md` running through
   every requirement-doc item with [ok] / [issue] / [fix]. Implement
   the fixes. Build and verify page count.
4. **Anonymization turn (optional, for blind submission):** remove
   author info, generate `paper/main_anonymous.pdf`, strip metadata
   with `exiftool -all=`.
5. **Camera-ready turn (after notification):** add author info and
   ORCIDs, regenerate PDF, build `paper/main_cameraready.pdf`.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Hallucinated reference (the canonical LLM-paper failure) | `references_verified.md` table with URL/DOI + date, generated *during* the drafting turn, not after. Every `\cite` must point at an entry there. |
| Numerical drift between text and CSVs | `paper/scripts/check_numbers.sh` greps the 3 headline numbers in main.tex and asserts they match exact strings in the result CSVs. CI-friendly. |
| Page overflow | Plan target 12 pages out of 14 max — 2 pages of slack. If overflow during drafting, cut §7 reproducibility-checklist appendix into a `paper/checklist.md` linked from the paper. |
| LNCS template breakage by overzealous editing | Keep `paper/template/` *unmodified*. Our `main.tex` will `\documentclass{llncs}` from the local `llncs.cls` and we will not edit the cls file under any circumstance. |
| Wrong workshop scope (Track 1 vs Track 2) | We are clearly Track 2; §1 of the paper says so explicitly; §B.3 of `paper_requirements.md` enforces it. |
| User's URL points to a different paper than they remember (already happened once during planning) | Every link in `paper_plan.md` and `paper_requirements.md` was re-resolved by us at planning time, with the correction noted in `paper/README.md`. We will not silently rebrand a paper. |
| Author names leak in metadata | Anonymization turn (§7.4) strips PDF metadata via `exiftool -all=` and grep-checks for known affiliation strings before submission. |
| The "1 in 324" Opus draft outlier (verbatim copy) | Disclose it in the limitations section, also disclose that removing it changes the headline `g` by less than 0.01. |

## 9. What this plan deliberately does NOT do

- **Does not write any paper text.** Drafting is a separate turn.
- **Does not commit `main.tex` or `refs.bib`.** Those land in turn 2.
- **Does not change the existing reproducibility code or figures.** The
  whole point of an RR companion paper is that the underlying pipeline
  is **fixed and frozen** by the time the paper is written. The paper
  *describes* what we already have on `main`; it does not modify it.
