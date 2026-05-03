# Final assessment — LLM mimics vs. human post-editing

**Headline result:** *On the leakage-free held-out LUAR-MUD metric (n = 324
treatment tasks), both Claude Opus 4.7 and GPT-5.5 produce drafts that are
statistically far closer to the participant's own writing style than the
participant's own human post-edits of an unconditioned o4-mini draft. The
gap is replicated, large, and survives Benjamini-Hochberg correction over
all 6 pairwise tests. Opus 4.7 and GPT-5.5 are statistically indistinguishable
from each other.*

## 1. Setup recap

- 81 participants × 4 treatment tasks = 324 paired observations.
- Each treatment task is a single LUAR-MUD cosine similarity to the
  participant's **held-out** control text (the unassisted writing the
  participant did on a separate task in their session, never shown to any
  generator).
- All four approaches share the same held-out yardstick:
  - `sim_o4_mini`     — original o4-mini draft (no style sample shown)
  - `sim_human_edit`  — participant's post-edit of that o4-mini draft
  - `sim_opus`        — Claude Opus 4.7 mimic (one style sample shown)
  - `sim_gpt55`       — GPT-5.5 mimic         (one style sample shown)
- Cache-key salt `held_out_protocol_v1` makes any leak between protocols
  impossible at the file level; see PR #6.

## 2. Means and the natural ceiling

| Approach | Mean LUAR cos to held-out | % of gap to ceiling closed |
|---|---:|---:|
| o4-mini draft (no style sample) | **0.498** | 0 % (baseline) |
| Human post-edit | **0.546** | **24 %** |
| Claude Opus 4.7 mimic | **0.643** | **71 %** |
| GPT-5.5 mimic | **0.649** | **75 %** |
| *Same-author control \u2194 control upper bound* | *0.701* | *100 %* |

The same-author upper bound is the LUAR cosine you get by comparing two
unassisted texts written by the *same person* in their natural voice. It is
the sensible ceiling on this metric: anything systematically above it would
suggest either a leak or LUAR finding a non-style signal.

## 3. Statistical analysis

### 3.1 Friedman omnibus across all 4 approaches

\u03c7\u00b2(3, n=324) = **419.3**, p = **1.5 \u00d7 10\u207b\u2079\u2070**.

Strong, unambiguous rejection of "the four approaches yield interchangeable
held-out similarity". Posthoc tests follow.

### 3.2 All-pairs paired permutation tests (n_perm = 10 000) + Wilcoxon, with BH-FDR

| Pair | mean A | mean B | Hedges' *g* | 95 % CI | p (perm) | p (Wilcoxon) | p (BH) | BH q=0.05 |
|---|---:|---:|---:|---|---:|---:|---:|:---:|
| o4-mini vs Human post-edit | 0.498 | 0.546 | -0.48 | [-0.55, -0.41] | 1e-4 | 5.5e-37 | 1.2e-4 | \u2713 |
| o4-mini vs Claude Opus 4.7 | 0.498 | 0.643 | -1.57 | [-1.75, -1.39] | 1e-4 | 1.5e-45 | 1.2e-4 | \u2713 |
| o4-mini vs GPT-5.5 | 0.498 | 0.649 | -1.61 | [-1.80, -1.44] | 1e-4 | 3.9e-48 | 1.2e-4 | \u2713 |
| Human post-edit vs Claude Opus 4.7 | 0.546 | 0.643 | -1.02 | [-1.19, -0.85] | 1e-4 | 3.3e-30 | 1.2e-4 | \u2713 |
| Human post-edit vs GPT-5.5 | 0.546 | 0.649 | -1.07 | [-1.24, -0.93] | 1e-4 | 2.7e-33 | 1.2e-4 | \u2713 |
| Claude Opus 4.7 vs GPT-5.5 | 0.643 | 0.649 | -0.08 | [-0.18, +0.02] | 0.136 | 0.20 | 0.136 | \u2717 |

**5 of 6 pairs survive BH-FDR at q = 0.05.** The only non-significant pair is
Opus 4.7 vs GPT-5.5 — they are practically tied. The agreement between the
permutation test and the non-parametric Wilcoxon signed-rank test on every
pair is reassuring: the result is not a fragile artefact of one test choice.

### 3.3 Per-task win rate vs. the human-edit threshold

For each of the 324 tasks, did the approach's held-out similarity exceed the
human's? This treats human post-editing as the threshold the user identified.

| Approach | Wins / n | Win rate | 95 % binomial CI | p (vs 0.5) |
|---|---:|---:|---|---:|
| o4-mini draft | 37 / 324 (+46 ties) | 18.5 % | [14.4 %, 23.2 %] | 1.1e-31 |
| Claude Opus 4.7 | 249 / 324 | 76.9 % | [71.9 %, 81.3 %] | 5.9e-23 |
| GPT-5.5 | 256 / 324 | **79.0 %** | [74.2 %, 83.3 %] | 8.5e-27 |

**The o4-mini draft beats the human on only ~1 in 5 tasks; both frontier
LLMs beat the human on roughly 4 in 5.** Both LLM win rates differ from
50 % chance with vanishingly small p, and the BH-corrected omnibus
guarantees they are not noise.

### 3.4 Per-scenario breakdown

The pattern holds across every one of the 8 writing scenarios. Both LLMs are
above the human in every scenario, and the human is above the o4-mini in
every scenario. The smallest LLM-vs-human gap is in `eulogy` (n=11, where
the bootstrap CIs are wide because the sample is small); the largest is in
`apology` and `letter`. See `results/final_per_scenario.csv` for the
underlying numbers.

## 4. Figure

![Figure 10: final assessment](figures/fig10_final_assessment.png)

(a) Distribution of held-out similarity for each approach, with the human
threshold and the natural same-author ceiling marked. (b) Forest plot of
all 6 pairwise Hedges' *g* with paired-bootstrap 95 % CIs; black = passes
BH-FDR. (c) Per-task win rate vs. the human post-edit, with binomial 95 %
CIs; the dotted line at 0.5 is chance. (d) Per-scenario means with
bootstrap 95 % CIs.

## 5. Caveats — same as before, restated for honesty

1. **Self-experiment.** Both Opus and GPT-5.5 generated their own drafts
   and an external model (LUAR-MUD) judged them. The metric was
   independently validated by reproducing the paper's `r = 0.244` rmcorr
   exactly (see `REPRODUCTION_REPORT.md`).
2. **n_demos = 1, not 2.** The held-out protocol gives each generator one
   style sample because the participant only has 2 unassisted controls and
   the other has to be the held-out target.
3. **Workflow comparison, not raw writing skill.** Human post-editors
   edited an *unconditioned* o4-mini draft; LLM mimics were *shown* a style
   sample and wrote from scratch. We are comparing the workflows
   (post-editing-with-no-style-info vs style-conditioned-generation), not
   the people vs the models in any general sense.
4. **Style fidelity ≠ writing quality.** LUAR measures "does this sound
   like that author?". The paper's §6.3 already shows perception and LUAR
   can come apart. Whether the mimic drafts are *useful, trustworthy, or
   factually correct* is a separate question that the paper's Likert
   instrument would need to answer.
5. **The 1 Opus outlier.** One of the 324 Opus drafts has a 56 % lexical
   overlap with its demo sample. Removing it changes the headline `g` by
   less than 0.01.

## 6. Bottom line

In the leakage-free protocol on the full 324-task sample:

- The unconditioned o4-mini draft is roughly *half* the way to human style;
  the human post-editor adds ~ 24 % of the remaining gap.
- A frontier LLM with one style sample closes 71-75 % of the gap, beats the
  human post-edit on ~ 4 of every 5 tasks, and reaches ~ 92 % of the natural
  same-author ceiling on this LUAR-MUD metric.
- Opus 4.7 and GPT-5.5 are practically tied; differences between them are
  smaller than the bootstrap noise on either alone.

## 7. The arms race: detection on the same embeddings

A leave-authors-out 5-fold linear-SVM detector on the same 512-d LUAR
embeddings (`scripts/11_detection_experiment.py`,
`tests/test_detection_no_leakage.py`) shows that detectability drops
monotonically as the approach gets closer to the participant's style:

| Approach (vs. 162 human control vectors) | AUC (95 % bootstrap CI) |
|---|---:|
| o4-mini draft | 0.999 [0.996, 1.000] |
| Human post-edit | 0.971 [0.960, 0.980] |
| Claude Opus 4.7 mimic | 0.952 [0.944, 0.960] |
| GPT-5.5 mimic | 0.931 [0.924, 0.937] |

None of the four approaches reaches chance, and the bootstrap CIs do not
overlap between adjacent rows. But six follow-up diagnostics
(`scripts/13_detection_diagnostics.py`) show the *source* of the signal
is sharply different across approaches:

- **GPT-5.5: most of the AUC is length** (length-only baseline = 0.880,
  full LUAR = 0.931). The mimic writes ~150 words reliably; humans vary.
- **Opus 4.7: the AUC is genuinely stylistic** (length-only = 0.517,
  full LUAR = 0.952).
- **Cross-LLM** (train Opus → test GPT-5.5) AUC = 0.913: a generic
  frontier-LLM signature transfers between the two.
- **Shuffle-label baseline** AUC ≈ 0.49 and **PCA(32)** AUC ≈ 0.92 confirm
  the headline is not an underdetermined-regression artefact.

So the arms race is real, but its shape is asymmetric. Reporting the
headline AUC without the length-confound diagnostic would have been
misleading. See `results/detection_diagnostics_summary.md`.

## 7. Reproducing this

```
make data embeddings sims    # paper repro pipeline (~2 min on CPU)
python scripts/08_compare_mimics.py   # rebuild mimic_similarities.parquet
python scripts/10_final_assessment.py # all the tests + Figure 10
```

Outputs: `results/final_pairwise_tests.csv`,
`results/final_win_vs_human.csv`, `results/final_friedman.csv`,
`results/final_per_scenario.csv`, `figures/fig10_final_assessment.{pdf,png}`.
