# Detection diagnostics — what the AUCs actually mean

After reporting AUC = 0.999 / 0.971 / 0.952 / 0.931 in §7, we ran six
diagnostics to test whether the numbers reflect a real stylistic
signature, an underdetermined-regression artefact, an author leak, a
length confound, or some mix. The honest answer turns out to be a mix
that is more interesting than the headline numbers suggest.

## Headline diagnostics (full numbers in `detection_diagnostics.csv`)

| Diagnostic | o4-mini | Human edit | Opus 4.7 | GPT-5.5 |
|---|---:|---:|---:|---:|
| **Original LinearSVC on 512-d LUAR** | 0.999 | 0.971 | 0.952 | 0.931 |
| A. Param count                       | 513 weights, 384–390 samples per fold; **underdetermined** but harmless (B / E / F below) ||||
| A. Author audit                      | 0/0/0/0/0 pid overlap per fold for every approach — no leak ||||
| B. Shuffled labels                   | 0.472 | 0.469 | 0.494 | 0.478 |
| C. **Length-only** (single feature)  | **0.818** | 0.565 | **0.517** | **0.880** |
| D. Cross-LLM (train Opus → test GPT) | — | — | 0.913 | 0.888 |
| E. PCA(32) + LinearSVC               | 0.998 | 0.968 | 0.925 | 0.912 |
| F. L2-LR, C = 1e-3                   | 0.998 | 0.976 | 0.931 | 0.926 |

## Three honest take-aways

1. **The AUCs are not artefacts.** Shuffling labels gives AUC ≈ 0.49,
   PCA-32 + LinearSVC (33 params, well-determined) and a heavily-regularised
   LR both recover essentially the same AUCs. The detection result
   survives every robustness check.

2. **GPT-5.5 detection is mostly a length game.** A single-feature
   length-only model already reaches AUC 0.880 against the human controls,
   versus 0.931 for the full 512-d LUAR-SVM. So only ~0.05 of those AUC
   points are stylistic; the other ~0.38 is "GPT-5.5 writes ~150 words
   reliably; humans vary more". The **t-SNE picture** that prompted these
   diagnostics is consistent: GPT-5.5 mimics overlap human controls in
   stylistic-direction space, but a length axis separates them.

3. **Opus detection is genuinely stylistic.** Length-only gets only 0.517
   on Opus; the full LUAR-SVM gets 0.952. The 0.95 AUC really is picking
   up a stylistic signature — it just lives in directions of the 512-d
   space that t-SNE projects onto axes shared with human controls.

4. **Cross-LLM detection** gives ~0.9 AUC. Whatever the SVM picks up on
   one frontier LLM transfers strongly to the other — there is a generic
   "frontier-LLM mimic" signature that is shared across Opus and GPT-5.5.

## What this means for the paper claim

The paper currently says the detection signal is a "residual stylometric
signature persists even in the strongest mimic". For Opus this is
correct. For GPT-5.5 this is **misleading** — most of that signal is
length, not style. The honest update is:

> *Detection AUC drops monotonically with stylistic similarity, but the
> sources of the residual signal differ across approaches: the
> unconditioned o4-mini and the GPT-5.5 mimic are largely flagged by
> output length, while the Claude Opus 4.7 mimic carries an actual
> stylistic signature that LUAR-MUD captures in directions a 2-D
> projection mostly hides.*

This is the version we now report.

## What it means about the underdetermined regime

LinearSVC on raw 512-d has more parameters (513) than per-fold training
samples (~390). That is genuinely a regime where a hard-margin SVM can
linearly separate almost any binary labelling. Two safeguards:

- The shuffled-labels diagnostic gives AUC ~0.5 on test, proving the
  SVM does **not** memorize random labellings *across* authors (because
  GroupKFold prevents author leakage; the test fold contains different
  authors, so memorising in-fold patterns does not help test-fold AUC).
- The PCA(32) variant cuts the parameter count to 33, well below the
  per-fold sample count of ~390. The AUCs barely move (e.g. Opus 0.952
  → 0.925), proving the original numbers are not an underdetermined
  artefact.
