# Style-Mimicking Comparison: Claude Opus 4.7 vs. Human Post-Editing
*(leakage-free held-out protocol, n = 324)*

**Result in one sentence:** *Under a strict held-out protocol where the
model never sees the text it is later evaluated against, Claude Opus 4.7,
given a single unassisted writing sample from the author, produces drafts
that are significantly more similar to the participant's own style than what
the participants themselves produced through manual post-editing of an
unconditioned o4-mini draft.*

## 1. Setup

Each participant in the released study has exactly **2 control texts** (their
own unassisted writing) and 4 treatment texts (post-edits of o4-mini drafts).
With only 2 controls available, the only way to evaluate "does it sound like
this author?" without leaking the answer into the prompt is a **leave-one-out
protocol**:

- For each of the 324 treatment tasks, deterministically pick the participant's
  **lower-task_idx control** as the demo (shown to the model). Their other
  control (higher task_idx) becomes the **held-out evaluation target**, never
  shown to anyone.
- The Opus mimic, the o4-mini draft, and the human post-edit are all scored
  by LUAR-MUD cosine similarity to the same single held-out vector. Same
  yardstick for all three approaches.
- Cache keys for mimic generations are salted with a `held_out_protocol_v1`
  tag so leaky drafts from earlier iterations cannot accidentally be re-used.

The drafts themselves were produced by 8 parallel Opus 4.7 cloud-agent
subprocesses (40-41 prompts each, all 324 covered). They are committed at
[`data/processed/mimics/claude-opus-4-7.json`](data/processed/mimics/claude-opus-4-7.json)
keyed by SHA-256 of the prompt content + protocol salt, so the entire
analysis re-runs from scratch with no further model calls.

A spot check on the 324 saved drafts: mean lexical overlap with the demo
sample (difflib SequenceMatcher) is 0.07 with a single outlier at 0.56. Mean
draft length is 165 words (range 144-204; 322/324 within the requested
100-200 range). The drafts are mimicking, not copying.

## 2. Results (n = 324)

| Test | Hedges' *g* | 95 % CI | p (perm) | Interpretation |
|---|---:|---|---:|---|
| Opus mimic vs. unconditioned o4-mini draft | **+1.57** | [+1.39, +1.77] | 1e-4 | Opus drafts are dramatically closer to author style than the original LLM draft |
| Opus mimic vs. human post-edited | **+1.02** | [+0.86, +1.19] | 1e-4 | **Opus beats human post-editing on style fidelity** |
| Opus mimic to *own* held-out vs. *other* participants' held-outs | +1.82 | [+1.66, +2.01] | 1e-4 | Opus is targeting *this* author, not just being generically humanish |

Mean LUAR cosine similarity to the participant's held-out control text:

| Approach | Mean | Notes |
|---|---:|---|
| Unconditioned LLM draft (o4-mini) | **0.498** | Gets the same details prompt the human got; sees no style sample |
| Human post-edited                 | **0.546** | Editor never saw any control text either; this is the paper's main treatment condition |
| Claude Opus 4.7 style-mimic       | **0.643** | Sees one control as style demo; never sees the held-out target |
| *Same-author control \u2194 control upper bound* | *0.701* | *A real person writing two unassisted texts in their own voice* |

Opus closes about **65 % of the gap** between the unconditioned o4-mini
baseline (0.498) and the natural same-author within-style ceiling (0.701),
versus **24 %** for human post-editing of the same o4-mini draft.

## 3. Comparison to the previous (leaky) result

For full transparency, an earlier version of this experiment showed *both*
controls as style demos and then evaluated against the centroid of those
same two controls. Effect sizes shrunk meaningfully when that leak was
removed:

| Test | Leaky (n = 85) | Held-out v1 (n = 324) |
|---|---:|---:|
| Opus mimic vs o4-mini  | g = +1.80 | **g = +1.57** |
| Opus mimic vs edited    | g = +1.18 | **g = +1.02** |
| Self-vs-other targeting | g = +2.42 | **g = +1.82** |

The conclusion is unchanged but more conservative under the strict protocol.
That's the kind of shrinkage we should expect once memorization is no longer
available, and it is reassuring that the qualitative answer survives.

![Figure 9: leakage-free held-out comparison](figures/fig9_llm_mimic_vs_human.png)

## 4. Caveats

1. **Self-experiment caveat.** I (Claude Opus 4.7) generated the drafts and
   I am being graded by an *external* model (LUAR-MUD) on whether they
   sound like their authors. The metric was independently validated by
   reproducing the paper's reported `r = 0.244` rmcorr exactly (see
   `REPRODUCTION_REPORT.md`), so I trust the LUAR pipeline. Still, the
   result wants confirmation from an independent generator (GPT-5.5).
2. **n_demos = 1, not 2.** Each prompt sees only one writing sample. The
   participant block randomization unfortunately limits us to this. With
   more controls per author the result would likely strengthen.
3. **Workflow comparison, not "humans vs LLMs at writing".** Human
   post-editors edited an unconditioned o4-mini draft; the Opus mimic was
   shown one style sample and wrote from scratch. Two workflows, not two
   raw writing skills.
4. **Style fidelity \u2260 writing quality.** LUAR measures "does this *sound*
   like that author?" The paper's own \u00a76.3 already shows perception and
   LUAR can come apart \u2014 nothing here measures whether participants would
   actually find Opus drafts more authentic, useful, or factually trustworthy.
5. **One outlier.** One of the 324 drafts (key `9b809b861881e027`) has a
   ~56 % lexical overlap with its demo sample, suggesting partial verbatim
   reuse on that single prompt. Removing that one row changes the headline
   `g` by less than 0.01, so I left it in for honesty rather than cherry-pick.

## 5. Reproducing this

```
# 1. Make sure the paper reproduction is built (so embeddings.npz exists)
make data embeddings sims

# 2. The Opus drafts are committed in data/processed/mimics/claude-opus-4-7.json
#    so no API calls are needed to re-run the analysis below:
python scripts/08_compare_mimics.py
```

Outputs:

- `figures/fig9_llm_mimic_vs_human.{pdf,png}`
- `results/mimic_comparison.csv`
- `data/processed/mimic_similarities.parquet`

## 6. Next step (over to GPT-5.5)

Once `OPENAI_API_KEY` is added in the Cursor Dashboard:

```
python scripts/07_generate_mimics.py --generators gpt-5.5
make mimic-compare
```

That will add a fourth violin to Figure 9 and let us paired-test Opus 4.7
directly against GPT-5.5 on the exact same 324 tasks under the same
held-out protocol.
