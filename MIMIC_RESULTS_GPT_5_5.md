# Style-Mimicking Comparison: GPT-5.5 vs. Human Post-Editing and Opus 4.7
*(leakage-free held-out protocol, n = 324)*

**Result in one sentence:** *Under the same strict held-out protocol used for Opus 4.7, GPT-5.5 produces drafts that are significantly closer to the participant's own style than both the unconditioned o4-mini draft and the participant's human post-edit, and it is statistically tied with Opus 4.7 in the direct paired comparison.*

## 1. Setup

Each participant in the released study has exactly **2 control texts** (their own unassisted writing) and **4 treatment texts** (post-edits of o4-mini drafts). With only 2 controls available, the held-out v1 protocol is:

- For each of the 324 treatment tasks, deterministically pick the participant's **lower-task_idx control** as the demo shown to the generator.
- The other control text, with higher `task_idx`, is the **held-out evaluation target** and is never shown in the prompt.
- GPT-5.5, Claude Opus 4.7, the original o4-mini draft, and the human post-edit are all scored against that same held-out LUAR-MUD vector.
- Cache keys are salted by generator name and the `held_out_protocol_v1` prompt content, so GPT-5.5 keys are disjoint from the Opus cache.

The GPT-5.5 drafts were produced by 8 parallel cloud-agent subagents from `/tmp/gpt55_chunks/chunk_{00..07}.jsonl` (40-41 prompts each, all 324 covered). The merged cache is committed at [`data/processed/mimics/gpt-5.5.json`](data/processed/mimics/gpt-5.5.json).

A spot check on the 324 saved drafts: mean lexical overlap with the demo sample (difflib SequenceMatcher) is **0.088**, max overlap is **0.426**, mean draft length is **148 words** (range **133-164**), and **324/324** drafts are within the requested 100-200 word range. The drafts are mimicking, not copying.

## 2. Results (n = 324)

| Test | Hedges' *g* | 95 % CI | p (perm) | Interpretation |
|---|---:|---|---:|---|
| GPT-5.5 mimic vs. unconditioned o4-mini draft | **+1.61** | [+1.44, +1.82] | 1e-4 | GPT-5.5 drafts are dramatically closer to author style than the original LLM draft |
| GPT-5.5 mimic vs. human post-edited | **+1.07** | [+0.92, +1.25] | 1e-4 | **GPT-5.5 beats human post-editing on style fidelity** |
| GPT-5.5 mimic to *own* held-out vs. *other* participants' held-outs | +1.79 | [+1.60, +2.00] | 1e-4 | GPT-5.5 is targeting *this* author, not just being generically humanish |

Mean LUAR cosine similarity to the participant's held-out control text:

| Approach | Mean | Notes |
|---|---:|---|
| Unconditioned LLM draft (o4-mini) | **0.498** | Gets the same details prompt the human got; sees no style sample |
| Human post-edited | **0.546** | Editor never saw any control text either; this is the paper's main treatment condition |
| Claude Opus 4.7 style-mimic | **0.643** | Sees one control as style demo; never sees the held-out target |
| GPT-5.5 style-mimic | **0.649** | Same held-out prompt protocol and same 324 tasks as Opus |
| *Same-author control ↔ control upper bound* | *0.701* | *A real person writing two unassisted texts in their own voice* |

GPT-5.5 closes about **75 % of the gap** between the unconditioned o4-mini baseline (0.498) and the natural same-author within-style ceiling (0.701), versus **24 %** for human post-editing and **71 %** for Opus 4.7.

![Figure 9: leakage-free held-out comparison with GPT-5.5 and Opus 4.7](figures/fig9_llm_mimic_vs_human.png)

## 3. Direct paired comparison: GPT-5.5 vs. Opus 4.7

The direct head-to-head test uses the exact same 324 `(pid, task_idx)` pairs and compares each GPT-5.5 held-out similarity against the matching Opus 4.7 held-out similarity.

| Test | n | Mean GPT-5.5 | Mean Opus 4.7 | Hedges' *g* | 95 % CI | p (perm) | Interpretation |
|---|---:|---:|---:|---:|---|---:|---|
| GPT-5.5 vs. Claude Opus 4.7 (held-out, paired) | 324 | **0.649** | **0.643** | +0.08 | [-0.03, +0.18] | 0.138 | GPT-5.5 is numerically higher, but the paired difference is not statistically significant |

## 4. Comparison to Opus 4.7 and the previous leaky result

| Test | Opus 4.7 held-out v1 | GPT-5.5 held-out v1 |
|---|---:|---:|
| Mimic vs. o4-mini | g = +1.57 | **g = +1.61** |
| Mimic vs. human-edited | g = +1.02 | **g = +1.07** |
| Self-vs-other targeting | **g = +1.82** | g = +1.79 |

Both frontier-model mimic conditions survive the strict leakage-free evaluation. GPT-5.5 lands slightly above Opus on mean same-author held-out similarity, while Opus lands slightly above GPT-5.5 on the targeting effect size; neither difference changes the qualitative conclusion.

## 5. Caveats

1. **Self-experiment caveat.** I (GPT-5.5) generated the drafts and I am being graded by an *external* model (LUAR-MUD) on whether they sound like their authors. The metric was independently validated by reproducing the paper's reported `r = 0.244` rmcorr exactly (see `REPRODUCTION_REPORT.md`), but this is still a model-on-model evaluation.
2. **n_demos = 1, not 2.** Each prompt sees only one writing sample because the other control is reserved as the held-out target. With more controls per author the mimic condition would likely strengthen.
3. **Workflow comparison, not "humans vs LLMs at writing".** Human post-editors edited an unconditioned o4-mini draft; GPT-5.5 and Opus 4.7 were shown one style sample and wrote from scratch.
4. **Style fidelity ≠ writing quality.** LUAR measures whether a draft sounds like the author. It does not measure whether participants would prefer, trust, or endorse the generated draft.
5. **No high-overlap outlier.** The maximum GPT-5.5 demo overlap is 0.426, below the pre-specified 0.7 sanity threshold and below the Opus outlier at ~0.56.

## 6. Reproducing this

```bash
# 1. Make sure the paper reproduction is built.
make data embeddings sims

# 2. Merge agent-produced GPT-5.5 drafts into the cache.
python scripts/09_save_drafts.py --generator gpt-5.5 /tmp/gpt55_drafts/chunk_*.json

# 3. Embed mimic drafts, run paired tests, and render Figure 9.
python scripts/08_compare_mimics.py

# 4. Run the direct paired Opus-vs-GPT-5.5 comparison.
python - <<'PY'
import sys, pandas as pd
sys.path.insert(0, 'src')
from personal_style.stats import run_test
df = pd.read_parquet('data/processed/mimic_similarities.parquet')
o = df[df['generator']=='claude-opus-4-7'][['pid','task_idx','sim_mimic_heldout']].rename(columns={'sim_mimic_heldout':'opus'})
g = df[df['generator']=='gpt-5.5'][['pid','task_idx','sim_mimic_heldout']].rename(columns={'sim_mimic_heldout':'gpt'})
m = o.merge(g, on=['pid','task_idx'])
res = run_test('gpt-5.5 vs claude-opus-4-7 (heldout, paired)', m['gpt'].to_numpy(), m['opus'].to_numpy(), paired=True, n_perm=10000, seed=42)
print(res)
PY
```

Outputs:

- `data/processed/mimics/gpt-5.5.json`
- `data/processed/mimic_similarities.parquet`
- `results/mimic_comparison.csv`
- `results/mimic_head_to_head.csv`
- `figures/fig9_llm_mimic_vs_human.{pdf,png}`
