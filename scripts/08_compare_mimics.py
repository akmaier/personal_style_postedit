"""Stage 08: embed mimic drafts with LUAR and compare to human post-editing.

For every treatment task with a mimic draft we compute:
  sim(mimic_<model>, participant's control style)

We then compare distributions of this similarity across:
  - LLM original (the o4-mini draft used in the study)
  - human post-edited (treatment final_text)
  - GPT-5.5 mimic
  - Claude Opus 4.7 mimic

Outputs:
  - data/processed/mimic_similarities.parquet
  - results/mimic_comparison.csv (paired stats edited <-> mimic)
  - figures/fig9_llm_mimic_vs_human.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from personal_style.data import Paths, load_processed  # noqa: E402
from personal_style.embeddings import embed_texts, load_luar  # noqa: E402
from personal_style.similarity import (  # noqa: E402
    EmbeddingTable,
    cosine,
    mean_cos_to_set,
    normalize,
    participant_kind_indices,
)
from personal_style.stats import run_test  # noqa: E402


MIMIC_GENERATORS = ["gpt-5.5", "claude-opus-4-7", "stub"]


def load_mimic_drafts(paths: Paths) -> pd.DataFrame:
    rows = []
    mimic_dir = paths.processed_dir / "mimics"
    if not mimic_dir.exists():
        return pd.DataFrame()
    for fname in mimic_dir.glob("*.json"):
        data = json.loads(fname.read_text())
        for v in data.values():
            rows.append(v)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["pid"] = df["pid"].astype(str)
    df["task_idx"] = df["task_idx"].astype(int)
    return df


def main() -> None:
    paths = Paths()
    obs, _ = load_processed(paths)

    mimics = load_mimic_drafts(paths)
    if mimics.empty:
        raise SystemExit("no mimic drafts found; run scripts/07_generate_mimics.py first")
    gens = sorted(mimics["generator"].unique())
    print(f"mimic drafts loaded: {len(mimics)} ({mimics['generator'].value_counts().to_dict()})")

    # -- Embed mimic drafts with LUAR
    cache_path = paths.processed_dir / "mimic_embeddings.npz"
    if cache_path.exists():
        z = np.load(cache_path)
        m_vecs = z["vecs"]
        m_keys = list(zip(z["pids"].astype(str), z["task_idx"].astype(int), z["generators"].astype(str)))
        cached = dict(zip(m_keys, m_vecs))
    else:
        cached = {}
    keys_needed = list(zip(mimics["pid"].astype(str), mimics["task_idx"].astype(int),
                            mimics["generator"].astype(str)))
    new_idx = [i for i, k in enumerate(keys_needed) if k not in cached]
    if new_idx:
        tokenizer, model, device = load_luar()
        print(f"loaded LUAR-MUD on {device}; embedding {len(new_idx)} new mimic drafts")
        new_texts = mimics["text"].iloc[new_idx].tolist()
        new_vecs = embed_texts(new_texts, tokenizer, model, device, batch_size=16)
        for i, vec in zip(new_idx, new_vecs):
            cached[keys_needed[i]] = vec
    # Persist cache
    if cached:
        all_keys = list(cached.keys())
        all_vecs = np.stack([cached[k] for k in all_keys]).astype(np.float32)
        pids = np.array([k[0] for k in all_keys], dtype="U64")
        task_idx = np.array([k[1] for k in all_keys], dtype=np.int32)
        gens_arr = np.array([k[2] for k in all_keys], dtype="U32")
        np.savez_compressed(cache_path, pids=pids, task_idx=task_idx, generators=gens_arr, vecs=all_vecs)

    # -- Build a per-task table with similarity to participant's control style
    # We need each participant's control-text vectors (already embedded in stage 01)
    base_npz = np.load(paths.processed_dir / "embeddings.npz")
    base_pids = base_npz["pids"].astype(str)
    base_kinds = base_npz["kinds"].astype(str)
    base_task = base_npz["task_idx"].astype(int)
    base_vecs = base_npz["vecs"].astype(np.float32)
    base = EmbeddingTable(base_pids, base_kinds, base_task, base_npz["scenarios"].astype(str), base_vecs)

    pki = participant_kind_indices(base)
    pids_all = sorted(set(base.pids.tolist()))
    pid_ctrl_vecs = {p: base.vecs[pki.get((p, "control"), [])] for p in pids_all}
    pid_ctrl_centroid = {
        p: (normalize(v.mean(axis=0)) if len(v) else None) for p, v in pid_ctrl_vecs.items()
    }

    # Existing similarity table (treatment rows already have edited & llm sims to control)
    treat_sim = pd.read_parquet(paths.processed_dir / "sim_treatment.parquet")

    rows = []
    for (pid, ti, gen), vec in cached.items():
        cent = pid_ctrl_centroid.get(pid)
        if cent is None:
            continue
        # this mimic vs participant's own control style centroid
        s_self = cosine(vec, cent)
        # this mimic vs *other* participants' controls
        other_pids = [q for q in pids_all if q != pid]
        scores = []
        for q in other_pids:
            if pid_ctrl_centroid[q] is None or len(pid_ctrl_vecs[q]) == 0:
                continue
            scores.append(mean_cos_to_set(vec, pid_ctrl_vecs[q]))
        s_other = float(np.mean(scores)) if scores else float("nan")
        rows.append(
            {
                "pid": pid,
                "task_idx": int(ti),
                "generator": gen,
                "sim_mimic_control_self": float(s_self),
                "sim_mimic_control_other_mean": s_other,
            }
        )
    sim_df = pd.DataFrame(rows)
    out = sim_df.merge(
        treat_sim[
            [
                "pid", "task_idx", "scenario",
                "sim_edited_control_self", "sim_llm_control_self",
                "sim_edited_control_other_mean", "sim_llm_control_other_mean",
            ]
        ],
        on=["pid", "task_idx"],
        how="inner",
    )
    out_path = paths.processed_dir / "mimic_similarities.parquet"
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path}: {len(out)} rows")

    # -- Hypothesis tests: paired (per task) for each generator
    # H_mimic_a: mimic similarity to own control >  llm draft similarity to own control
    # H_mimic_b: mimic similarity to own control vs human-edited similarity to own control
    # H_mimic_c: mimic vs llm in self-vs-other gap (does mimic shift toward THIS author?)
    test_rows = []
    for gen, sub in out.groupby("generator"):
        sub = sub.dropna(subset=["sim_mimic_control_self", "sim_llm_control_self", "sim_edited_control_self"])
        if len(sub) < 5:
            continue
        m = sub["sim_mimic_control_self"].to_numpy()
        l = sub["sim_llm_control_self"].to_numpy()
        e = sub["sim_edited_control_self"].to_numpy()
        # vs LLM original
        r1 = run_test(f"{gen}: mimic vs llm (self-similarity)", m, l, paired=True, n_perm=10000, seed=8000 + len(test_rows))
        test_rows.append(_row(r1, len(sub)))
        # vs human-edited
        r2 = run_test(f"{gen}: mimic vs human-edited (self-similarity)", m, e, paired=True, n_perm=10000, seed=8000 + len(test_rows))
        test_rows.append(_row(r2, len(sub)))
        # mimic self - mimic other (should be > 0 if mimic is targeting THIS author)
        m_other = sub["sim_mimic_control_other_mean"].to_numpy()
        r3 = run_test(f"{gen}: mimic self vs mimic other (targeting check)", m, m_other, paired=True, n_perm=10000, seed=8000 + len(test_rows))
        test_rows.append(_row(r3, len(sub)))

    test_df = pd.DataFrame(test_rows)
    res_csv = REPO_ROOT / "results" / "mimic_comparison.csv"
    res_csv.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(res_csv, index=False)
    print("\n", test_df.to_string(index=False))
    print(f"\nwrote {res_csv}")

    # -- Figure 9: comparative violin
    figdir = REPO_ROOT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    _make_figure9(out, figdir)


def _row(res, n: int) -> dict:
    return {
        "name": res.name,
        "n": n,
        "g": res.g,
        "ci_low": res.ci_low,
        "ci_high": res.ci_high,
        "p": res.p,
        "mean_x": res.mean_x,
        "mean_y": res.mean_y,
    }


def _make_figure9(df: pd.DataFrame, out_dir: Path) -> None:
    """Distribution of similarity-to-own-control by approach."""
    series = []
    labels = []
    colors = []
    palette = {"llm": "#d95f02", "edited": "#1b9e77", "gpt-5.5": "#7570b3", "claude-opus-4-7": "#e7298a", "stub": "#666666"}
    # original LLM and human-edited (one row per task -> one value)
    series.append(df["sim_llm_control_self"].dropna().to_numpy())
    labels.append("LLM draft\n(o4-mini)")
    colors.append(palette["llm"])
    series.append(df["sim_edited_control_self"].dropna().to_numpy())
    labels.append("Human\npost-edited")
    colors.append(palette["edited"])
    for gen in sorted(df["generator"].unique()):
        sub = df[df["generator"] == gen]["sim_mimic_control_self"].dropna().to_numpy()
        series.append(sub)
        nice = {"gpt-5.5": "GPT-5.5\nstyle-mimic", "claude-opus-4-7": "Claude Opus 4.7\nstyle-mimic",
                "stub": "Stub\nstyle-mimic"}.get(gen, gen)
        labels.append(nice)
        colors.append(palette.get(gen, "#888888"))

    fig, ax = plt.subplots(figsize=(2.0 + 1.6 * len(series), 4.4))
    parts = ax.violinplot(series, positions=range(len(series)), widths=0.85,
                          showmeans=False, showextrema=False)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.55)
        pc.set_edgecolor("black")
    rng = np.random.RandomState(0)
    for i, arr in enumerate(series):
        ax.scatter(np.full_like(arr, i, dtype=float) + rng.uniform(-0.07, 0.07, size=len(arr)),
                   arr, s=5, color="black", alpha=0.25)
        ax.hlines(arr.mean(), i - 0.3, i + 0.3, color="black", lw=1.5)
        ax.text(i, arr.mean() + 0.02, f"{arr.mean():.3f}", ha="center", fontsize=8, color="black")
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("LUAR cosine sim. to own control text")
    ax.set_title("Figure 9: how close to the participant's own style does each approach get?")
    fig.tight_layout()
    fig.savefig(out_dir / "fig9_llm_mimic_vs_human.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig9_llm_mimic_vs_human.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
