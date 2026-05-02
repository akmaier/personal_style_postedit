"""Stage 08: embed mimic drafts with LUAR and compare to human post-editing.

**Held-out evaluation protocol (v1).** Each participant has only 2 control texts.
The generator is shown exactly one of them (the demo, with the lower task_idx)
and is *never* shown the other (the held-out target). All four approaches
(o4-mini draft, human post-edit, Opus 4.7 mimic, GPT-5.5 mimic) are scored
against the same single held-out vector, so they share a leakage-free yardstick.

For each treatment task with a mimic draft we compute:
  sim_<approach>_heldout = cos( embed(<approach output>), embed(held-out control) )
  sim_<approach>_other_mean = mean over OTHER participants of cos( <approach>, that pid's heldout )

Outputs:
  - data/processed/mimic_similarities.parquet
  - results/mimic_comparison.csv  (paired stats per generator)
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


def build_heldout_index(base: EmbeddingTable, obs: pd.DataFrame) -> dict[str, dict]:
    """For each pid: dict with keys
        demo_idx, demo_vec, held_out_idx, held_out_vec
    using the same protocol as scripts/07: the lower-task_idx control is the
    demo, the higher-task_idx control is the held-out target.
    """
    pki = participant_kind_indices(base)
    out: dict[str, dict] = {}
    pids = sorted(set(base.pids.tolist()))
    for pid in pids:
        ctrl_indices = pki.get((pid, "control"), [])
        if len(ctrl_indices) < 2:
            continue
        # Sort by task_idx
        rows = sorted(
            [(int(base.task_idx[i]), i) for i in ctrl_indices], key=lambda x: x[0]
        )
        demo_ti, demo_emb_idx = rows[0]
        ho_ti, ho_emb_idx = rows[1]
        out[pid] = {
            "demo_idx": demo_ti,
            "demo_vec": base.vecs[demo_emb_idx],
            "held_out_idx": ho_ti,
            "held_out_vec": base.vecs[ho_emb_idx],
        }
    return out


def main() -> None:
    paths = Paths()
    obs, _ = load_processed(paths)

    mimics = load_mimic_drafts(paths)
    if mimics.empty:
        raise SystemExit("no mimic drafts found; run scripts/07_generate_mimics.py first")
    print(f"mimic drafts loaded: {len(mimics)} ({mimics['generator'].value_counts().to_dict()})")

    # -- Embed mimic drafts with LUAR (idempotent)
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
    if cached:
        all_keys = list(cached.keys())
        all_vecs = np.stack([cached[k] for k in all_keys]).astype(np.float32)
        pids_arr = np.array([k[0] for k in all_keys], dtype="U64")
        task_arr = np.array([k[1] for k in all_keys], dtype=np.int32)
        gens_arr = np.array([k[2] for k in all_keys], dtype="U32")
        np.savez_compressed(cache_path, pids=pids_arr, task_idx=task_arr,
                            generators=gens_arr, vecs=all_vecs)

    # -- Load base (control / llm / edited) embeddings and build the held-out index
    base_npz = np.load(paths.processed_dir / "embeddings.npz")
    base = EmbeddingTable(
        base_npz["pids"].astype(str),
        base_npz["kinds"].astype(str),
        base_npz["task_idx"].astype(int),
        base_npz["scenarios"].astype(str),
        base_npz["vecs"].astype(np.float32),
    )
    held = build_heldout_index(base, obs)
    if not held:
        raise SystemExit("no participants have >= 2 control texts; cannot run held-out protocol")
    print(f"held-out index built for {len(held)} participants")

    # -- Build per-treatment-task scores against held-out target
    # Index base table for quick lookup of (pid, kind, task_idx) -> vec
    base_idx = {}
    for i, (p, k, t) in enumerate(zip(base.pids, base.kinds, base.task_idx)):
        base_idx[(str(p), str(k), int(t))] = base.vecs[i]

    pids_all = sorted(held.keys())
    held_vec = {pid: held[pid]["held_out_vec"] for pid in pids_all}

    rows = []
    for (pid, ti, gen), vec in cached.items():
        pid = str(pid)
        ti = int(ti)
        if pid not in held:
            continue
        ho_vec = held_vec[pid]
        s_self = cosine(vec, ho_vec)
        # Mean cos to OTHER participants' held-out vectors
        scores = [cosine(vec, held_vec[q]) for q in pids_all if q != pid]
        s_other = float(np.mean(scores)) if scores else float("nan")

        # Same-task baselines: o4-mini draft and human post-edit, also scored
        # against THIS participant's held-out vector
        llm_vec = base_idx.get((pid, "llm", ti))
        edited_vec = base_idx.get((pid, "edited", ti))
        s_llm_heldout = cosine(llm_vec, ho_vec) if llm_vec is not None else float("nan")
        s_edited_heldout = cosine(edited_vec, ho_vec) if edited_vec is not None else float("nan")

        rows.append(
            {
                "pid": pid,
                "task_idx": ti,
                "generator": str(gen),
                "held_out_task_idx": int(held[pid]["held_out_idx"]),
                "demo_task_idx": int(held[pid]["demo_idx"]),
                "sim_mimic_heldout": float(s_self),
                "sim_mimic_heldout_other_mean": s_other,
                "sim_llm_heldout": float(s_llm_heldout),
                "sim_edited_heldout": float(s_edited_heldout),
            }
        )
    out = pd.DataFrame(rows)

    # Attach scenario for plotting nicety
    obs_t = obs[obs["condition"] == "treatment"][["pid", "task_idx", "scenario"]].copy()
    obs_t["pid"] = obs_t["pid"].astype(str)
    out = out.merge(obs_t, on=["pid", "task_idx"], how="left")

    out_path = paths.processed_dir / "mimic_similarities.parquet"
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path}: {len(out)} rows ({out['generator'].value_counts().to_dict()})")

    # -- Hypothesis tests, paired (per task), per generator, against held-out target
    test_rows = []
    for gen, sub in out.groupby("generator"):
        sub = sub.dropna(subset=["sim_mimic_heldout", "sim_llm_heldout", "sim_edited_heldout"])
        if len(sub) < 5:
            continue
        m = sub["sim_mimic_heldout"].to_numpy()
        l = sub["sim_llm_heldout"].to_numpy()
        e = sub["sim_edited_heldout"].to_numpy()
        m_other = sub["sim_mimic_heldout_other_mean"].to_numpy()
        r1 = run_test(f"{gen}: mimic vs o4-mini (heldout)", m, l, paired=True, n_perm=10000, seed=9001 + len(test_rows))
        test_rows.append(_row(r1, len(sub)))
        r2 = run_test(f"{gen}: mimic vs human-edited (heldout)", m, e, paired=True, n_perm=10000, seed=9001 + len(test_rows))
        test_rows.append(_row(r2, len(sub)))
        r3 = run_test(f"{gen}: mimic self vs other-author heldout (targeting)", m, m_other, paired=True, n_perm=10000, seed=9001 + len(test_rows))
        test_rows.append(_row(r3, len(sub)))

    test_df = pd.DataFrame(test_rows)
    res_csv = REPO_ROOT / "results" / "mimic_comparison.csv"
    res_csv.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(res_csv, index=False)
    if not test_df.empty:
        print("\n", test_df.to_string(index=False))
    print(f"\nwrote {res_csv}")

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


def _same_author_upper_bound() -> float | None:
    """Mean LUAR cos between each participant's two control texts.

    This is the natural upper bound on the held-out metric: a person writing
    two unassisted texts in their own voice. Any approach that exceeds this
    has plausibly memorized the held-out target rather than mimicking style.
    """
    try:
        z = np.load(Paths().processed_dir / "embeddings.npz")
        t = EmbeddingTable(
            z["pids"].astype(str), z["kinds"].astype(str),
            z["task_idx"].astype(int), z["scenarios"].astype(str),
            z["vecs"].astype(np.float32),
        )
        pi = participant_kind_indices(t)
        sims = []
        for pid in sorted(set(t.pids.tolist())):
            ctrl = pi.get((pid, "control"), [])
            if len(ctrl) < 2:
                continue
            rows = sorted([(int(t.task_idx[i]), i) for i in ctrl], key=lambda x: x[0])
            sims.append(cosine(t.vecs[rows[0][1]], t.vecs[rows[1][1]]))
        return float(np.mean(sims)) if sims else None
    except Exception as e:
        print(f"upper-bound calculation skipped: {e!r}")
        return None


def _make_figure9(df: pd.DataFrame, out_dir: Path) -> None:
    """Distribution of similarity-to-held-out-control by approach."""
    palette = {"llm": "#d95f02", "edited": "#1b9e77",
               "gpt-5.5": "#7570b3", "claude-opus-4-7": "#e7298a", "stub": "#666666"}

    # Use per-row dedup so o4-mini and human-edited each contribute one value per task,
    # not one per (task, generator) row.
    base = df.drop_duplicates(subset=["pid", "task_idx"])

    series = [base["sim_llm_heldout"].dropna().to_numpy(),
              base["sim_edited_heldout"].dropna().to_numpy()]
    labels = ["LLM draft\n(o4-mini)", "Human\npost-edited"]
    colors = [palette["llm"], palette["edited"]]
    for gen in sorted(df["generator"].unique()):
        sub = df[df["generator"] == gen]["sim_mimic_heldout"].dropna().to_numpy()
        series.append(sub)
        nice = {"gpt-5.5": "GPT-5.5\nstyle-mimic",
                "claude-opus-4-7": "Claude Opus 4.7\nstyle-mimic",
                "stub": "Stub\nstyle-mimic"}.get(gen, gen)
        labels.append(nice)
        colors.append(palette.get(gen, "#888888"))

    fig, ax = plt.subplots(figsize=(2.0 + 1.6 * len(series), 4.6))
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
        ax.text(i, arr.mean() + 0.02, f"{arr.mean():.3f}",
                ha="center", fontsize=8, color="black")

    upper = _same_author_upper_bound()
    if upper is not None:
        ax.axhline(upper, color="#444", ls="--", lw=1.0)
        ax.text(len(series) - 0.5, upper + 0.005,
                f"same-author control \u2194 control\nupper bound = {upper:.3f}",
                ha="right", va="bottom", fontsize=8, color="#333")

    ax.set_xticks(range(len(series)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("LUAR cosine sim. to HELD-OUT control text\n(participant's other unassisted text)")
    ax.set_title("Figure 9: leakage-free comparison vs the participant's held-out control text")
    fig.tight_layout()
    fig.savefig(out_dir / "fig9_llm_mimic_vs_human.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig9_llm_mimic_vs_human.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
