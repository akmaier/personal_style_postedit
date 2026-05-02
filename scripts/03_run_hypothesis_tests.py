"""Stage 03: run all preregistered hypothesis tests on the similarity tables."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from personal_style.data import Paths, load_processed  # noqa: E402
from personal_style.stats import benjamini_hochberg, run_test  # noqa: E402

# Paper-reported reference values (g, ci_low, ci_high, p) for each hypothesis.
# Source: Section 6 of arxiv 2604.24444.
PAPER_REFERENCE = {
    "H1a (sim_edited_control_self vs sim_llm_control_self)":             {"g":  0.55, "ci": [ 0.38,  0.71], "p": 0.0002},
    "H1b (sim_edited_llm_other vs sim_llm_llm_other)":                   {"g": -0.41, "ci": [-0.44, -0.39], "p": 0.0002},
    "H1a' (sim_edited_control_other vs sim_edited_control_self)":        {"g": -0.56, "ci": [-0.70, -0.43], "p": 0.0002},
    "H1c (sim_edited_control_self vs sim_edited_llm_other)":             {"g": -1.43, "ci": [-1.55, -1.32], "p": 0.0002},
    "H2a (pool_edited vs pool_control)":                                  {"g":  1.42, "ci": [ 1.33,  1.51], "p": 0.0002},
    "H2b (pool_edited vs pool_llm)":                                      {"g": -0.69, "ci": [-0.74, -0.63], "p": 0.0002},
    "H2c (sim_edited_edited_other vs sim_edited_control_self)":           {"g":  1.14, "ci": [ 1.02,  1.26], "p": 0.0002},
}

N_PERM = 10000
N_BOOT = 1000


def main() -> None:
    paths = Paths()
    treat = pd.read_parquet(paths.processed_dir / "sim_treatment.parquet")

    pool_ctrl = np.load(paths.processed_dir / "pool_pairwise_control.npy")
    pool_edit = np.load(paths.processed_dir / "pool_pairwise_edited.npy")
    pool_llm = np.load(paths.processed_dir / "pool_pairwise_llm.npy")

    rows = []

    # H1a: post-edit increases similarity to participant's own unassisted (control) text.
    # Pairs (per treatment observation):
    #   x = sim(edited, own control centroid) -- "after"
    #   y = sim(llm,    own control centroid) -- "before"
    # Paper g = +0.55 with x>y.
    x = treat["sim_edited_control_self"].to_numpy()
    y = treat["sim_llm_control_self"].to_numpy()
    rows.append(_row("H1a (sim_edited_control_self vs sim_llm_control_self)", x, y, paired=True, seed=1001))

    # H1b: post-edit decreases similarity to LLM-generated text.
    # The paper's Fig 3 (left) compares "before" (LLM draft -> LLM-generated text)
    # against "after" (edited -> LLM-generated text). The participant's own LLM
    # draft is excluded from "LLM-generated text" because the LLM draft has cosine
    # 1.0 with itself; we therefore use the participant's mean cosine to the
    # *other* participants' LLM drafts in both conditions.
    # Paper g = -0.41 with x="after" (edited) < y="before" (llm).
    x = treat["sim_edited_llm_other_mean"].to_numpy()
    y = treat["sim_llm_llm_other_mean"].to_numpy()
    rows.append(_row("H1b (sim_edited_llm_other vs sim_llm_llm_other)", x, y, paired=True, seed=1002))

    # H1a': post-edit moves text toward OWN style more than to OTHERS' style.
    # Paper g = -0.56 with x = sim(edited, others' control), y = sim(edited, own control)
    # i.e. self-similarity is larger than other-similarity.
    x = treat["sim_edited_control_other_mean"].to_numpy()
    y = treat["sim_edited_control_self"].to_numpy()
    rows.append(_row("H1a' (sim_edited_control_other vs sim_edited_control_self)", x, y, paired=True, seed=1003))

    # H1c: post-edited text remains closer to LLM-generated text than to own control text.
    # Paper g = -1.43 with x = sim(edited, own control), y = sim(edited, LLM-generated text).
    # Use the pooled-LLM similarity (excluding own LLM draft) to mirror Fig 4 right.
    x = treat["sim_edited_control_self"].to_numpy()
    y = treat["sim_edited_llm_other_mean"].to_numpy()
    rows.append(_row("H1c (sim_edited_control_self vs sim_edited_llm_other)", x, y, paired=True, seed=1004))

    # H2a: pool of edited texts is more homogeneous than pool of control texts.
    rows.append(_row("H2a (pool_edited vs pool_control)", pool_edit, pool_ctrl, paired=False, seed=2001))

    # H2b: pool of edited texts is less homogeneous than pool of llm texts.
    rows.append(_row("H2b (pool_edited vs pool_llm)", pool_edit, pool_llm, paired=False, seed=2002))

    # H2c: edited text is more similar to others' edited text than to own control.
    x = treat["sim_edited_edited_other_mean"].to_numpy()
    y = treat["sim_edited_control_self"].to_numpy()
    rows.append(_row("H2c (sim_edited_edited_other vs sim_edited_control_self)", x, y, paired=True, seed=2003))

    df = pd.DataFrame(rows)

    # BH-FDR over the preregistered tests (all 7 reported in §6).
    p_bh, reject = benjamini_hochberg(df["p"].to_numpy(), q=0.05)
    df["p_bh"] = p_bh
    df["bh_reject"] = reject

    # Attach paper reference values.
    paper = PAPER_REFERENCE
    df["paper_g"] = df["name"].map(lambda n: paper.get(_canonical(n), {}).get("g"))
    df["paper_ci_low"] = df["name"].map(lambda n: (paper.get(_canonical(n), {}).get("ci") or [None, None])[0])
    df["paper_ci_high"] = df["name"].map(lambda n: (paper.get(_canonical(n), {}).get("ci") or [None, None])[1])
    df["paper_p"] = df["name"].map(lambda n: paper.get(_canonical(n), {}).get("p"))

    # Sign-aware agreement: same direction and overlapping CIs.
    def _agree(row):
        if row["paper_g"] is None or pd.isna(row["paper_g"]):
            return None
        same_sign = np.sign(row["g"]) == np.sign(row["paper_g"])
        ci_overlap = not (row["ci_high"] < row["paper_ci_low"] or row["ci_low"] > row["paper_ci_high"])
        return bool(same_sign and ci_overlap)

    df["matches_paper"] = df.apply(_agree, axis=1)

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    out_csv = REPO_ROOT / "results" / "hypothesis_tests.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out_csv}")


def _canonical(name: str) -> str:
    """Map our test names to the keys used in PAPER_REFERENCE."""
    return name


def _row(name: str, x: np.ndarray, y: np.ndarray, paired: bool, seed: int) -> dict:
    res = run_test(name, x, y, paired=paired, n_perm=N_PERM, n_boot=N_BOOT, seed=seed)
    return {
        "name": res.name,
        "n": res.n,
        "paired": res.paired,
        "mean_x": res.mean_x,
        "mean_y": res.mean_y,
        "g": res.g,
        "ci_low": res.ci_low,
        "ci_high": res.ci_high,
        "p": res.p,
        "n_perm": res.n_perm,
    }


if __name__ == "__main__":
    main()
