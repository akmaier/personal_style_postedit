"""Figure-generating functions for Figures 3-8."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    }
)


def _save(fig, out_dir: Path, name: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def _violin_pair(ax, x: np.ndarray, y: np.ndarray, labels: tuple[str, str], color: str):
    parts = ax.violinplot([x, y], positions=[0, 1], widths=0.8, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.55)
        pc.set_edgecolor("black")
    for i, arr in enumerate([x, y]):
        ax.scatter(np.full_like(arr, i, dtype=float) + np.random.RandomState(0).uniform(-0.07, 0.07, size=len(arr)),
                   arr, s=4, color="black", alpha=0.25)
        ax.hlines(arr.mean(), i - 0.3, i + 0.3, color="white", lw=2)
        ax.hlines(arr.mean(), i - 0.3, i + 0.3, color="black", lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)


def figure3(treat: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Fig. 3: similarity to LLM-text (left) and control text (right) before vs. after."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
    _violin_pair(
        axes[0],
        x=treat["sim_llm_llm_other_mean"].to_numpy(),
        y=treat["sim_edited_llm_other_mean"].to_numpy(),
        labels=("Before\n(LLM draft)", "After\n(post-edited)"),
        color="#d95f02",
    )
    axes[0].set_ylabel("LUAR cosine sim. to LLM-generated text")
    axes[0].set_title("H1b: similarity to LLM-generated text")

    _violin_pair(
        axes[1],
        x=treat["sim_llm_control_self"].to_numpy(),
        y=treat["sim_edited_control_self"].to_numpy(),
        labels=("Before\n(LLM draft)", "After\n(post-edited)"),
        color="#1b9e77",
    )
    axes[1].set_ylabel("LUAR cosine sim. to own control text")
    axes[1].set_title("H1a: similarity to own control text")
    fig.suptitle("Figure 3 (reproduction): post-editing shifts style toward control & away from LLM")
    fig.tight_layout()
    return _save(fig, out_dir, "fig3_before_after")


def figure4(treat: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Fig. 4: post-edited text closer to LLM than to control (H1c). Before & after side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), sharey=True)
    _violin_pair(
        axes[0],
        x=treat["sim_llm_control_self"].to_numpy(),
        y=treat["sim_llm_llm_other_mean"].to_numpy(),
        labels=("vs. control", "vs. LLM"),
        color="#7570b3",
    )
    axes[0].set_title("Before post-editing\n(LLM draft compared to ...)")
    axes[0].set_ylabel("LUAR cosine similarity")

    _violin_pair(
        axes[1],
        x=treat["sim_edited_control_self"].to_numpy(),
        y=treat["sim_edited_llm_other_mean"].to_numpy(),
        labels=("vs. control", "vs. LLM"),
        color="#7570b3",
    )
    axes[1].set_title("After post-editing\n(edited compared to ...)")
    fig.suptitle("Figure 4 (reproduction): edited text remains closer to LLM than to own control")
    fig.tight_layout()
    return _save(fig, out_dir, "fig4_treatment_vs_control_llm")


def figure5(treat: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Fig. 5: self vs other for control (left) and edited (right) reference texts."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
    _violin_pair(
        axes[0],
        x=treat["sim_edited_control_self"].to_numpy(),
        y=treat["sim_edited_control_other_mean"].to_numpy(),
        labels=("vs. own\ncontrol", "vs. others'\ncontrol"),
        color="#1b9e77",
    )
    axes[0].set_title("H1a': edited text vs. control")
    axes[0].set_ylabel("LUAR cosine similarity")

    _violin_pair(
        axes[1],
        x=treat["sim_edited_control_self"].to_numpy(),
        y=treat["sim_edited_edited_other_mean"].to_numpy(),
        labels=("vs. own\ncontrol", "vs. others'\nedited"),
        color="#d95f02",
    )
    axes[1].set_title("H2c: edited text vs. own-control vs. others'-edited")
    fig.suptitle("Figure 5 (reproduction): self vs. other-participant similarity")
    fig.tight_layout()
    return _save(fig, out_dir, "fig5_self_vs_other")


def figure6(pool_ctrl: np.ndarray, pool_edit: np.ndarray, pool_llm: np.ndarray, out_dir: Path) -> tuple[Path, Path]:
    """Fig. 6: within-pool homogeneity for the three text kinds."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    data = [pool_llm, pool_edit, pool_ctrl]
    labels = ["LLM-generated", "Post-edited", "Control (human)"]
    colors = ["#d95f02", "#7570b3", "#1b9e77"]
    parts = ax.violinplot(data, positions=range(len(data)), widths=0.85, showmeans=False, showextrema=False)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.55)
        pc.set_edgecolor("black")
    for i, arr in enumerate(data):
        ax.hlines(arr.mean(), i - 0.3, i + 0.3, color="black", lw=1.5)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pairwise LUAR cosine sim.\n(across-participant)")
    ax.set_title("Figure 6 (reproduction): within-pool stylistic homogeneity")
    fig.tight_layout()
    return _save(fig, out_dir, "fig6_homogeneity")


def figure7(perceived: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Fig. 7: perceived self-similarity by condition (LLM draft, edited, control)."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    data = [perceived["llm"].dropna().to_numpy(), perceived["edited"].dropna().to_numpy(), perceived["control"].dropna().to_numpy()]
    labels = ["LLM draft\n(perceived)", "Post-edited\n(perceived)", "Control\n(perceived)"]
    colors = ["#d95f02", "#7570b3", "#1b9e77"]
    parts = ax.violinplot(data, positions=range(len(data)), widths=0.85, showmeans=False, showextrema=False)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.55)
        pc.set_edgecolor("black")
    for i, arr in enumerate(data):
        ax.scatter(np.full_like(arr, i, dtype=float) + np.random.RandomState(0).uniform(-0.07, 0.07, size=len(arr)),
                   arr, s=6, color="black", alpha=0.25)
        ax.hlines(arr.mean(), i - 0.3, i + 0.3, color="black", lw=1.5)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Perceived self-similarity (1-5 Likert)")
    ax.set_title("Figure 7 (reproduction): perceived self-similarity by condition")
    fig.tight_layout()
    return _save(fig, out_dir, "fig7_perceived")


def figure8(merged: pd.DataFrame, out_dir: Path, rmcorr_r: float, rmcorr_p: float) -> tuple[Path, Path]:
    """Fig. 8: perceived (Likert, scaled to LUAR range) vs. LUAR self-similarity."""
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    pids = merged["pid"].unique()
    cmap = plt.colormaps.get_cmap("tab20")
    for i, pid in enumerate(pids):
        sub = merged[merged["pid"] == pid]
        ax.scatter(sub["luar"], sub["perc_scaled"], s=18, alpha=0.55, color=cmap(i % 20))
        if len(sub) > 1:
            slope, intercept = np.polyfit(sub["luar"], sub["perc_scaled"], 1)
            xs = np.linspace(sub["luar"].min(), sub["luar"].max(), 5)
            ax.plot(xs, slope * xs + intercept, color=cmap(i % 20), alpha=0.4, lw=0.7)
    # overall best-fit
    s_all, i_all = np.polyfit(merged["luar"], merged["perc_scaled"], 1)
    xs = np.linspace(merged["luar"].min(), merged["luar"].max(), 50)
    ax.plot(xs, s_all * xs + i_all, color="black", lw=2, ls="--", label="overall fit")

    ax.set_xlabel("LUAR cosine sim. to own control text (treatment rows)")
    ax.set_ylabel("Perceived self-similarity\n(scaled from 1-5 Likert)")
    ax.set_title(f"Figure 8 (reproduction): rmcorr r={rmcorr_r:+.3f}, p={rmcorr_p:.4g}")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return _save(fig, out_dir, "fig8_rmcorr")
