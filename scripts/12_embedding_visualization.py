"""Stage 12: 2-D visualization of the LUAR embedding space colored by approach.

Purpose: give the reader a geometric sense of *why* the detection AUCs in
Section 7 of the paper come out where they do. Five colored point clouds
in the same 2-D projection show how each approach sits relative to the
human-control distribution.

Method:
    - X: every embedding vector that participates in Section 7
      (162 human controls + 4 x 324 = 1458 AI vectors = 1620 total).
    - Project to 2-D with PCA(2) and t-SNE(2). Both are released so the
      reader can see that the qualitative story is not an artefact of
      one projection method.
    - Color by approach. Edge color = none for AI samples, black for the
      human reference points so they pop visually.

Outputs:
    figures/fig12_embedding_pca.{pdf,png}
    figures/fig12_embedding_tsne.{pdf,png}
    data/processed/embedding_2d.parquet  (PCA + t-SNE coords for every vector)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

from personal_style.data import Paths  # noqa: E402
from personal_style.embeddings import load_embeddings  # noqa: E402
from personal_style.similarity import EmbeddingTable, participant_kind_indices  # noqa: E402

# Re-use the script-11 builder to keep the dataset definition in one place.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_d11", REPO_ROOT / "scripts" / "11_detection_experiment.py")
_d11 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_d11)

SEED = 20260101

APPROACHES = [
    ("Human (control)", "control"),
    ("o4-mini draft", "llm"),
    ("Human post-edit", "edited"),
    ("Claude Opus 4.7", "claude-opus-4-7"),
    ("GPT-5.5", "gpt-5.5"),
]

PALETTE = {
    "Human (control)":  "#1f78b4",
    "o4-mini draft":    "#d95f02",
    "Human post-edit":  "#1b9e77",
    "Claude Opus 4.7":  "#e7298a",
    "GPT-5.5":          "#7570b3",
}

MARKER = {
    "Human (control)": "^",
    "o4-mini draft":   "o",
    "Human post-edit": "o",
    "Claude Opus 4.7": "o",
    "GPT-5.5":         "o",
}


def assemble_full_table(paths: Paths, base: EmbeddingTable
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack ALL the embeddings used in Section 7 into one X with labels.

    Human (control)  : 162 samples = 81 pids x 2 controls
    o4-mini draft    : 324 samples = 81 pids x 4 LLM drafts
    Human post-edit  : 324 samples = 81 pids x 4 post-edited treatment texts
    Claude Opus 4.7  : 324 samples = 81 pids x 4 mimic drafts
    GPT-5.5          : 324 samples = 81 pids x 4 mimic drafts
    """
    pki = participant_kind_indices(base)
    pids_with_both = [
        pid for pid in sorted(set(base.pids.tolist()))
        if len(pki.get((pid, "control"), [])) >= 2
    ]

    X_list: list[np.ndarray] = []
    L_list: list[str] = []  # approach label per row
    P_list: list[str] = []  # pid per row

    # Human controls
    for pid in pids_with_both:
        for ei in pki.get((pid, "control"), []):
            X_list.append(base.vecs[ei])
            L_list.append("Human (control)")
            P_list.append(pid)

    # o4-mini and human-edit (in the base cache under kinds 'llm' and 'edited')
    for label, kind in [("o4-mini draft", "llm"), ("Human post-edit", "edited")]:
        for pid in pids_with_both:
            for ei in pki.get((pid, kind), []):
                X_list.append(base.vecs[ei])
                L_list.append(label)
                P_list.append(pid)

    # Mimic generators (from the mimic_embeddings cache)
    for label, gen in [("Claude Opus 4.7", "claude-opus-4-7"),
                       ("GPT-5.5", "gpt-5.5")]:
        mimic = _d11.load_mimic_vectors(paths, gen)
        for pid in pids_with_both:
            for (p, t), v in mimic.items():
                if p != pid:
                    continue
                X_list.append(v)
                L_list.append(label)
                P_list.append(pid)

    X = np.stack(X_list).astype(np.float32)
    L = np.array(L_list, dtype=object)
    P = np.array(P_list, dtype=object)
    return X, L, P


def plot_2d(X2: np.ndarray, L: np.ndarray, title: str,
            out_path_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    # Plot AI clouds first so the human triangles sit on top.
    plot_order = ["o4-mini draft", "Human post-edit",
                  "Claude Opus 4.7", "GPT-5.5", "Human (control)"]
    for label in plot_order:
        mask = L == label
        if not mask.any():
            continue
        col = PALETTE[label]
        marker = MARKER[label]
        if label == "Human (control)":
            ax.scatter(X2[mask, 0], X2[mask, 1],
                       s=28, color=col, marker=marker,
                       edgecolor="black", linewidth=0.5, alpha=0.95,
                       label=f"{label} (n={int(mask.sum())})", zorder=4)
        else:
            ax.scatter(X2[mask, 0], X2[mask, 1],
                       s=14, color=col, marker=marker,
                       alpha=0.55, linewidth=0,
                       label=f"{label} (n={int(mask.sum())})", zorder=3)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=9, frameon=True, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path_base.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    paths = Paths()
    fig_dir = REPO_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    z = load_embeddings(paths.processed_dir / "embeddings.npz")
    base = EmbeddingTable(z["pids"], z["kinds"], z["task_idx"],
                          z["scenarios"], z["vecs"])
    X, L, P = assemble_full_table(paths, base)
    print(f"total vectors: {len(X)}; per-approach: "
          + ", ".join(f"{lbl}={int((L == lbl).sum())}"
                      for lbl in [a for a, _ in APPROACHES]))

    print("running PCA(n_components=2) ...")
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X)

    print("running t-SNE(n_components=2, perplexity=30) ...")
    # `n_iter` was renamed to `max_iter` in scikit-learn 1.5+.
    tsne = TSNE(n_components=2, perplexity=30, init="pca",
                random_state=SEED, learning_rate="auto", max_iter=1000)
    X_tsne = tsne.fit_transform(X)

    df = pd.DataFrame({
        "pid": P, "approach": L,
        "pca_x": X_pca[:, 0], "pca_y": X_pca[:, 1],
        "tsne_x": X_tsne[:, 0], "tsne_y": X_tsne[:, 1],
    })
    df.to_parquet(paths.processed_dir / "embedding_2d.parquet", index=False)

    plot_2d(X_pca, L,
            "Figure 12a: PCA(2) of LUAR embeddings, colored by approach",
            fig_dir / "fig12a_embedding_pca")
    plot_2d(X_tsne, L,
            "Figure 12b: t-SNE(2) of LUAR embeddings, colored by approach",
            fig_dir / "fig12b_embedding_tsne")

    var_ratio = pca.explained_variance_ratio_
    print(f"PCA variance explained by first 2 PCs: "
          f"{var_ratio[0]:.3f} + {var_ratio[1]:.3f} = "
          f"{var_ratio[:2].sum():.3f}")
    print(f"figures/fig12a_embedding_pca.{{pdf,png}} written")
    print(f"figures/fig12b_embedding_tsne.{{pdf,png}} written")


if __name__ == "__main__":
    main()
