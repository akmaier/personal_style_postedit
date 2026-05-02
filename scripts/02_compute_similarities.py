"""Stage 02: compute cosine-similarity tables from cached embeddings."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from personal_style.data import Paths  # noqa: E402
from personal_style.embeddings import load_embeddings  # noqa: E402
from personal_style.similarity import (  # noqa: E402
    EmbeddingTable,
    per_observation_similarities,
    pool_pairwise_similarities,
)


def main() -> None:
    paths = Paths()
    npz = load_embeddings(paths.processed_dir / "embeddings.npz")
    t = EmbeddingTable.from_npz(npz)
    print(f"loaded embeddings: {t.vecs.shape}; kinds={dict(zip(*np.unique(t.kinds, return_counts=True)))}")

    treat, ctrl = per_observation_similarities(t)
    treat.to_parquet(paths.processed_dir / "sim_treatment.parquet", index=False)
    ctrl.to_parquet(paths.processed_dir / "sim_control.parquet", index=False)
    print(f"treatment-row similarities: {len(treat)} -> sim_treatment.parquet")
    print(f"control-row similarities:   {len(ctrl)} -> sim_control.parquet")

    for kind in ("control", "edited", "llm"):
        arr = pool_pairwise_similarities(t, kind)
        np.save(paths.processed_dir / f"pool_pairwise_{kind}.npy", arr)
        print(f"pool_pairwise_{kind}.npy: n_pairs={len(arr)} mean={arr.mean():.3f}")


if __name__ == "__main__":
    main()
