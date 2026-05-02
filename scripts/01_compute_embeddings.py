"""Stage 01: compute LUAR-MUD embeddings for control / llm / edited texts."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from personal_style.data import Paths, load_processed, texts_for_embedding  # noqa: E402
from personal_style.embeddings import embed_texts, load_luar, save_embeddings  # noqa: E402


def main() -> None:
    paths = Paths()
    obs, _ = load_processed(paths)
    frame = texts_for_embedding(obs)
    print(f"texts to embed: {len(frame)} ({frame['kind'].value_counts().to_dict()})")

    out_path = paths.processed_dir / "embeddings.npz"
    if out_path.exists():
        print(f"embeddings already exist at {out_path}; skipping (delete to recompute)")
        return

    tokenizer, model, device = load_luar()
    print(f"loaded LUAR-MUD on {device}")
    vecs = embed_texts(frame["text"].tolist(), tokenizer, model, device, batch_size=16)
    save_embeddings(out_path, frame, vecs)
    print(f"saved {vecs.shape} embeddings to {out_path}")


if __name__ == "__main__":
    torch.manual_seed(20260101)
    main()
