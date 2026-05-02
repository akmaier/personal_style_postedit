"""LUAR-MUD embedding wrapper.

We embed each text as a single LUAR "episode" of length 1, max-token-length 512.
This is the convention used by the upstream model card for arbitrary documents.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from . import LUAR_MODEL_ID, LUAR_REVISION


def load_luar(device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        LUAR_MODEL_ID, revision=LUAR_REVISION, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        LUAR_MODEL_ID, revision=LUAR_REVISION, trust_remote_code=True
    )
    model.eval()
    model.to(device)
    return tokenizer, model, device


@torch.inference_mode()
def embed_texts(
    texts: Iterable[str],
    tokenizer=None,
    model=None,
    device: str | None = None,
    batch_size: int = 16,
    max_length: int = 512,
    show_progress: bool = True,
) -> np.ndarray:
    """Return an (N, 512) float32 array of LUAR vectors.

    Each text is treated as a single-document "episode".
    """
    if tokenizer is None or model is None:
        tokenizer, model, device = load_luar(device)
    elif device is None:
        device = next(model.parameters()).device

    texts = [t if isinstance(t, str) and t.strip() else " " for t in texts]
    out = np.zeros((len(texts), 512), dtype=np.float32)
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="LUAR embed", unit="batch")
    for start in iterator:
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        bsz = len(batch)
        # LUAR expects shape (batch_size, episode_length, seq_len)
        input_ids = enc["input_ids"].reshape(bsz, 1, -1).to(device)
        attention_mask = enc["attention_mask"].reshape(bsz, 1, -1).to(device)
        vecs = model(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(vecs, tuple):
            vecs = vecs[0]
        out[start : start + bsz] = vecs.detach().cpu().numpy().astype(np.float32)
    return out


def save_embeddings(path: Path, frame, vecs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pids=np.asarray(frame["pid"].astype(str).to_numpy(), dtype="U64"),
        kinds=np.asarray(frame["kind"].astype(str).to_numpy(), dtype="U16"),
        task_idx=frame["task_idx"].to_numpy().astype(np.int32),
        scenarios=np.asarray(frame["scenario"].astype(str).to_numpy(), dtype="U32"),
        vecs=vecs,
    )


def load_embeddings(path: Path) -> dict:
    z = np.load(path)
    return {
        "pids": z["pids"].astype(str),
        "kinds": z["kinds"].astype(str),
        "task_idx": z["task_idx"].astype(int),
        "scenarios": z["scenarios"].astype(str),
        "vecs": z["vecs"].astype(np.float32),
    }
