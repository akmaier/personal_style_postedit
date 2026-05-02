"""Merge agent-produced mimic drafts into the canonical cache.

Reads one or more JSON files of `{cache_key: {"text": "..."}}` mappings and
merges them into `data/processed/mimics/<generator>.json`. Every key is
validated against the v1 held-out protocol expected by `scripts/07_generate_mimics.py`.

Usage:
  python scripts/09_save_drafts.py --generator claude-opus-4-7 BATCH.json [BATCH2.json ...]
  python scripts/09_save_drafts.py --generator gpt-5.5         BATCH.json [BATCH2.json ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from personal_style.data import Paths, load_processed  # noqa: E402
from personal_style.llm_mimic import MimicCache, MimicRequest  # noqa: E402

# Pull in the same protocol-aware request builder as scripts/07
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("_gen07", REPO_ROOT / "scripts" / "07_generate_mimics.py")
_gen07 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_gen07)
build_requests = _gen07.build_requests


def build_keys(generator: str) -> dict[str, MimicRequest]:
    paths = Paths()
    obs, _ = load_processed(paths)
    out: dict[str, MimicRequest] = {}
    for req in build_requests(obs):
        out[MimicCache.key(req, generator)] = req
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generator", required=True,
                   help="generator name, e.g. claude-opus-4-7 or gpt-5.5")
    p.add_argument("batches", nargs="+", help="JSON batch files to merge")
    args = p.parse_args()

    paths = Paths()
    cache_path = paths.processed_dir / "mimics" / f"{args.generator}.json"
    cache = MimicCache(cache_path)
    data = cache.load()
    expected = build_keys(args.generator)
    added = 0
    bad = 0
    for arg in args.batches:
        batch = json.loads(Path(arg).read_text())
        for key, payload in batch.items():
            if key not in expected:
                print(f"  [WARN] unknown key {key} in {arg}; skipping")
                bad += 1
                continue
            if isinstance(payload, str):
                text = payload
            else:
                text = payload.get("text", "")
            if not text or not text.strip():
                print(f"  [WARN] empty text for {key}; skipping")
                bad += 1
                continue
            req = expected[key]
            data[key] = {
                "generator": args.generator,
                "pid": req.pid,
                "task_idx": req.task_idx,
                "scenario": req.scenario,
                "text": text.strip(),
            }
            added += 1
    cache.save(data)
    own = sum(1 for v in data.values() if v["generator"] == args.generator)
    print(f"[{args.generator}] merged {added} drafts ({bad} skipped) -> {cache_path}")
    print(f"[{args.generator}] cache now contains {own} drafts")


if __name__ == "__main__":
    main()
