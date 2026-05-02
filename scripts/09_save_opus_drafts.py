"""Merge Opus 4.7 drafts produced inline by the agent into the mimic cache.

Reads a JSON file of {cache_key: {"text": "..."}} mappings and merges them
into data/processed/mimics/claude-opus-4-7.json. Validates that every key
matches one of the expected requests under the v1 held-out protocol used by
scripts/07_generate_mimics.py.

Usage:
  python scripts/09_save_opus_drafts.py path/to/batch.json [path/to/another.json ...]
"""
from __future__ import annotations

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

GENERATOR = "claude-opus-4-7"


def build_keys() -> dict[str, MimicRequest]:
    paths = Paths()
    obs, _ = load_processed(paths)
    out: dict[str, MimicRequest] = {}
    for req in build_requests(obs):
        out[MimicCache.key(req, GENERATOR)] = req
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: scripts/09_save_opus_drafts.py BATCH.json [...]")
    paths = Paths()
    cache_path = paths.processed_dir / "mimics" / f"{GENERATOR}.json"
    cache = MimicCache(cache_path)
    data = cache.load()
    expected = build_keys()
    added = 0
    bad = 0
    for arg in sys.argv[1:]:
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
                "generator": GENERATOR,
                "pid": req.pid,
                "task_idx": req.task_idx,
                "scenario": req.scenario,
                "text": text.strip(),
            }
            added += 1
    cache.save(data)
    print(f"[{GENERATOR}] merged {added} drafts ({bad} skipped) -> {cache_path}")
    print(f"[{GENERATOR}] cache now contains {sum(1 for v in data.values() if v['generator'] == GENERATOR)} drafts")


if __name__ == "__main__":
    main()
