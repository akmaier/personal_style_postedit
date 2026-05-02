"""Merge Opus 4.7 drafts produced inline by the agent into the mimic cache.

Reads a JSON file of {cache_key: {"text": "..."}} mappings and merges them
into data/processed/mimics/claude-opus-4-7.json. Validates that every key
matches one of the 324 expected requests.

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

GENERATOR = "claude-opus-4-7"


def build_keys() -> dict[str, MimicRequest]:
    paths = Paths()
    obs, _ = load_processed(paths)
    treat = obs[obs["condition"] == "treatment"]
    ctrl = obs[obs["condition"] == "control"]
    by_pid_ctrl = ctrl.groupby("pid")["final_text"].apply(list).to_dict()
    out: dict[str, MimicRequest] = {}
    for _, r in treat.iterrows():
        samples = tuple(s for s in by_pid_ctrl.get(r["pid"], []) if isinstance(s, str) and s.strip())
        req = MimicRequest(
            pid=str(r["pid"]),
            task_idx=int(r["task_idx"]),
            scenario=str(r["scenario"]),
            details=str(r["details"] or ""),
            style_samples=samples,
        )
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
