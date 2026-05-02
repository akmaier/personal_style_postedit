"""Stage 07: ask GPT 5.5 / Claude Opus 4.7 to generate style-mimicking drafts.

Per treatment task, we feed the model:
  - the same writing prompt the original GPT-o4-mini draft saw (paper §4.1)
  - the same planning details
  - the participant's two control texts as style-demonstration examples

Generated drafts are cached to data/processed/mimics/<generator>.json so reruns
are idempotent and partial failures are recoverable.

Usage:
  python scripts/07_generate_mimics.py --generators stub
  python scripts/07_generate_mimics.py --generators gpt-5.5 claude-opus-4-7
  python scripts/07_generate_mimics.py --generators gpt-5.5 --limit 5  # smoke test
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from personal_style.data import Paths, load_processed  # noqa: E402
from personal_style.llm_mimic import MimicCache, MimicRequest, make_generator  # noqa: E402


def build_requests(obs: pd.DataFrame) -> list[MimicRequest]:
    """One request per treatment task. Style samples = that participant's control texts."""
    treat = obs[obs["condition"] == "treatment"]
    ctrl = obs[obs["condition"] == "control"]
    by_pid_ctrl = ctrl.groupby("pid")["final_text"].apply(list).to_dict()

    requests: list[MimicRequest] = []
    for _, r in treat.iterrows():
        samples = by_pid_ctrl.get(r["pid"], [])
        samples = tuple(s for s in samples if isinstance(s, str) and s.strip())
        requests.append(
            MimicRequest(
                pid=str(r["pid"]),
                task_idx=int(r["task_idx"]),
                scenario=str(r["scenario"]),
                details=str(r["details"] or ""),
                style_samples=samples,
            )
        )
    return requests


def run_for_generator(name: str, requests: list[MimicRequest], cache_path: Path,
                      limit: int | None = None, sleep_s: float = 0.0) -> None:
    cache = MimicCache(cache_path)
    data = cache.load()
    gen = make_generator(name)

    targets = requests[:limit] if limit else requests
    pbar = tqdm(targets, desc=f"generate[{name}]", unit="req")
    saved_recently = 0
    for req in pbar:
        key = MimicCache.key(req, gen.name)
        if key in data:
            continue
        try:
            text = gen.generate(req)
        except Exception as e:
            pbar.write(f"  [error] {gen.name} pid={req.pid} task={req.task_idx}: {e}")
            continue
        data[key] = {
            "generator": gen.name,
            "pid": req.pid,
            "task_idx": req.task_idx,
            "scenario": req.scenario,
            "text": text,
        }
        saved_recently += 1
        if sleep_s:
            time.sleep(sleep_s)
        if saved_recently % 25 == 0:
            cache.save(data)

    cache.save(data)
    print(f"[{name}] cached {sum(1 for v in data.values() if v['generator'] == gen.name)} drafts -> {cache_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generators", nargs="+", required=True,
                   help="any subset of: stub gpt-5.5 claude-opus-4-7")
    p.add_argument("--limit", type=int, default=None,
                   help="generate at most N drafts per model (for smoke testing)")
    p.add_argument("--sleep", type=float, default=0.0, help="seconds between requests")
    args = p.parse_args()

    paths = Paths()
    obs, _ = load_processed(paths)
    requests = build_requests(obs)
    print(f"built {len(requests)} mimic requests")

    out_dir = paths.processed_dir / "mimics"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.generators:
        # gentle env-var preflight so we fail fast and clearly.
        env_required = {
            "gpt-5.5": "OPENAI_API_KEY",
            "claude-opus-4-7": "ANTHROPIC_API_KEY",
        }
        env = env_required.get(name)
        if env and not os.environ.get(env):
            print(f"[{name}] SKIP: {env} is not set. "
                  "Add it in Cursor Dashboard -> Cloud Agents -> Secrets.")
            continue
        run_for_generator(name, requests, out_dir / f"{name}.json",
                          limit=args.limit, sleep_s=args.sleep)


if __name__ == "__main__":
    main()
