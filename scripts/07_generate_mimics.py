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
    """One request per treatment task using the leakage-free held-out protocol.

    Each participant has exactly 2 control texts. We deterministically assign
    the one with the *lower* task_idx as the demo (shown to the model) and the
    other as the held-out evaluation target (NOT shown). The held-out task_idx
    is stored on the MimicRequest so scripts/08_compare_mimics.py can score the
    mimic against the right vector.
    """
    treat = obs[obs["condition"] == "treatment"]
    ctrl = obs[obs["condition"] == "control"].sort_values(["pid", "task_idx"])
    # Map each pid -> [(task_idx, final_text), (task_idx, final_text)]
    pid_to_controls: dict[str, list[tuple[int, str]]] = {}
    for pid, sub in ctrl.groupby("pid"):
        pid_to_controls[str(pid)] = [
            (int(r.task_idx), str(r.final_text))
            for r in sub.itertuples(index=False)
            if isinstance(r.final_text, str) and r.final_text.strip()
        ]

    requests: list[MimicRequest] = []
    for _, r in treat.iterrows():
        controls = pid_to_controls.get(str(r["pid"]), [])
        if len(controls) < 2:
            # Skip: cannot run the held-out protocol with fewer than 2 controls.
            continue
        # Sorted ascending by task_idx; demo is the earlier control, held-out is the later.
        demo_idx, demo_text = controls[0]
        held_out_idx, _held_out_text = controls[1]
        requests.append(
            MimicRequest(
                pid=str(r["pid"]),
                task_idx=int(r["task_idx"]),
                scenario=str(r["scenario"]),
                details=str(r["details"] or ""),
                style_samples=(demo_text,),
                held_out_task_idx=held_out_idx,
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
