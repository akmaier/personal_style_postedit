"""Load study logs into tidy DataFrames.

Each `logs/<pid>.json` file contains:
  - user_info: pid, conditions (6 ints), scenarios (6 strs), start_time
  - responses: dict keyed by str index "0".."5" with one row each
  - pre_survey, mid_survey_1, mid_survey_2, post_survey
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGS_DIR = REPO_ROOT / "logs"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"


@dataclass(frozen=True)
class Paths:
    logs_dir: Path = DEFAULT_LOGS_DIR
    processed_dir: Path = DEFAULT_PROCESSED_DIR

    @property
    def observations_parquet(self) -> Path:
        return self.processed_dir / "observations.parquet"

    @property
    def participants_parquet(self) -> Path:
        return self.processed_dir / "participants.parquet"


def _iter_log_paths(logs_dir: Path) -> list[Path]:
    return sorted(p for p in logs_dir.glob("*.json"))


def _load_log(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_observations(logs_dir: Path = DEFAULT_LOGS_DIR) -> pd.DataFrame:
    """One row per (participant, task). 81*6 = 486 rows expected."""
    rows: list[dict] = []
    for path in _iter_log_paths(logs_dir):
        log = _load_log(path)
        ui = log["user_info"]
        pid = ui["id"]
        scenarios = ui["scenarios"]
        conditions = ui["conditions"]
        for idx_str, resp in log["responses"].items():
            i = int(idx_str)
            if i >= len(conditions):
                # stray response slot beyond the 6 randomized tasks (e.g. tutorial leak)
                continue
            scenario = resp.get("scenario") or (scenarios[i] if i < len(scenarios) else None)
            if scenario is None:
                continue
            condition_int = int(conditions[i])
            condition = "treatment" if condition_int == 1 else "control"
            edits = resp.get("edits", []) or []
            edit_duration_s = None
            if edits:
                ts = []
                for e in edits:
                    t = e.get("timestamp")
                    if t is None or t == "":
                        continue
                    try:
                        ts.append(float(t))
                    except (TypeError, ValueError):
                        continue
                if ts:
                    edit_duration_s = float(max(ts) - min(ts))
            likert = resp.get("likert") or {}
            edit_type = likert.get("edit_type")
            if isinstance(edit_type, list):
                edit_type_str = "|".join(map(str, edit_type))
            else:
                edit_type_str = edit_type
            rows.append(
                {
                    "pid": pid,
                    "task_idx": i,
                    "scenario": scenario,
                    "condition": condition,
                    "model_generation_shown": int(resp.get("model_generation_shown", condition_int) or 0),
                    "details": resp.get("details", ""),
                    "llm_draft": resp.get("model_generation", ""),
                    "final_text": resp.get("final_version", ""),
                    "n_edits": len(edits),
                    "edit_duration_s": edit_duration_s,
                    "start_time": resp.get("start_time"),
                    "submit_details_time": resp.get("submit_details_time"),
                    "submit_final_text_time": resp.get("submit_final_text_time"),
                    # task-level Likert items
                    "likert_original_useable": _to_int(likert.get("original-useable")),
                    "likert_original_capture": _to_int(likert.get("original-capture")),
                    "likert_original_friend": _to_int(likert.get("original-friend")),
                    "likert_original_content": _to_int(likert.get("original-content")),
                    "likert_edited_useable": _to_int(likert.get("edited-useable")),
                    "likert_edited_capture": _to_int(likert.get("edited-capture")),
                    "likert_edited_friend": _to_int(likert.get("edited-friend")),
                    "likert_written_before": _to_int(likert.get("written-before")),
                    "likert_voice_important": _to_int(likert.get("voice-important")),
                    "edit_type": edit_type_str,
                }
            )
    df = pd.DataFrame(rows)
    df["scenario"] = df["scenario"].astype("category")
    df["condition"] = df["condition"].astype("category")
    return df


def _to_int(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except Exception:
            return None


def build_participants(logs_dir: Path = DEFAULT_LOGS_DIR) -> pd.DataFrame:
    """One row per participant: pre/post-survey + demographics."""
    rows: list[dict] = []
    for path in _iter_log_paths(logs_dir):
        log = _load_log(path)
        ui = log["user_info"]
        pre = log.get("pre_survey", {}) or {}
        post = log.get("post_survey", {}) or {}
        m1 = log.get("mid_survey_1", {}) or {}
        m2 = log.get("mid_survey_2", {}) or {}
        # mid surveys carry a `condition` flag; align to treatment / control
        mids = {}
        for m in (m1, m2):
            cond = m.get("condition")
            tag = "treatment" if cond == 1 else ("control" if cond == 0 else None)
            if tag:
                for k in ("mental", "hard", "insecure", "performance", "temporal"):
                    mids[f"tlx_{tag}_{k}"] = _to_int(m.get(k))
        rankings = post.get("rankings")
        if isinstance(rankings, list):
            rankings_str = "|".join(map(str, rankings))
        else:
            rankings_str = rankings
        race = post.get("race")
        if isinstance(race, list):
            race_str = "|".join(map(str, race))
        else:
            race_str = race
        gender = post.get("gender")
        if isinstance(gender, list):
            gender_str = "|".join(map(str, gender))
        else:
            gender_str = gender
        rows.append(
            {
                "pid": ui["id"],
                "start_time": ui.get("start_time"),
                "conditions": "|".join(map(str, ui.get("conditions", []))),
                "scenarios": "|".join(map(str, ui.get("scenarios", []))),
                "pre_conf": _to_int(pre.get("conf")),
                "pre_likely_i": _to_int(pre.get("likely_i")),
                "pre_likely_ni": _to_int(pre.get("likely_ni")),
                "post_conf": _to_int(post.get("conf")),
                "post_likely_i": _to_int(post.get("likely_i")),
                "post_likely_ni": _to_int(post.get("likely_ni")),
                "future_pref": post.get("future_pref"),
                "rankings": rankings_str,
                "gender": gender_str,
                "trans": post.get("trans"),
                "hisp": post.get("hisp"),
                "race": race_str,
                "age": _to_int(post.get("age")),
                "loe": post.get("loe"),
                "gender_table": _to_int(post.get("gender_table")),
                "race_table": _to_int(post.get("race_table")),
                "age_table": _to_int(post.get("age_table")),
                "loe_table": _to_int(post.get("loe_table")),
                **mids,
            }
        )
    return pd.DataFrame(rows)


def save_processed(paths: Paths = Paths()) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    obs = build_observations(paths.logs_dir)
    parts = build_participants(paths.logs_dir)
    obs.to_parquet(paths.observations_parquet, index=False)
    parts.to_parquet(paths.participants_parquet, index=False)
    return obs, parts


def load_processed(paths: Paths = Paths()) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(paths.observations_parquet),
        pd.read_parquet(paths.participants_parquet),
    )


def texts_for_embedding(obs: pd.DataFrame) -> pd.DataFrame:
    """Return a long-format DataFrame: one row per text to embed.

    kinds:
      - control: final_text from control rows
      - llm:     llm_draft from treatment rows
      - edited:  final_text from treatment rows
    """
    parts = []
    ctrl = obs[obs["condition"] == "control"][["pid", "task_idx", "scenario", "final_text"]].copy()
    ctrl = ctrl.rename(columns={"final_text": "text"})
    ctrl["kind"] = "control"
    parts.append(ctrl)

    tr = obs[obs["condition"] == "treatment"]
    llm = tr[["pid", "task_idx", "scenario", "llm_draft"]].rename(columns={"llm_draft": "text"}).copy()
    llm["kind"] = "llm"
    parts.append(llm)

    edited = tr[["pid", "task_idx", "scenario", "final_text"]].rename(columns={"final_text": "text"}).copy()
    edited["kind"] = "edited"
    parts.append(edited)

    out = pd.concat(parts, ignore_index=True)
    out["scenario"] = out["scenario"].astype(str)
    return out
