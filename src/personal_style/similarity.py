"""Cosine-similarity tables built from LUAR embeddings."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) or 1.0)
    b = b / (np.linalg.norm(b) or 1.0)
    return float(np.dot(a, b))


@dataclass
class EmbeddingTable:
    pids: np.ndarray
    kinds: np.ndarray
    task_idx: np.ndarray
    scenarios: np.ndarray
    vecs: np.ndarray

    @classmethod
    def from_npz(cls, d: dict) -> "EmbeddingTable":
        return cls(d["pids"], d["kinds"], d["task_idx"], d["scenarios"], d["vecs"])

    def index(self) -> dict[tuple, int]:
        out: dict[tuple, int] = {}
        for i, (p, k, t) in enumerate(zip(self.pids, self.kinds, self.task_idx)):
            out[(str(p), str(k), int(t))] = i
        return out

    def by_pid_kind(self) -> dict[tuple, list[int]]:
        out: dict[tuple, list[int]] = defaultdict(list)
        for i, (p, k) in enumerate(zip(self.pids, self.kinds)):
            out[(str(p), str(k))].append(i)
        return out

    def vec(self, i: int) -> np.ndarray:
        return self.vecs[i]


def participant_kind_centroid(t: EmbeddingTable, pid: str, kind: str) -> np.ndarray | None:
    bk = t.by_pid_kind()
    idx = bk.get((pid, kind), [])
    if not idx:
        return None
    v = t.vecs[idx]
    return normalize(v.mean(axis=0))


def participant_kind_indices(t: EmbeddingTable):
    """Return dict[(pid, kind)] -> list[int] of row indices in t."""
    out: dict[tuple, list[int]] = {}
    for i, (p, k) in enumerate(zip(t.pids, t.kinds)):
        out.setdefault((str(p), str(k)), []).append(i)
    return out


def mean_cos_to_set(v: np.ndarray, vecs: np.ndarray) -> float:
    """Mean of pairwise cosine similarities between v and each row of vecs."""
    if len(vecs) == 0:
        return float("nan")
    vn = v / (np.linalg.norm(v) or 1.0)
    rn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    return float((rn @ vn).mean())


def per_observation_similarities(t: EmbeddingTable) -> pd.DataFrame:
    """For every treatment observation, compute the similarity values used in §6.

    Columns:
      pid, task_idx, scenario,
      sim_edited_llm,
      sim_edited_control_self, sim_edited_control_other_mean,
      sim_edited_edited_other_mean,
      sim_llm_control_self, sim_llm_control_other_mean,
      sim_llm_llm_other_mean,
      sim_control_self_other_mean (only for control rows)
    """
    idx = t.index()
    bk = t.by_pid_kind()

    pids = sorted(set(t.pids.tolist()))
    # Per-pid centroid per kind (used for own-self similarity).
    cent_self_control = {p: participant_kind_centroid(t, p, "control") for p in pids}
    # Per-pid raw vector lists per kind (used for "mean cosine to other participants' texts").
    pki = participant_kind_indices(t)
    pid_ctrl_vecs = {p: t.vecs[pki.get((p, "control"), [])] for p in pids}
    pid_llm_vecs = {p: t.vecs[pki.get((p, "llm"), [])] for p in pids}
    pid_edit_vecs = {p: t.vecs[pki.get((p, "edited"), [])] for p in pids}

    rows_treat: list[dict] = []
    rows_ctrl: list[dict] = []

    treat_keys = [(p, k, ti) for (p, k, ti) in idx if k == "edited"]
    ctrl_keys = [(p, k, ti) for (p, k, ti) in idx if k == "control"]

    for p, _, ti in treat_keys:
        i_e = idx[(p, "edited", ti)]
        i_l = idx[(p, "llm", ti)]
        v_e = t.vecs[i_e]
        v_l = t.vecs[i_l]

        # Same-task LLM draft <-> edited
        s_e_l = cosine(v_e, v_l)
        # Self control centroid
        s_l_self = cosine(v_l, cent_self_control[p]) if cent_self_control[p] is not None else np.nan
        s_e_self = cosine(v_e, cent_self_control[p]) if cent_self_control[p] is not None else np.nan

        # "Similarity to other participants' X" = average over OTHER participants of
        # the within-participant mean cosine (so each other-participant contributes
        # equally regardless of how many X-texts they wrote).
        other_pids = [q for q in pids if q != p]

        def _avg_per_participant(v: np.ndarray, vec_dict: dict) -> float:
            scores = []
            for q in other_pids:
                vs = vec_dict[q]
                if len(vs) == 0:
                    continue
                scores.append(mean_cos_to_set(v, vs))
            return float(np.mean(scores)) if scores else float("nan")

        s_e_ctrl_other = _avg_per_participant(v_e, pid_ctrl_vecs)
        s_l_ctrl_other = _avg_per_participant(v_l, pid_ctrl_vecs)
        s_e_edit_other = _avg_per_participant(v_e, pid_edit_vecs)
        s_e_llm_other = _avg_per_participant(v_e, pid_llm_vecs)
        s_l_llm_other = _avg_per_participant(v_l, pid_llm_vecs)

        rows_treat.append(
            {
                "pid": p,
                "task_idx": int(ti),
                "scenario": str(t.scenarios[i_e]),
                "sim_edited_llm_self": s_e_l,
                "sim_edited_control_self": s_e_self,
                "sim_edited_control_other_mean": float(s_e_ctrl_other),
                "sim_edited_edited_other_mean": float(s_e_edit_other),
                "sim_edited_llm_other_mean": float(s_e_llm_other),
                "sim_llm_control_self": s_l_self,
                "sim_llm_control_other_mean": float(s_l_ctrl_other),
                "sim_llm_llm_other_mean": float(s_l_llm_other),
            }
        )

    for p, _, ti in ctrl_keys:
        i_c = idx[(p, "control", ti)]
        v_c = t.vecs[i_c]
        other_pids = [q for q in pids if q != p]
        ctrl_other_vecs = np.concatenate(
            [pid_ctrl_vecs[q] for q in other_pids if len(pid_ctrl_vecs[q])]
        )
        s_c_other = mean_cos_to_set(v_c, ctrl_other_vecs)
        # this control text vs participant's *other* control text(s)
        own_idx = [j for j in bk[(p, "control")] if j != i_c]
        if own_idx:
            s_c_self = mean_cos_to_set(v_c, t.vecs[own_idx])
        else:
            s_c_self = np.nan
        rows_ctrl.append(
            {
                "pid": p,
                "task_idx": int(ti),
                "scenario": str(t.scenarios[i_c]),
                "sim_control_self_other_mean": float(s_c_other),
                "sim_control_self_self": float(s_c_self),
            }
        )

    return pd.DataFrame(rows_treat), pd.DataFrame(rows_ctrl)


def pool_pairwise_similarities(t: EmbeddingTable, kind: str) -> np.ndarray:
    """Return all pairwise cosine similarities between texts of the given `kind`,
    where the two texts come from *different* participants (used for Fig. 6)."""
    mask = t.kinds == kind
    vecs = t.vecs[mask]
    pids = t.pids[mask]
    if len(vecs) == 0:
        return np.array([])
    vn = normalize(vecs)
    sim = vn @ vn.T
    out = []
    n = len(pids)
    for i in range(n):
        for j in range(i + 1, n):
            if str(pids[i]) != str(pids[j]):
                out.append(sim[i, j])
    return np.array(out, dtype=float)
