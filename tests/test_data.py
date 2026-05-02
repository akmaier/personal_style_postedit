"""Schema invariants: 81 participants, 6 tasks each, 324 treatment + 162 control."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from personal_style.data import build_observations, build_participants


def test_n_participants():
    parts = build_participants()
    assert len(parts) == 81


def test_observation_counts():
    obs = build_observations()
    assert len(obs) == 81 * 6  # 486
    counts = obs["condition"].value_counts().to_dict()
    assert counts["treatment"] == 324
    assert counts["control"] == 162
