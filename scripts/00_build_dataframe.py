"""Stage 00: tidy logs/*.json into observation + participant parquet files."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from personal_style.data import Paths, save_processed  # noqa: E402


def main() -> None:
    paths = Paths()
    obs, parts = save_processed(paths)
    print(f"observations: {len(obs)} rows -> {paths.observations_parquet}")
    print(f"  conditions: {obs['condition'].value_counts().to_dict()}")
    print(f"  scenarios:  {obs['scenario'].value_counts().to_dict()}")
    print(f"participants: {len(parts)} rows -> {paths.participants_parquet}")


if __name__ == "__main__":
    main()
