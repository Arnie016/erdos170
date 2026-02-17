from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse_ruler.e_hunt import run_e_hunt_from_config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E-hunt delete-and-repair campaign")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to E-hunt JSON config",
    )
    args = parser.parse_args()

    summary = run_e_hunt_from_config_path(args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
