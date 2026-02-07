from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse_ruler.stage1 import run_stage1_scaffold


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage1 scaffold repair suite.")
    parser.add_argument("config", type=Path, help="Path to suite JSON config")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    result = run_stage1_scaffold(config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
