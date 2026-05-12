from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from cage_env.evaluate import run_session
from cage_env.visualize import run_visualizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Oscillation Chamber in order: evaluate, then interact.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--log-dir", type=Path, default=Path("cage_env/logs"))
    parser.add_argument("--skip-ui", action="store_true", help="Run the deterministic session only.")
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"session_{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"

    print("[1/2] running deterministic evaluation...")
    result = run_session(seed=args.seed, steps=args.steps, log_path=log_path)
    print(f"saved log: {result['log_path']}")
    for name, value in result["summary"].items():
        print(f"{name}: {value}")

    if args.skip_ui:
        return

    print("[2/2] opening live visualizer...")
    run_visualizer(seed=args.seed)


if __name__ == "__main__":
    main()