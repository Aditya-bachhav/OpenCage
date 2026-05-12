from __future__ import annotations

import argparse
from pathlib import Path

from cage_env.experiment import compare_controllers
from cage_env.visualize import run_visualizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Oscillation Chamber evaluation workflow.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--log-dir", type=Path, default=Path("cage_env/logs"))
    parser.add_argument("--replay", type=Path, help="Replay a saved session log instead of live visualization.")
    parser.add_argument("--skip-ui", action="store_true", help="Skip the visualizer after evaluation.")
    parser.add_argument("--long-run", action="store_true", help="Run long-run preference persistence experiment (5000 steps per seed).")
    parser.add_argument("--num-seeds", type=int, default=10, help="Number of seeds for long-run experiment.")
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)

    if args.long_run:
        print(f"[LONG-RUN] Running preference persistence experiment: {args.num_seeds} seeds × 5000 steps")
        seeds = list(range(100, 100 + args.num_seeds))
        comparison = compare_controllers(seeds=seeds, steps=5000, log_dir=args.log_dir)
        for controller_name, payload in comparison.items():
            print(f"[{controller_name}] {payload['aggregate']}")
        return

    print("[1/2] running controller comparison...")
    comparison = compare_controllers(seeds=args.seeds, steps=args.steps, log_dir=args.log_dir)
    for controller_name, payload in comparison.items():
        print(f"[{controller_name}] {payload['aggregate']}")

    if args.skip_ui:
        return

    print("[2/2] opening live visualizer...")
    if args.replay is not None:
        run_visualizer(replay=args.replay)
    else:
        run_visualizer(seed=args.seeds[0])


if __name__ == "__main__":
    main()