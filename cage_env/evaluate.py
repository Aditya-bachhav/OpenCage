from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from cage_env.experiment import compare_controllers, run_multi_seed, run_session
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from cage_env.experiment import compare_controllers, run_multi_seed, run_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed Oscillation Chamber experiments.")
    parser.add_argument("--controller", choices=("emergent", "random", "heuristic", "all"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--log-dir", type=Path, default=Path("cage_env/logs"))
    args = parser.parse_args()

    if args.controller == "all":
        comparison = compare_controllers(seeds=args.seeds, steps=args.steps, log_dir=args.log_dir)
        for controller_name, payload in comparison.items():
            print(f"[{controller_name}]")
            print(f"aggregate: {payload['aggregate']}")
    else:
        runs = run_multi_seed(controller_name=args.controller, seeds=args.seeds, steps=args.steps, log_dir=args.log_dir)
        for run in runs:
            print(f"seed={run['seed']} steps={run['steps']} log={run['log_path']}")
            for name, value in run["summary"].items():
                print(f"{name}: {value}")


if __name__ == "__main__":
    main()