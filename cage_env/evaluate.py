from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

try:
    from cage_env.controller import SignalResponder
    from cage_env.env import OscillationChamberEnv
    from cage_env.session_log import SessionLogger
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from cage_env.controller import SignalResponder
    from cage_env.env import OscillationChamberEnv
    from cage_env.session_log import SessionLogger


def build_summary(final_info: dict) -> dict:
    metrics = final_info.get("metrics", {})
    signals = final_info.get("signals", {})

    attraction_scores = signals.get("attraction_scores", {})
    best_attraction = max(attraction_scores.items(), key=lambda item: item[1], default=(None, 0.0))

    return {
        "revisit_count": {system_id: data.get("revisit_count", 0) for system_id, data in metrics.items()},
        "dwell_time": {system_id: data.get("dwell_time", 0.0) for system_id, data in metrics.items()},
        "phase_alignment": {system_id: data.get("phase_alignment", 0.0) for system_id, data in metrics.items()},
        "synchronization_attempts": {system_id: data.get("synchronization_attempts", 0) for system_id, data in metrics.items()},
        "attraction_preference": {"system_id": best_attraction[0], "score": best_attraction[1]},
    }


def run_session(seed: int, steps: int, log_path: Path) -> dict:
    env = OscillationChamberEnv()
    controller = SignalResponder(seed=seed)
    logger = SessionLogger(log_path)

    obs, info = env.reset(seed=seed)
    final_info = info
    completed_steps = 0

    for step in range(steps):
        action = controller.choose_action(info)
        obs, reward, terminated, truncated, info = env.step(action)
        logger.write_step(
            {
                "type": "step",
                "seed": seed,
                "step": step,
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "obs": obs,
                "info": info,
            }
        )
        final_info = info
        completed_steps = step + 1
        if terminated or truncated:
            break

    summary = build_summary(final_info)
    logger.write_summary({"seed": seed, "steps": completed_steps, "summary": summary})
    return {"seed": seed, "steps": completed_steps, "log_path": str(log_path), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic Oscillation Chamber session.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--log-dir", type=Path, default=Path("cage_env/logs"))
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"session_{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
    result = run_session(args.seed, args.steps, log_path)

    print(f"seed={result['seed']} steps={result['steps']}")
    print(f"log={result['log_path']}")
    for name, value in result["summary"].items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()