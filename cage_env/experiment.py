from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

try:
    from cage_env.controller import SignalResponder
    from cage_env.env import ACTION_NAMES, OscillationChamberEnv
    from cage_env.session_log import SessionLogger
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from cage_env.controller import SignalResponder
    from cage_env.env import ACTION_NAMES, OscillationChamberEnv
    from cage_env.session_log import SessionLogger


CONTROLLER_NAMES = ("emergent", "random", "heuristic")
SYSTEM_IDS = ("pendulum", "spring", "plate", "wheel", "wave")


@dataclass
class RandomController:
    seed: int

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def choose_action(self, info: dict) -> int:
        return self.rng.randrange(len(ACTION_NAMES))


@dataclass
class WeakHeuristicController:
    seed: int

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def choose_action(self, info: dict) -> int:
        agent = info.get("agent") or {}
        systems = info.get("systems") or []
        if not systems:
            return 0

        nearest = min(systems, key=lambda system: math.hypot(system["x"] - agent.get("x", 0.0), system["y"] - agent.get("y", 0.0)))
        dx = nearest["x"] - agent.get("x", 0.0)
        dy = nearest["y"] - agent.get("y", 0.0)
        dist = math.hypot(dx, dy)

        if dist < 0.7:
            return 5
        if abs(dx) > abs(dy):
            return 1 if dx < 0 else 2
        return 4 if dy < 0 else 3


def make_controller(name: str, seed: int):
    if name == "emergent":
        return SignalResponder(seed=seed)
    if name == "random":
        return RandomController(seed=seed)
    if name == "heuristic":
        return WeakHeuristicController(seed=seed)
    raise ValueError(f"Unknown controller: {name}")


def load_session_rows(log_path: Path) -> list[dict]:
    return SessionLogger(log_path).read_all()


def _entropy_from_counts(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        probability = value / total
        entropy -= probability * math.log2(probability)
    return round(entropy, 4)


def _longest_streak(values: Iterable[str | None]) -> int:
    best = current = 0
    last = object()
    for value in values:
        if value is not None and value == last:
            current += 1
        else:
            current = 1 if value is not None else 0
            last = value
        best = max(best, current)
    return best


def _entropy_from_counts(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        probability = value / total
        entropy -= probability * math.log2(probability)
    return round(entropy, 4)


def _summarize_preference_evolution(preference_history: list[dict]) -> dict:
    """Summarize how preferences evolved across the session."""
    if not preference_history:
        return {}
    
    entropies = [p.get("entropy", 0.0) for p in preference_history]
    top_weights = [p.get("top_preference", {}).get("weight", 0.0) for p in preference_history]
    top_systems = [p.get("top_preference", {}).get("system_id") for p in preference_history]
    
    preference_changes = 0
    for i in range(1, len(top_systems)):
        if top_systems[i] != top_systems[i - 1] and top_systems[i] is not None:
            preference_changes += 1
    
    return {
        "mean_entropy": round(sum(entropies) / len(entropies), 4) if entropies else 0.0,
        "min_entropy": round(min(entropies), 4) if entropies else 0.0,
        "max_entropy": round(max(entropies), 4) if entropies else 0.0,
        "mean_top_weight": round(sum(top_weights) / len(top_weights), 4) if top_weights else 0.0,
        "max_top_weight": round(max(top_weights), 4) if top_weights else 0.0,
        "preference_changes": preference_changes,
        "final_top_system": top_systems[-1] if top_systems else None,
    }


def summarize_session(rows: list[dict]) -> dict:
    step_rows = [row for row in rows if row.get("type") == "step"]
    if not step_rows:
        return {}

    final_info = step_rows[-1]["info"]
    signals_series = [row["info"].get("signals", {}) for row in step_rows]
    nearest_series = [signals.get("nearest_system_id") for signals in signals_series]
    attraction_series = [signals.get("attraction_scores", {}) for signals in signals_series]
    final_metrics = final_info.get("metrics", {})
    final_signals = final_info.get("signals", {})

    revisit_count = {system_id: data.get("revisit_count", 0) for system_id, data in final_metrics.items()}
    dwell_time = {system_id: data.get("dwell_time", 0.0) for system_id, data in final_metrics.items()}
    phase_alignment = {system_id: data.get("phase_alignment", 0.0) for system_id, data in final_metrics.items()}
    synchronization_attempts = {system_id: data.get("synchronization_attempts", 0) for system_id, data in final_metrics.items()}

    attraction_scores = final_signals.get("attraction_scores", {})
    best_attraction = max(attraction_scores.items(), key=lambda item: item[1], default=(None, 0.0))

    system_counts = Counter(system_id for system_id in nearest_series if system_id)
    repeated_visits = sum(max(data.get("visits", 0) - 1, 0) for data in final_metrics.values())
    first_time_visits = sum(1 for data in final_metrics.values() if data.get("visits", 0) > 0)

    stability: dict[str, float] = {}
    for system_id in SYSTEM_IDS:
        series = [scores.get(system_id, 0.0) for scores in attraction_series]
        if not series:
            stability[system_id] = 0.0
            continue
        mean = sum(series) / len(series)
        variance = sum((value - mean) ** 2 for value in series) / len(series)
        stddev = math.sqrt(variance)
        stability[system_id] = round(max(0.0, 1.0 - (stddev / (mean + 1e-6))), 4)

    summary = {
        "revisit_count": revisit_count,
        "dwell_time": dwell_time,
        "attraction_stability": stability,
        "phase_alignment": phase_alignment,
        "synchronization_attempts": synchronization_attempts,
        "exploration_entropy": _entropy_from_counts(system_counts),
        "longest_continuous_attraction_streak": _longest_streak(nearest_series),
        "repeated_to_first_visit_ratio": round(repeated_visits / max(first_time_visits, 1), 4),
        "attraction_preference": {"system_id": best_attraction[0], "score": best_attraction[1]},
    }
    return summary


def run_session(seed: int, steps: int, log_path: Path, controller_name: str = "emergent") -> dict:
    env = OscillationChamberEnv()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)
    controller = make_controller(controller_name, seed)
    
    # Pass preference memory to emergent controller
    if controller_name == "emergent":
        controller.preference_memory = env.preference_memory
    
    logger = SessionLogger(log_path)

    obs, info = env.reset(seed=seed)
    final_info = info
    completed_steps = 0
    preference_evolution = []

    for step in range(steps):
        action = controller.choose_action(info)
        obs, reward, terminated, truncated, info = env.step(action)
        logger.write_step(
            {
                "type": "step",
                "seed": seed,
                "controller": controller_name,
                "step": step,
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "obs": obs,
                "info": info,
            }
        )
        preference_evolution.append(info.get("preferences", {}))
        final_info = info
        completed_steps = step + 1
        if terminated or truncated:
            break

    summary = summarize_session(logger.read_all())
    summary["preference_evolution"] = _summarize_preference_evolution(preference_evolution)
    logger.write_summary({"seed": seed, "controller": controller_name, "steps": completed_steps, "summary": summary})
    return {"seed": seed, "steps": completed_steps, "controller": controller_name, "log_path": str(log_path), "summary": summary}


def run_multi_seed(controller_name: str, seeds: Iterable[int], steps: int, log_dir: Path) -> list[dict]:
    results = []
    log_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        log_path = log_dir / f"{controller_name}_seed_{seed}.jsonl"
        results.append(run_session(seed=seed, steps=steps, log_path=log_path, controller_name=controller_name))
    return results


def aggregate_runs(results: list[dict]) -> dict:
    if not results:
        return {}

    preferred_counts = Counter(result["summary"]["attraction_preference"]["system_id"] for result in results if result.get("summary"))
    total = len(results)
    preferred_system, preferred_count = preferred_counts.most_common(1)[0] if preferred_counts else (None, 0)

    by_system_stability = defaultdict(list)
    entropy_values = []
    streak_values = []
    repeated_ratios = []
    for result in results:
        summary = result["summary"]
        entropy_values.append(summary.get("exploration_entropy", 0.0))
        streak_values.append(summary.get("longest_continuous_attraction_streak", 0))
        repeated_ratios.append(summary.get("repeated_to_first_visit_ratio", 0.0))
        for system_id, value in summary.get("attraction_stability", {}).items():
            by_system_stability[system_id].append(value)

    stability_mean = {
        system_id: round(sum(values) / len(values), 4) if values else 0.0
        for system_id, values in by_system_stability.items()
    }

    return {
        "preference_persistence": {
            "preferred_system_id": preferred_system,
            "preferred_seed_count": preferred_count,
            "seed_count": total,
            "consensus_rate": round(preferred_count / total, 4),
            "chance_rate": round(1 / len(SYSTEM_IDS), 4),
        },
        "mean_exploration_entropy": round(sum(entropy_values) / len(entropy_values), 4),
        "mean_longest_continuous_attraction_streak": round(sum(streak_values) / len(streak_values), 4),
        "mean_repeated_to_first_visit_ratio": round(sum(repeated_ratios) / len(repeated_ratios), 4),
        "mean_attraction_stability": stability_mean,
    }


def compare_controllers(seeds: Iterable[int], steps: int, log_dir: Path, controller_names: Iterable[str] = CONTROLLER_NAMES) -> dict:
    comparison = {}
    for controller_name in controller_names:
        runs = run_multi_seed(controller_name=controller_name, seeds=seeds, steps=steps, log_dir=log_dir / controller_name)
        comparison[controller_name] = {
            "runs": runs,
            "aggregate": aggregate_runs(runs),
        }
    return comparison