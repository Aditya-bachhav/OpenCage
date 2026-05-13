from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from cage_env.env import OscillationChamberEnv
from cage_env.dreamer import DreamerPolicy, DreamerUnavailableError
from cage_env.policy import make_policy as make_basic_policy
from cage_env.session_log import SessionLogger


POLICY_NAMES = ("emergent", "random", "heuristic")
SYSTEM_IDS = ("pendulum", "spring", "plate", "wheel", "wave")


def _append_master_log(master_log_path: Path | None, title: str, payload: Any) -> None:
	if master_log_path is None:
		return
	master_log_path.parent.mkdir(parents=True, exist_ok=True)
	with master_log_path.open("a", encoding="utf-8") as handle:
		handle.write("\n" + "=" * 80 + "\n")
		handle.write(title + "\n")
		handle.write("=" * 80 + "\n")
		if isinstance(payload, str):
			handle.write(payload.rstrip() + "\n")
		else:
			handle.write(json.dumps(payload, indent=2, default=str) + "\n")


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


def _summarize_preference_evolution(preference_history: list[dict]) -> dict:
	if not preference_history:
		return {}

	entropies = [p.get("entropy", 0.0) for p in preference_history]
	top_weights = [p.get("top_preference", {}).get("weight", 0.0) for p in preference_history]
	top_systems = [p.get("top_preference", {}).get("system_id") for p in preference_history]

	preference_changes = 0
	for index in range(1, len(top_systems)):
		if top_systems[index] != top_systems[index - 1] and top_systems[index] is not None:
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

	return {
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


def run_session(
    seed: int,
    steps: int,
    log_path: Path,
    policy_name: str = "emergent",
    checkpoint_dir: Path | None = None,
    rollout_dir: Path | None = None,
	master_log_path: Path | None = None,
) -> dict:
	env = OscillationChamberEnv()
	log_path.parent.mkdir(parents=True, exist_ok=True)
	log_path.unlink(missing_ok=True)
	if policy_name == "dreamer":
		policy = DreamerPolicy(
			env.observation_space,
			env.action_space,
			seed=seed,
			checkpoint_dir=checkpoint_dir,
			rollout_dir=rollout_dir,
		)
	else:
		policy = make_basic_policy(policy_name, seed)
	logger = SessionLogger(log_path)

	obs, info = env.reset(seed=seed)
	completed_steps = 0
	preference_evolution: list[dict] = []
	policy_diagnostics = getattr(policy, "diagnostics", lambda: {})()
	if isinstance(policy, DreamerPolicy) and not policy.available:
		logger.write_summary(
			{
				"seed": seed,
				"policy": policy_name,
				"controller": policy_name,
				"steps": 0,
				"summary": {},
				"status": "unavailable",
				"diagnostics": policy_diagnostics,
			}
		)
		_append_master_log(
			master_log_path,
			f"SESSION policy={policy_name} seed={seed} status=unavailable",
			logger.read_all(),
		)
		return {
			"seed": seed,
			"steps": 0,
			"policy": policy_name,
			"controller": policy_name,
			"status": "unavailable",
			"diagnostics": policy_diagnostics,
			"log_path": str(log_path),
			"summary": {},
		}

	for step in range(steps):
		previous_obs = obs
		action = int(policy.act(obs))
		obs, reward, terminated, truncated, info = env.step(action)
		policy.update(
			{
				"observation": previous_obs,
				"next_observation": obs,
				"action": action,
				"reward": reward,
				"step": info.get("step"),
				"terminated": terminated,
				"truncated": truncated,
			}
		)
		logger.write_step(
			{
				"type": "step",
				"seed": seed,
				"policy": policy_name,
				"controller": policy_name,
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
		completed_steps = step + 1
		if terminated or truncated:
			break

	summary = summarize_session(logger.read_all())
	summary["preference_evolution"] = _summarize_preference_evolution(preference_evolution)
	logger.write_summary(
		{
			"seed": seed,
			"policy": policy_name,
			"controller": policy_name,
			"steps": completed_steps,
			"summary": summary,
			"diagnostics": policy_diagnostics,
		}
	)
	result = {
		"seed": seed,
		"steps": completed_steps,
		"policy": policy_name,
		"controller": policy_name,
		"log_path": str(log_path),
		"summary": summary,
		"diagnostics": policy_diagnostics,
	}
	if isinstance(policy, DreamerPolicy):
		result["status"] = "ok"
	_append_master_log(
		master_log_path,
		f"SESSION policy={policy_name} seed={seed} status={result.get('status', 'ok')}",
		logger.read_all(),
	)
	return result


def run_multi_seed(policy_name: str, seeds: Iterable[int], steps: int, log_dir: Path, master_log_path: Path | None = None) -> list[dict]:
	results = []
	log_dir.mkdir(parents=True, exist_ok=True)
	for seed in seeds:
		log_path = log_dir / f"{policy_name}_seed_{seed}.jsonl"
		results.append(run_session(seed=seed, steps=steps, log_path=log_path, policy_name=policy_name, master_log_path=master_log_path))
	_append_master_log(
		master_log_path,
		f"POLICY policy={policy_name}",
		{
			"seeds": list(seeds),
			"steps": steps,
			"aggregate": aggregate_runs(results),
		},
	)
	return results


def aggregate_runs(results: list[dict]) -> dict:
	usable_results = [result for result in results if result.get("summary")]
	if not usable_results:
		return {}

	preferred_counts = Counter(result["summary"]["attraction_preference"]["system_id"] for result in usable_results)
	total = len(usable_results)
	preferred_system, preferred_count = preferred_counts.most_common(1)[0] if preferred_counts else (None, 0)

	by_system_stability = defaultdict(list)
	entropy_values = []
	streak_values = []
	repeated_ratios = []
	for result in usable_results:
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


def compare_policies(seeds: Iterable[int], steps: int, log_dir: Path, policy_names: Iterable[str] = POLICY_NAMES, master_log_path: Path | None = None) -> dict:
	comparison = {}
	for policy_name in policy_names:
		runs = run_multi_seed(policy_name=policy_name, seeds=seeds, steps=steps, log_dir=log_dir / policy_name, master_log_path=master_log_path)
		comparison[policy_name] = {
			"runs": runs,
			"aggregate": aggregate_runs(runs),
			"status": "unavailable" if any(run.get("status") == "unavailable" for run in runs) else "ok",
		}
	_append_master_log(master_log_path, "COMPARISON", comparison)
	return comparison


compare_controllers = compare_policies
