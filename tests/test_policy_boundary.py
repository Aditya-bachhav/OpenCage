from __future__ import annotations

from pathlib import Path

from cage_env.env import OscillationChamberEnv
from cage_env.policy import EmergentPolicy, HeuristicPolicy, RandomPolicy
from cage_env.runner import run_session
from cage_env.visualize import load_replay


def test_policy_interface_and_env_boundary():
	env = OscillationChamberEnv()
	obs, info = env.reset(seed=7)

	policies = [RandomPolicy(seed=7), HeuristicPolicy(seed=7), EmergentPolicy()]
	for policy in policies:
		policy.reset(seed=7)
		action = policy.act(obs)
		assert isinstance(action, int)
		assert 0 <= action < env.action_space.n
		policy.update({"reward": 0.0})

	assert not hasattr(env, "policy")
	assert not hasattr(env, "controller")
	assert env.action_space.n == 6


def test_long_sessions_across_policies(tmp_path: Path):
	for policy_name in ("emergent", "random", "heuristic"):
		log_path = tmp_path / f"{policy_name}.jsonl"
		result = run_session(seed=31, steps=400, log_path=log_path, policy_name=policy_name)
		assert result["policy"] == policy_name
		assert result["steps"] > 0
		assert log_path.exists()
		assert "repeated_to_first_visit_ratio" in result["summary"]
		assert "mean_exploration_entropy" not in result["summary"]


def test_replay_logs_still_load(tmp_path: Path):
	log_path = tmp_path / "replay.jsonl"
	run_session(seed=19, steps=80, log_path=log_path, policy_name="emergent")
	frames = load_replay(log_path)
	assert len(frames) > 0
	assert all(frame.get("type") == "step" for frame in frames)
	assert all("info" in frame and "signals" in frame["info"] for frame in frames)