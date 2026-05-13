from __future__ import annotations

from pathlib import Path

from cage_env.dreamer import DreamerPolicy, DreamerUnavailableError
from cage_env.env import OscillationChamberEnv
from cage_env.runner import compare_policies, run_session


def test_dreamer_adapter_diagnostics_and_action_boundary():
	env = OscillationChamberEnv()
	obs, _ = env.reset(seed=11)
	policy = DreamerPolicy(env.observation_space, env.action_space, seed=11)
	diagnostics = policy.diagnostics()

	assert diagnostics["observation_shape"] == (3,)
	assert diagnostics["action_count"] == env.action_space.n
	assert diagnostics["action_mapping"]["inspect"] == 5

	if policy.available:
		action = policy.act(obs)
		assert isinstance(action, int)
		assert 0 <= action < env.action_space.n
	else:
		try:
			policy.act(obs)
		except DreamerUnavailableError as exc:
			assert "DreamerV3" in str(exc)
		else:
			raise AssertionError("DreamerPolicy should raise when the backend is unavailable")


def test_dreamer_compare_surface(tmp_path: Path):
	comparison = compare_policies(seeds=[11, 12], steps=20, log_dir=tmp_path / "compare", policy_names=("dreamer", "emergent", "random", "heuristic"))
	assert set(comparison) == {"dreamer", "emergent", "random", "heuristic"}
	assert "status" in comparison["dreamer"]
	assert "aggregate" in comparison["emergent"]


def test_dreamer_rollout_and_checkpoint_hooks(tmp_path: Path):
	log_path = tmp_path / "dreamer.jsonl"
	result = run_session(
		seed=13,
		steps=25,
		log_path=log_path,
		policy_name="dreamer",
		checkpoint_dir=tmp_path / "checkpoints",
		rollout_dir=tmp_path / "rollouts",
	)
	assert result["policy"] == "dreamer"
	assert result.get("status") in {"ok", "unavailable"}
	assert log_path.exists()