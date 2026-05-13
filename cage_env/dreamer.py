from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from importlib import import_module, util
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cage_env.policy import BasePolicy


_PACKAGE_NAME = "dreamerv3"
_CONFIG_FILENAME = "configs.yaml"
_BACKEND_CANDIDATES = (_PACKAGE_NAME,)


class DreamerUnavailableError(RuntimeError):
	pass


@dataclass
class DreamerDiagnostics:
	available: bool
	backend_name: str | None
	observation_shape: tuple[int, ...]
	action_count: int
	action_mapping: dict[str, int]
	config_path: str | None = None
	failure_reason: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"available": self.available,
			"backend_name": self.backend_name,
			"observation_shape": self.observation_shape,
			"action_count": self.action_count,
			"action_mapping": self.action_mapping,
			"config_path": self.config_path,
			"failure_reason": self.failure_reason,
		}


def _repo_config_path() -> Path:
	return Path(__file__).with_name("dreamerv3_configs.yaml")


def _ensure_package_config() -> Path | None:
	os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
	os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
	os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

	spec = util.find_spec(_PACKAGE_NAME)
	if spec is None or not spec.submodule_search_locations:
		return None

	package_dir = Path(next(iter(spec.submodule_search_locations)))
	package_config = package_dir / _CONFIG_FILENAME
	repo_config = _repo_config_path()
	if not package_config.exists() and repo_config.exists():
		shutil.copyfile(repo_config, package_config)
	return package_config if package_config.exists() else None


def _load_dreamerv3():
	package_config = _ensure_package_config()
	if package_config is None:
		raise DreamerUnavailableError("DreamerV3 package not found.")
	package_dir = Path(package_config).parent
	if str(package_dir) not in sys.path:
		sys.path.insert(0, str(package_dir))
	path_module_name = "embodied.core.path"
	if path_module_name not in sys.modules:
		path_spec = util.spec_from_file_location(path_module_name, package_dir / "embodied/core/path.py")
		if path_spec is not None and path_spec.loader is not None:
			path_module = util.module_from_spec(path_spec)
			path_spec.loader.exec_module(path_module)
			sys.modules[path_module_name] = path_module
			sys.modules.setdefault("dreamerv3.embodied.core.path", path_module)
	path_module = sys.modules.get(path_module_name)
	if path_module is not None:
		original_init = path_module.Path.__init__

		def patched_init(self, path):
			path = str(path).replace("\\", "/")
			original_init(self, path)

		path_module.Path.__init__ = patched_init
	return import_module(_PACKAGE_NAME)


def _make_space(dtype, shape=(), low=None, high=None):
	space_module = import_module("dreamerv3.embodied.core.space")
	return space_module.Space(dtype, shape=shape, low=low, high=high)


class DreamerPolicy(BasePolicy):
	name = "dreamer"

	def __init__(
		self,
		observation_space,
		action_space,
		seed: int | None = None,
		checkpoint_dir: str | Path | None = None,
		rollout_dir: str | Path | None = None,
	):
		super().__init__(seed=seed)
		self.observation_space = observation_space
		self.action_space = action_space
		self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
		self.rollout_dir = Path(rollout_dir) if rollout_dir is not None else None
		self._dreamerv3 = None
		self._agent = None
		self._policy_state = None
		self._train_state = None
		self._episode: list[dict[str, Any]] = []
		self._is_first = True
		self._failure_reason: str | None = None
		self._last_metrics: dict[str, Any] = {}
		self._backend_name: str | None = None
		self._diagnostics = DreamerDiagnostics(
			available=False,
			backend_name=None,
			observation_shape=self._observation_shape(),
			action_count=self.action_space.n,
			action_mapping={name: index for index, name in enumerate(self._action_names())},
		)
		self._build_agent()

	@property
	def available(self) -> bool:
		return self._agent is not None

	def reset(self, seed: int | None = None) -> None:
		super().reset(seed=seed)
		self._is_first = True
		self._episode = []
		if self._agent is not None:
			self._policy_state = self._agent.policy_initial(1)
			self._train_state = self._agent.train_initial(1)

	def act(self, observation: Mapping[str, Any]) -> int:
		if self._agent is None or self._policy_state is None:
			raise DreamerUnavailableError(self._failure_reason or "DreamerV3 backend is unavailable.")

		backend_obs = self._encode_observation(observation, is_terminal=False, is_first=self._is_first)
		outs, self._policy_state = self._agent.policy(backend_obs, self._policy_state, mode="eval")
		self._is_first = False
		action = np.asarray(outs["action"])
		if action.shape:
			action = action.reshape(-1)[0]
		return int(action)

	def update(self, feedback: Mapping[str, Any] | None = None) -> None:
		feedback = dict(feedback or {})
		if self._agent is None:
			return

		transition = {
			"observation": feedback.get("observation"),
			"next_observation": feedback.get("next_observation"),
			"action": feedback.get("action"),
			"reward": float(feedback.get("reward", 0.0)),
			"terminated": bool(feedback.get("terminated", False)),
			"truncated": bool(feedback.get("truncated", False)),
		}
		self._episode.append(transition)
		if transition["terminated"] or transition["truncated"]:
			self._train_episode()

	def diagnostics(self) -> dict[str, Any]:
		payload = self._diagnostics.to_dict()
		payload.update({
			"backend_name": self._backend_name,
			"failure_reason": self._failure_reason,
			"episode_length": len(self._episode),
			"last_metrics": self._last_metrics,
		})
		return payload

	def save_checkpoint(self) -> Path | None:
		if self._agent is None or self.checkpoint_dir is None:
			return None
		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
		checkpoint_path = self.checkpoint_dir / "dreamer_policy_state.pkl"
		state = self._agent.save()
		with checkpoint_path.open("wb") as handle:
			import pickle

			pickle.dump(state, handle)
		return checkpoint_path

	def _build_agent(self) -> None:
		try:
			dreamerv3 = _load_dreamerv3()
			embodied = import_module("dreamerv3.embodied")
			config = embodied.Config(dreamerv3.Agent.configs["defaults"])
			config = config.update({"jax": {"platform": "cpu", "precision": "float32"}, "run": {"script": "train"}})
			obs_space = self._dreamer_obs_space()
			act_space = self._dreamer_act_space()
			step = embodied.Counter()
			self._agent = dreamerv3.Agent(obs_space, act_space, step, config)
			self._policy_state = self._agent.policy_initial(1)
			self._train_state = self._agent.train_initial(1)
			self._dreamerv3 = dreamerv3
			self._backend_name = dreamerv3.__name__
			self._diagnostics = DreamerDiagnostics(
				available=True,
				backend_name=self._backend_name,
				observation_shape=self._observation_shape(),
				action_count=self.action_space.n,
				action_mapping={name: index for index, name in enumerate(self._action_names())},
				config_path=str(_ensure_package_config()) if _ensure_package_config() else None,
			)
		except Exception as exc:  # pragma: no cover - backend initialization failure path
			self._failure_reason = f"DreamerV3 backend unavailable: {exc}"
			self._diagnostics = DreamerDiagnostics(
				available=False,
				backend_name=None,
				observation_shape=self._observation_shape(),
				action_count=self.action_space.n,
				action_mapping={name: index for index, name in enumerate(self._action_names())},
				config_path=str(_ensure_package_config()) if _ensure_package_config() else None,
				failure_reason=self._failure_reason,
			)

	def _train_episode(self) -> None:
		if self._agent is None or self._train_state is None or not self._episode:
			return

		batch = self._build_training_batch(self._episode)
		outs, self._train_state, metrics = self._agent.train(batch, self._train_state)
		self._last_metrics = {key: self._to_python(value) for key, value in metrics.items()}
		self._episode.clear()
		if self.rollout_dir is not None:
			self.rollout_dir.mkdir(parents=True, exist_ok=True)
			rollout_path = self.rollout_dir / "dreamer_rollout.jsonl"
			with rollout_path.open("a", encoding="utf-8") as handle:
				handle.write(json.dumps({"metrics": self._last_metrics, "outs": self._to_python(outs)}) + "\n")
		if self.checkpoint_dir is not None:
			self.save_checkpoint()

	def _build_training_batch(self, episode: list[dict[str, Any]]) -> dict[str, np.ndarray]:
		length = len(episode)
		observation_tensor = np.zeros((1, length, 3), dtype=np.float32)
		reward = np.zeros((1, length), dtype=np.float32)
		action = np.zeros((1, length), dtype=np.int32)
		is_first = np.zeros((1, length, 1), dtype=np.bool_)
		is_terminal = np.zeros((1, length, 1), dtype=np.bool_)
		for index, transition in enumerate(episode):
			observation_data = transition["observation"] or {}
			observation_tensor[0, index, 0] = float(observation_data.get("position_x", 0.0))
			observation_tensor[0, index, 1] = float(observation_data.get("position_y", 0.0))
			observation_tensor[0, index, 2] = float(observation_data.get("distance_to_nearest", 0.0))
			reward[0, index] = float(transition.get("reward", 0.0))
			action[0, index] = int(transition.get("action", 0))
			is_first[0, index, 0] = index == 0
			is_terminal[0, index, 0] = bool(transition.get("terminated", False) or transition.get("truncated", False))
		return {
			"observation": observation_tensor,
			"reward": reward,
			"action": action,
			"is_first": is_first,
			"is_terminal": is_terminal,
		}

	def _dreamer_obs_space(self):
		return {
			"observation": _make_space(np.float32, shape=(3,)),
		}

	def _dreamer_act_space(self):
		return {"action": _make_space(np.int32, shape=(), low=0, high=self.action_space.n - 1)}

	def _encode_observation(self, observation: Mapping[str, Any], is_terminal: bool, is_first: bool) -> dict[str, np.ndarray]:
		return {
			"observation": np.asarray(
				[[
					float(observation.get("position_x", 0.0)),
					float(observation.get("position_y", 0.0)),
					float(observation.get("distance_to_nearest", 0.0)),
				]],
				dtype=np.float32,
			),
			"is_first": np.asarray([[is_first]], dtype=np.bool_),
			"is_terminal": np.asarray([[is_terminal]], dtype=np.bool_),
		}

	def _observation_shape(self) -> tuple[int, ...]:
		return (3,)

	def _action_names(self) -> list[str]:
		return ["noop", "move_left", "move_right", "move_up", "move_down", "inspect"]

	def _to_python(self, value: Any) -> Any:
		if isinstance(value, dict):
			return {key: self._to_python(item) for key, item in value.items()}
		if isinstance(value, (list, tuple)):
			return [self._to_python(item) for item in value]
		if hasattr(value, "tolist"):
			try:
				return value.tolist()
			except Exception:
				return str(value)
		return value
