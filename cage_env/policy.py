from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Mapping, Protocol, runtime_checkable

from cage_env.env import ACTION_NAMES


SYSTEM_ANCHORS: dict[str, tuple[float, float]] = {
	"pendulum": (2.5, 7.0),
	"spring": (5.0, 5.5),
	"plate": (7.5, 3.0),
	"wheel": (4.0, 2.0),
	"wave": (1.5, 5.0),
}
SYSTEM_IDS = tuple(SYSTEM_ANCHORS)
ACTION_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}


@runtime_checkable
class Policy(Protocol):
	name: str

	def reset(self, seed: int | None = None) -> None:
		...

	def act(self, observation: Mapping[str, Any]) -> int:
		...

	def update(self, feedback: Mapping[str, Any] | None = None) -> None:
		...


class BasePolicy:
	name = "policy"

	def __init__(self, seed: int | None = None):
		self.seed = seed
		self.rng = random.Random(seed)

	def reset(self, seed: int | None = None) -> None:
		if seed is not None:
			self.seed = seed
		self.rng.seed(self.seed)

	def act(self, observation: Mapping[str, Any]) -> int:
		raise NotImplementedError

	def update(self, feedback: Mapping[str, Any] | None = None) -> None:
		return None


class RandomPolicy(BasePolicy):
	name = "random"

	def act(self, observation: Mapping[str, Any]) -> int:
		return self.rng.randrange(len(ACTION_NAMES))


class HeuristicPolicy(BasePolicy):
	name = "heuristic"

	def act(self, observation: Mapping[str, Any]) -> int:
		position_x = float(observation.get("position_x", 0.0))
		position_y = float(observation.get("position_y", 0.0))

		target_id = self._nearest_anchor(position_x, position_y)
		if target_id is None:
			return ACTION_INDEX["noop"]

		target_x, target_y = SYSTEM_ANCHORS[target_id]
		dx = target_x - position_x
		dy = target_y - position_y
		distance = math.hypot(dx, dy)

		if distance <= 0.55:
			return ACTION_INDEX["inspect"]
		if abs(dx) >= abs(dy):
			return ACTION_INDEX["move_left"] if dx < 0 else ACTION_INDEX["move_right"]
		return ACTION_INDEX["move_down"] if dy < 0 else ACTION_INDEX["move_up"]

	def _nearest_anchor(self, position_x: float, position_y: float) -> str | None:
		nearest_id = None
		nearest_distance = float("inf")
		for system_id, (target_x, target_y) in SYSTEM_ANCHORS.items():
			distance = math.hypot(target_x - position_x, target_y - position_y)
			if distance < nearest_distance:
				nearest_distance = distance
				nearest_id = system_id
		return nearest_id


@dataclass
class EmergentPolicy(BasePolicy):
	name: str = "emergent"
	preference_bias: dict[str, float] = field(default_factory=lambda: {system_id: 0.0 for system_id in SYSTEM_IDS})
	last_target: str | None = None
	exploration_temperature: float = 0.45
	learn_rate: float = 0.08
	decay_rate: float = 0.985

	def __post_init__(self):
		super().__init__(seed=None)

	def reset(self, seed: int | None = None) -> None:
		super().reset(seed=seed)
		self.last_target = None

	def act(self, observation: Mapping[str, Any]) -> int:
		position_x = float(observation.get("position_x", 0.0))
		position_y = float(observation.get("position_y", 0.0))
		target_id = self._choose_target(position_x, position_y)
		self.last_target = target_id
		if target_id is None:
			return ACTION_INDEX["noop"]

		target_x, target_y = SYSTEM_ANCHORS[target_id]
		dx = target_x - position_x
		dy = target_y - position_y
		distance = math.hypot(dx, dy)

		if distance <= 0.55:
			return ACTION_INDEX["inspect"]
		if distance <= 1.2 and self.rng.random() < 0.2 + self.exploration_temperature * 0.2:
			return ACTION_INDEX["noop"]
		if abs(dx) >= abs(dy):
			return ACTION_INDEX["move_left"] if dx < 0 else ACTION_INDEX["move_right"]
		return ACTION_INDEX["move_down"] if dy < 0 else ACTION_INDEX["move_up"]

	def update(self, feedback: Mapping[str, Any] | None = None) -> None:
		if self.last_target is None:
			return

		reward = float((feedback or {}).get("reward", 0.0))
		for system_id in self.preference_bias:
			self.preference_bias[system_id] *= self.decay_rate
		self.preference_bias[self.last_target] += self.learn_rate * reward
		self.exploration_temperature = min(0.7, max(0.12, self.exploration_temperature * 0.9995 + (0.01 if reward <= 0 else 0.0)))

	def _choose_target(self, position_x: float, position_y: float) -> str | None:
		scores: dict[str, float] = {}
		for system_id, (target_x, target_y) in SYSTEM_ANCHORS.items():
			distance = math.hypot(target_x - position_x, target_y - position_y)
			scores[system_id] = self.preference_bias[system_id] - distance * 0.35

		if not scores:
			return None

		if self.rng.random() < self.exploration_temperature:
			choices = list(scores)
			max_score = max(scores.values())
			weights = [max(0.05, math.exp(scores[system_id] - max_score)) for system_id in choices]
			return self.rng.choices(choices, weights=weights, k=1)[0]

		return max(scores, key=scores.get)


def make_policy(name: str, seed: int) -> Policy:
	if name == "emergent":
		policy = EmergentPolicy()
		policy.reset(seed)
		return policy
	if name == "random":
		policy = RandomPolicy(seed=seed)
		policy.reset(seed)
		return policy
	if name == "heuristic":
		policy = HeuristicPolicy(seed=seed)
		policy.reset(seed)
		return policy
	raise ValueError(f"Unknown policy: {name}")
