from __future__ import annotations

import math
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cage_env.measurements import MeasurementEngine, SystemMetrics
from cage_env.objects import (
    OscillatingSystem,
    make_agent,
    make_pendulum,
    make_resonance_plate,
    make_rotating_wheel,
    make_spring,
    make_wave_emitter,
)
from cage_env.physics import DT, agent_interact, distance, tick
from cage_env.preference import PreferenceMemory

MAX_STEPS = 4000
ACTION_NAMES = ["noop", "move_left", "move_right", "move_up", "move_down", "inspect"]


class OscillationChamberEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode: str = "ansi", log_dir: str | Path = "cage_env/logs"):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Dict(
            {
                "position_x": spaces.Box(0.0, 10.0, shape=(), dtype=np.float32),
                "position_y": spaces.Box(0.0, 10.0, shape=(), dtype=np.float32),
                "distance_to_nearest": spaces.Box(0.0, 15.0, shape=(), dtype=np.float32),
            }
        )
        self.log_dir = Path(log_dir)
        self.agent = make_agent()
        self.systems = [make_pendulum(), make_spring(), make_resonance_plate(), make_rotating_wheel(), make_wave_emitter()]
        self.measurements = MeasurementEngine()
        self.system_ids = [system.id for system in self.systems]
        self.preference_memory = PreferenceMemory(system_ids=self.system_ids)
        self.step_count = 0
        self.episode = 0
        self._last_transfer = 0.0
        self._last_dwell_start = {}  # track dwell start time per system

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
        self.step_count = 0
        self.episode += 1
        self.agent.x = float(self.np_random.uniform(4.4, 5.6))
        self.agent.y = float(self.np_random.uniform(4.4, 5.6))
        self.agent.energy = 1.0
        self._last_transfer = 0.0
        self._last_dwell_start = {}
        self.measurements.reset()
        # Don't reset preference_memory — allow persistence across episodes
        return self._build_obs(), self._info()

    def step(self, action: int):
        self.step_count += 1
        self._apply_action(action)

        for system in self.systems:
            tick(system)

        nearest = self._nearest_system()
        transfer = 0.0
        if action == ACTION_NAMES.index("inspect") and nearest is not None:
            self.agent, nearest, transfer = agent_interact(self.agent, nearest)
            if transfer > 0:
                metrics = self.measurements.metrics.setdefault(nearest.id, SystemMetrics(system_id=nearest.id))
                metrics.interactions += 1
                metrics.energy_transferred += transfer
                # Reinforce preference on successful interaction
                self.preference_memory.reinforce_interaction(nearest.id, transfer, self.step_count)

        self._last_transfer = transfer
        self.measurements.observe(self.agent, self.systems, self.step_count)
        
        # Track preference reinforcement based on measurements
        nearest_id = self.measurements.get_signals().nearest_system_id
        if nearest_id is not None:
            # Reinforce on revisit (entering dwell zone)
            if nearest_id not in self._last_dwell_start:
                self.preference_memory.reinforce_revisit(nearest_id, self.step_count)
                self._last_dwell_start[nearest_id] = self.step_count
            
            # Reinforce dwell time
            dwell_start = self._last_dwell_start.get(nearest_id, self.step_count)
            dwell_duration = (self.step_count - dwell_start) * DT
            self.preference_memory.reinforce_dwell(nearest_id, dwell_duration, self.step_count)
            
            # Reinforce on phase alignment
            signals = self.measurements.get_signals()
            phase_align = signals.phase_alignment.get(nearest_id, 0.0)
            self.preference_memory.reinforce_phase_alignment(nearest_id, phase_align, self.step_count)
        else:
            # Clear dwell tracking when no longer in any dwell zone
            self._last_dwell_start.clear()
        
        # Apply preference decay and update boredom
        self.preference_memory.decay(self.step_count)
        self.preference_memory.update_boredom(self.step_count)
        
        reward = self._compute_reward(nearest, transfer)
        obs = self._build_obs()
        info = self._info()
        terminated = False
        truncated = self.step_count >= MAX_STEPS
        return obs, float(reward), terminated, truncated, info

    def render(self):
        print(self._info())

    def _apply_action(self, action: int):
        speed = 0.25
        if action == 1:
            self.agent.x = max(self.agent.radius, self.agent.x - speed)
        elif action == 2:
            self.agent.x = min(10.0 - self.agent.radius, self.agent.x + speed)
        elif action == 3:
            self.agent.y = min(10.0 - self.agent.radius, self.agent.y + speed)
        elif action == 4:
            self.agent.y = max(self.agent.radius, self.agent.y - speed)

    def _nearest_system(self) -> OscillatingSystem | None:
        return min(self.systems, key=lambda system: distance(self.agent, system), default=None)

    def _compute_reward(self, nearest: OscillatingSystem | None, transfer: float) -> float:
        signals = self.measurements.get_signals()
        motion = math.hypot(*signals.velocity)
        proximity = 1.0 / (1.0 + signals.distance_to_nearest)
        attraction = signals.attraction_scores.get(nearest.id, 0.0) if nearest is not None else 0.0
        phase_alignment = signals.phase_alignment.get(nearest.id, 0.0) if nearest is not None else 0.0
        return round(0.08 * proximity + 0.04 * motion + 0.1 * attraction + 0.05 * phase_alignment + transfer, 4)

    def _build_obs(self) -> dict:
        signals = self.measurements.get_signals()
        return {
            "position_x": float(self.agent.x),
            "position_y": float(self.agent.y),
            "distance_to_nearest": float(signals.distance_to_nearest),
        }

    def _info(self) -> dict:
        signals = self.measurements.get_signals()
        return {
            "step": self.step_count,
            "episode": self.episode,
            "agent": self.agent.to_dict(),
            "nearest_system_id": signals.nearest_system_id,
            "signals": signals.to_dict(),
            "systems": [system.to_dict() for system in self.systems],
            "metrics": self.measurements.summary(),
            "last_transfer": self._last_transfer,
            "preferences": self.preference_memory.summary(self.step_count),
        }
