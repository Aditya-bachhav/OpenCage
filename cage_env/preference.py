"""
preference.py — Persistent attraction preference system.

Allows the agent to develop stable, long-term attraction patterns through
gradual reinforcement and controlled forgetting, WITHOUT hardcoded fixation.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class PreferenceMemory:
    """
    Persistent preference weights for each oscillating system.
    
    Weights evolve through:
    - revisit reinforcement (repeated visits strengthen attraction)
    - dwell-time reinforcement (time spent near a system increases weight)
    - interaction reinforcement (successful energy transfer increases weight)
    - phase alignment reinforcement (synchronization increases weight)
    - decay (forgetting: unused weights decay over time)
    - boredom (exploration pressure increases if entropy drops too far)
    """
    
    system_ids: Sequence[str]
    
    # Per-system preference weights (0.0 → 1.0, higher = more attractive)
    attraction_weight: dict[str, float] = field(default_factory=dict)
    
    # Tracking
    last_reinforced: dict[str, float] = field(default_factory=dict)  # time (steps) when last reinforced
    revisit_count: dict[str, int] = field(default_factory=dict)      # total revisits
    interaction_count: dict[str, int] = field(default_factory=dict)  # total successful interactions
    
    # Recent event history (for trend analysis)
    reinforcement_history: deque = field(default_factory=lambda: deque(maxlen=128))  # (step, system_id, event_type, delta)
    preference_history: deque = field(default_factory=lambda: deque(maxlen=512))      # (step, entropy, top_weight)
    
    # Exploration control
    exploration_temperature: float = 0.5  # 0.0 = pure exploitation, 1.0 = pure exploration
    boredom_threshold: float = 0.3       # if entropy drops below this, increase exploration_temperature
    max_weight: float = 0.9              # hard cap to prevent lock-on fixation
    
    # Decay and reinforcement parameters
    decay_per_step: float = 0.9995       # weight *= decay_per_step each step (slow decay)
    decay_per_step_unused: float = 0.995 # faster decay if oscillator not visited recently
    revisit_reinforce_alpha: float = 0.05
    dwell_reinforce_alpha: float = 0.08
    interaction_reinforce_alpha: float = 0.12
    phase_reinforce_alpha: float = 0.03
    
    def __post_init__(self):
        """Initialize weights for all systems."""
        for system_id in self.system_ids:
            self.attraction_weight[system_id] = 0.1  # start slightly attracted to all
            self.last_reinforced[system_id] = -1000.0
            self.revisit_count[system_id] = 0
            self.interaction_count[system_id] = 0
    
    def reinforce_revisit(self, system_id: str, step_count: int) -> None:
        """Reinforce preference when revisiting an oscillator."""
        if system_id not in self.attraction_weight:
            return
        
        self.revisit_count[system_id] += 1
        delta = self.revisit_reinforce_alpha
        self.attraction_weight[system_id] = min(
            self.max_weight,
            self.attraction_weight[system_id] + delta
        )
        self.last_reinforced[system_id] = step_count
        self.reinforcement_history.append((step_count, system_id, "revisit", delta))
    
    def reinforce_dwell(self, system_id: str, dwell_duration: float, step_count: int) -> None:
        """Reinforce preference based on time spent near oscillator."""
        if system_id not in self.attraction_weight:
            return
        
        # Dwell reinforcement scales with duration (capped at 1 second of dwell)
        clamped_dwell = min(dwell_duration, 1.0)
        delta = self.dwell_reinforce_alpha * (clamped_dwell / 1.0)
        self.attraction_weight[system_id] = min(
            self.max_weight,
            self.attraction_weight[system_id] + delta
        )
        self.last_reinforced[system_id] = step_count
        self.reinforcement_history.append((step_count, system_id, "dwell", delta))
    
    def reinforce_interaction(self, system_id: str, energy_transfer: float, step_count: int) -> None:
        """Reinforce preference when successful energy transfer occurs."""
        if system_id not in self.attraction_weight:
            return
        
        self.interaction_count[system_id] += 1
        delta = self.interaction_reinforce_alpha * min(energy_transfer / 0.06, 1.0)  # normalize
        self.attraction_weight[system_id] = min(
            self.max_weight,
            self.attraction_weight[system_id] + delta
        )
        self.last_reinforced[system_id] = step_count
        self.reinforcement_history.append((step_count, system_id, "interaction", delta))
    
    def reinforce_phase_alignment(self, system_id: str, alignment_score: float, step_count: int) -> None:
        """Reinforce preference when synchronized with oscillator phase."""
        if system_id not in self.attraction_weight:
            return
        
        # Only reinforce if alignment is strong (> 0.7)
        if alignment_score < 0.7:
            return
        
        delta = self.phase_reinforce_alpha * alignment_score
        self.attraction_weight[system_id] = min(
            self.max_weight,
            self.attraction_weight[system_id] + delta
        )
        self.last_reinforced[system_id] = step_count
        self.reinforcement_history.append((step_count, system_id, "phase_sync", delta))
    
    def decay(self, step_count: int) -> None:
        """Apply decay to all weights each step."""
        for system_id in self.system_ids:
            # Check if recently visited
            time_since_reinforced = step_count - self.last_reinforced[system_id]
            if time_since_reinforced > 100:  # not visited in last 100 steps
                decay = self.decay_per_step_unused
            else:
                decay = self.decay_per_step
            
            self.attraction_weight[system_id] *= decay
            # Floor to zero
            if self.attraction_weight[system_id] < 0.01:
                self.attraction_weight[system_id] = 0.01
    
    def update_boredom(self, step_count: int) -> None:
        """
        Increase exploration temperature if preferences become too concentrated.
        This prevents lock-in and maintains adaptive behavior.
        """
        entropy = self.compute_entropy()
        self.preference_history.append((step_count, entropy, max(self.attraction_weight.values())))
        
        # If entropy drops too low, increase exploration
        if entropy < self.boredom_threshold:
            self.exploration_temperature = min(1.0, self.exploration_temperature + 0.02)
        else:
            # Gradually return to baseline if entropy recovers
            self.exploration_temperature = max(0.3, self.exploration_temperature - 0.01)
    
    def compute_entropy(self) -> float:
        """Compute Shannon entropy of preference distribution (0 = focused, ~log5 = uniform)."""
        total = sum(self.attraction_weight.values())
        if total == 0:
            return 0.0
        
        probs = [w / total for w in self.attraction_weight.values()]
        entropy = 0.0
        for p in probs:
            if p > 1e-6:
                entropy -= p * math.log2(p)
        return entropy
    
    def get_weights(self) -> dict[str, float]:
        """Return current preference weights."""
        return dict(self.attraction_weight)
    
    def get_top_preference(self) -> tuple[str | None, float]:
        """Return most-preferred oscillator and its weight."""
        if not self.attraction_weight:
            return None, 0.0
        return max(self.attraction_weight.items(), key=lambda item: item[1])
    
    def summary(self, step_count: int) -> dict:
        """Return preference state summary."""
        top_id, top_weight = self.get_top_preference()
        return {
            "step": step_count,
            "entropy": round(self.compute_entropy(), 4),
            "top_preference": {"system_id": top_id, "weight": round(top_weight, 4)},
            "exploration_temperature": round(self.exploration_temperature, 4),
            "weights": {sid: round(w, 4) for sid, w in self.attraction_weight.items()},
            "revisit_counts": dict(self.revisit_count),
            "interaction_counts": dict(self.interaction_count),
        }
