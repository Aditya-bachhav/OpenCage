"""
measurements.py — Physically grounded measurement system for the Oscillation Chamber.

Instead of fake labels like "discovering", we track:
- Time spent near each system
- Revisit intervals
- Interaction persistence
- Synchronization scores
- Energy transfer effects
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

from cage_env.physics import DT, phase_difference


@dataclass
class SystemMetrics:
    """Metrics for a single oscillating system."""
    
    system_id: str
    visits: int = 0
    total_dwell_time: float = 0.0  # seconds
    last_visit_time: float = 0.0
    last_visit_phase: float | None = None
    revisit_intervals: deque = field(default_factory=lambda: deque(maxlen=16))
    phase_alignment_samples: deque = field(default_factory=lambda: deque(maxlen=32))
    current_dwell_start: float | None = None
    is_currently_visited: bool = False
    
    # Interaction tracking
    interactions: int = 0
    energy_transferred: float = 0.0
    synchronization_attempts: int = 0
    
    # Synchronization
    synchronization_samples: deque = field(default_factory=lambda: deque(maxlen=32))


@dataclass
class RawSignals:
    """Raw, uninterpreted signals about agent behavior."""
    
    time_since_last_action: float = 0.0
    current_position: tuple[float, float] = (0.0, 0.0)
    distance_to_nearest: float = 10.0
    nearest_system_id: str | None = None
    velocity: tuple[float, float] = (0.0, 0.0)
    
    # Attraction scores (purely data-driven)
    attraction_scores: dict[str, float] = field(default_factory=dict)
    revisit_frequencies: dict[str, float] = field(default_factory=dict)
    phase_alignment: dict[str, float] = field(default_factory=dict)
    
    # System energies (direct physical state)
    system_energies: dict[str, float] = field(default_factory=dict)
    system_phases: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "position": self.current_position,
            "distance_to_nearest": round(self.distance_to_nearest, 3),
            "nearest_system_id": self.nearest_system_id,
            "velocity": (round(self.velocity[0], 3), round(self.velocity[1], 3)),
            "attraction_scores": {k: round(v, 3) for k, v in self.attraction_scores.items()},
            "revisit_frequencies": {k: round(v, 3) for k, v in self.revisit_frequencies.items()},
            "phase_alignment": {k: round(v, 3) for k, v in self.phase_alignment.items()},
            "system_energies": {k: round(v, 3) for k, v in self.system_energies.items()},
            "system_phases": {k: round(v, 3) for k, v in self.system_phases.items()},
        }


class MeasurementEngine:
    """Track physically grounded metrics, no fake labels."""
    
    def __init__(self):
        self.metrics: dict[str, SystemMetrics] = {}
        self.signals = RawSignals()
        self.step_count = 0
        self.position_history: deque = deque(maxlen=256)
        
    def reset(self):
        """Reset measurements for a new session."""
        self.metrics.clear()
        self.signals = RawSignals()
        self.step_count = 0
        self.position_history.clear()
    
    def observe(self, agent, systems: list, step_count: int):
        """Update measurements based on current state."""
        self.step_count = step_count
        
        # Record agent position
        agent_pos = (agent.x, agent.y)
        self.position_history.append(agent_pos)
        self.signals.current_position = agent_pos
        
        # Calculate velocity
        if len(self.position_history) > 1:
            prev = self.position_history[-2]
            curr = self.position_history[-1]
            vx = (curr[0] - prev[0]) / 0.05
            vy = (curr[1] - prev[1]) / 0.05
            self.signals.velocity = (vx, vy)
        
        # Nearest system
        nearest = None
        min_dist = 10.0
        for sys in systems:
            if sys.kind == "agent":
                continue
            dist = math.hypot(sys.x - agent.x, sys.y - agent.y)
            if dist < min_dist:
                min_dist = dist
                nearest = sys
        
        self.signals.distance_to_nearest = min_dist
        self.signals.nearest_system_id = nearest.id if nearest else None
        
        # Per-system metrics
        dwell_threshold = 0.5  # meters
        for sys in systems:
            if sys.kind == "agent":
                continue
            
            if sys.id not in self.metrics:
                self.metrics[sys.id] = SystemMetrics(system_id=sys.id)
            
            metrics = self.metrics[sys.id]
            dist = math.hypot(sys.x - agent.x, sys.y - agent.y)
            
            # Dwell time tracking
            if dist < dwell_threshold:
                if not metrics.is_currently_visited:
                    # Just entered
                    metrics.visits += 1
                    metrics.synchronization_attempts += 1
                    metrics.current_dwell_start = step_count * 0.05
                    
                    # Calculate interval since last visit
                    current_time = step_count * 0.05
                    if metrics.last_visit_time > 0:
                        interval = current_time - metrics.last_visit_time
                        metrics.revisit_intervals.append(interval)
                    current_phase = sys.phase % (2.0 * math.pi)
                    if metrics.last_visit_phase is not None:
                        alignment = 1.0 - phase_difference(current_phase, metrics.last_visit_phase)
                        metrics.phase_alignment_samples.append(alignment)
                    metrics.last_visit_phase = current_phase
                    metrics.last_visit_time = current_time
                    metrics.is_currently_visited = True
                
                # Accumulate dwell time
                metrics.total_dwell_time += 0.05
            else:
                metrics.is_currently_visited = False
            
            # Record system state
            self.signals.system_energies[sys.id] = sys.energy
            self.signals.system_phases[sys.id] = sys.phase / (2.0 * math.pi)
        
        # Calculate attraction scores (based on revisit frequency and dwell time)
        for sys_id, metrics in self.metrics.items():
            if metrics.visits == 0:
                score = 0.0
            else:
                # Attraction = visits * dwell_time
                score = min(1.0, (metrics.visits * metrics.total_dwell_time) / 100.0)
            self.signals.attraction_scores[sys_id] = score
            
            # Revisit frequency (visits per 100 steps)
            steps_per_visit = (self.step_count + 1) / max(metrics.visits, 1)
            freq = 100.0 / max(steps_per_visit, 1.0)
            self.signals.revisit_frequencies[sys_id] = freq

            alignment = sum(metrics.phase_alignment_samples) / len(metrics.phase_alignment_samples) if metrics.phase_alignment_samples else 0.0
            self.signals.phase_alignment[sys_id] = alignment
    
    def get_signals(self) -> RawSignals:
        """Return current raw signals."""
        return self.signals
    
    def get_metrics_for_system(self, system_id: str) -> SystemMetrics | None:
        """Get detailed metrics for a system."""
        return self.metrics.get(system_id)
    
    def all_metrics(self) -> dict[str, SystemMetrics]:
        """Get all system metrics."""
        return self.metrics

    def summary(self) -> dict:
        summary = {}
        for system_id, metrics in self.metrics.items():
            summary[system_id] = {
                "visits": metrics.visits,
                "dwell_time": round(metrics.total_dwell_time, 3),
                "revisit_count": len(metrics.revisit_intervals),
                "phase_alignment": round(sum(metrics.phase_alignment_samples) / len(metrics.phase_alignment_samples), 3) if metrics.phase_alignment_samples else 0.0,
                "synchronization_attempts": metrics.synchronization_attempts,
                "attraction_score": round((metrics.visits * metrics.total_dwell_time) / 100.0, 3) if metrics.visits else 0.0,
            }
        return summary
