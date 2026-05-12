from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OscillatingSystem:
    id: str
    kind: str
    x: float
    y: float
    radius: float = 0.3
    phase: float = 0.0
    frequency: float = 1.0
    amplitude: float = 1.0
    damping: float = 0.98
    energy: float = 1.0
    anchor_x: float | None = None
    anchor_y: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "x": self.x,
            "y": self.y,
            "radius": self.radius,
            "phase": self.phase,
            "frequency": self.frequency,
            "amplitude": self.amplitude,
            "damping": self.damping,
            "energy": self.energy,
            "anchor_x": self.anchor_x,
            "anchor_y": self.anchor_y,
        }


def make_pendulum() -> OscillatingSystem:
    return OscillatingSystem(id="pendulum", kind="pendulum", x=2.5, y=7.0, frequency=1.2, amplitude=0.8, damping=0.97, anchor_x=2.5, anchor_y=7.0)


def make_spring() -> OscillatingSystem:
    return OscillatingSystem(id="spring", kind="spring", x=5.0, y=5.5, frequency=2.1, amplitude=0.6, damping=0.96, anchor_x=5.0, anchor_y=5.5)


def make_resonance_plate() -> OscillatingSystem:
    return OscillatingSystem(id="plate", kind="plate", x=7.5, y=3.0, frequency=1.8, amplitude=0.5, damping=0.94, anchor_x=7.5, anchor_y=3.0)


def make_rotating_wheel() -> OscillatingSystem:
    return OscillatingSystem(id="wheel", kind="wheel", x=4.0, y=2.0, frequency=0.8, amplitude=1.2, damping=0.98, anchor_x=4.0, anchor_y=2.0)


def make_wave_emitter() -> OscillatingSystem:
    return OscillatingSystem(id="wave", kind="wave", x=1.5, y=5.0, frequency=1.5, amplitude=0.7, damping=0.95, anchor_x=1.5, anchor_y=5.0)


def make_agent() -> OscillatingSystem:
    return OscillatingSystem(id="agent", kind="agent", x=5.0, y=5.0, radius=0.2, frequency=0.0, amplitude=0.0, damping=1.0, energy=1.0, anchor_x=5.0, anchor_y=5.0)
