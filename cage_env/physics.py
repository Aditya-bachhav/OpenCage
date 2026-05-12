from __future__ import annotations

import math

from cage_env.objects import OscillatingSystem

WORLD_W = 10.0
WORLD_H = 10.0
DT = 0.05


def tick(system: OscillatingSystem) -> OscillatingSystem:
    if system.kind == "agent":
        return system

    system.energy = max(0.08, system.energy * system.damping)
    system.phase = (system.phase + 2.0 * math.pi * system.frequency * DT) % (2.0 * math.pi)

    anchor_x = system.anchor_x if system.anchor_x is not None else system.x
    anchor_y = system.anchor_y if system.anchor_y is not None else system.y

    if system.kind == "wheel":
        radius_offset = system.amplitude * system.energy * 0.75
        system.x = anchor_x + radius_offset * math.cos(system.phase)
        system.y = anchor_y + radius_offset * math.sin(system.phase)
    elif system.kind == "wave":
        system.x = anchor_x + math.sin(system.phase) * system.amplitude * 0.35 * system.energy
        system.y = anchor_y + math.cos(system.phase * 0.5) * system.amplitude * 0.12 * system.energy
    else:
        system.x = anchor_x + math.sin(system.phase) * system.amplitude * system.energy

    system.x = min(max(system.radius, system.x), WORLD_W - system.radius)
    system.y = min(max(system.radius, system.y), WORLD_H - system.radius)
    return system


def distance(a: OscillatingSystem, b: OscillatingSystem) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def agent_interact(agent: OscillatingSystem, target: OscillatingSystem) -> tuple[OscillatingSystem, OscillatingSystem, float]:
    dist = distance(agent, target)
    if dist > 1.2:
        return agent, target, 0.0

    transfer = 0.06 * (1.0 - dist / 1.2)
    target.energy = min(1.0, target.energy + transfer)
    return agent, target, transfer


def phase_difference(phase_a: float, phase_b: float) -> float:
    diff = abs(phase_a - phase_b) % (2.0 * math.pi)
    if diff > math.pi:
        diff = 2.0 * math.pi - diff
    return diff / math.pi


def predict_next_state(system: OscillatingSystem) -> dict:
    next_phase = (system.phase + 2.0 * math.pi * system.frequency * DT) % (2.0 * math.pi)
    next_energy = max(0.08, system.energy * system.damping)
    return {"phase": next_phase, "energy": next_energy}
