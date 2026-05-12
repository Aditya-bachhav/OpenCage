from __future__ import annotations

import math
import random


class SignalResponder:
    def __init__(self, seed: int = 11, preference_memory=None):
        self.rng = random.Random(seed)
        self.preference_memory = preference_memory

    def choose_action(self, info: dict) -> int:
        agent = info.get("agent") or {}
        systems = info.get("systems") or []
        nearest_id = info.get("nearest_system_id")
        signals = info.get("signals") or {}
        attraction_scores = signals.get("attraction_scores", {})

        if not systems:
            return 0

        nearest = next((system for system in systems if system.get("id") == nearest_id), None)
        if nearest is None:
            nearest = min(systems, key=lambda system: math.hypot(system["x"] - agent.get("x", 0.0), system["y"] - agent.get("y", 0.0)))

        dx = nearest["x"] - agent.get("x", 0.0)
        dy = nearest["y"] - agent.get("y", 0.0)
        dist = math.hypot(dx, dy)
        
        # Use learned preferences if available; otherwise use raw attraction scores
        if self.preference_memory is not None:
            learned_preference = self.preference_memory.attraction_weight.get(nearest["id"], 0.0)
            exploration_temp = self.preference_memory.exploration_temperature
        else:
            learned_preference = 0.0
            exploration_temp = 0.5
        
        attraction = (attraction_scores.get(nearest["id"], 0.0) * 0.4 + 
                      learned_preference * 0.6)

        weights = {0: 0.08, 1: 0.16, 2: 0.16, 3: 0.16, 4: 0.16, 5: 0.16}

        if dist > 1.6:
            if dx < 0:
                weights[1] += 0.30 + attraction * 0.15
            else:
                weights[2] += 0.30 + attraction * 0.15
            if dy < 0:
                weights[4] += 0.12
            else:
                weights[3] += 0.12
        elif dist > 0.7:
            weights[5] += 0.20 + attraction * 0.20
            weights[0] += 0.08
        else:
            weights[5] += 0.40 + attraction * 0.15
            weights[0] += 0.12

        # Apply exploration temperature: higher = more random
        if exploration_temp > 0.3:
            random_action = self.rng.randrange(6)
            exploit_action = self.rng.choices(range(6), weights=[weights.get(i, 0.1) for i in range(6)], k=1)[0]
            # Blend based on exploration_temperature
            return random_action if self.rng.random() < exploration_temp else exploit_action

        total = sum(weights.values())
        normalized = [weights[i] / total for i in range(6)]
        return self.rng.choices(range(6), weights=normalized, k=1)[0]