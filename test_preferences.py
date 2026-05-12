#!/usr/bin/env python3
"""Quick validation test for preference system."""

from cage_env.env import OscillationChamberEnv
from cage_env.controller import SignalResponder

def test_preference_system():
    print("Testing preference system integration...")
    
    # Create environment
    env = OscillationChamberEnv()
    controller = SignalResponder(seed=42)
    
    # Link preferences
    controller.preference_memory = env.preference_memory
    
    # Initialize
    obs, info = env.reset(seed=42)
    print("✓ Initialization OK")
    print(f"  Initial preferences: {info['preferences']['weights']}")
    
    # Run a few steps
    for i in range(50):
        action = controller.choose_action(info)
        obs, reward, terminated, truncated, info = env.step(action)
    
    pref_state = info['preferences']
    print(f"✓ After 50 steps:")
    print(f"  Top preference: {pref_state['top_preference']['system_id']} (weight={pref_state['top_preference']['weight']})")
    print(f"  Entropy: {pref_state['entropy']}")
    print(f"  Exploration temperature: {pref_state['exploration_temperature']}")
    print(f"  All weights: {pref_state['weights']}")
    print(f"  Revisit counts: {pref_state['revisit_counts']}")
    print()
    print("✓ All tests passed!")

if __name__ == "__main__":
    test_preference_system()
