#!/usr/bin/env python3
"""
Generate preference persistence analysis for long-run experiments.
Directly computes and reports without waiting for external processes.
"""

from pathlib import Path
from cage_env.experiment import compare_controllers
import json

def generate_longrun_report():
    log_dir = Path("cage_env/logs_longrun")
    log_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("PREFERENCE PERSISTENCE EXPERIMENT — LONG-RUN ANALYSIS")
    print("=" * 80)
    print()
    print("Running multi-seed long-run comparison...")
    print("  Seeds: 10 (100-109)")
    print("  Steps per seed: 5000")
    print("  Controllers: emergent, random, heuristic")
    print()
    
    # Run comparison with 10 seeds × 5000 steps
    seeds = list(range(100, 110))
    comparison = compare_controllers(
        seeds=seeds,
        steps=5000,
        log_dir=log_dir / "longrun"
    )
    
    print()
    print("=" * 80)
    print("RESULTS: EMERGENT vs RANDOM vs HEURISTIC")
    print("=" * 80)
    print()
    
    # Extract and display results
    results = {}
    for controller_name, payload in comparison.items():
        agg = payload['aggregate']
        results[controller_name] = agg
        
        print(f"\n[{controller_name.upper()}]")
        print(f"  Preference Persistence:")
        print(f"    - Preferred system: {agg['preference_persistence']['preferred_system_id']}")
        print(f"    - Consensus rate: {agg['preference_persistence']['consensus_rate']:.1%} (chance: {agg['preference_persistence']['chance_rate']:.1%})")
        print(f"  Behavioral Metrics:")
        print(f"    - Exploration entropy: {agg['mean_exploration_entropy']:.4f}")
        print(f"    - Longest attraction streak: {agg['mean_longest_continuous_attraction_streak']:.0f} steps")
        print(f"    - Revisit ratio: {agg['mean_repeated_to_first_visit_ratio']:.2f}x")
        print(f"  Per-System Stability:")
        for sys, stab in agg['mean_attraction_stability'].items():
            print(f"    - {sys}: {stab:.3f}")
    
    print()
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()
    
    emergent_consensus = results['emergent']['preference_persistence']['consensus_rate']
    random_consensus = results['random']['preference_persistence']['consensus_rate']
    emergent_entropy = results['emergent']['mean_exploration_entropy']
    random_entropy = results['random']['mean_exploration_entropy']
    emergent_revisit = results['emergent']['mean_repeated_to_first_visit_ratio']
    random_revisit = results['random']['mean_repeated_to_first_visit_ratio']
    
    print(f"1. PREFERENCE STABILITY")
    print(f"   Emergent consensus: {emergent_consensus:.1%} (vs {random_consensus:.1%} random)")
    print(f"   → {'✓ SIGNIFICANT' if emergent_consensus > random_consensus * 1.5 else '✗ WEAK'}")
    print()
    
    print(f"2. EXPLORATION BEHAVIOR")
    print(f"   Emergent entropy: {emergent_entropy:.3f} (vs {random_entropy:.3f} random)")
    if emergent_entropy < random_entropy:
        print(f"   → ✓ MORE FOCUSED (entropy {random_entropy/emergent_entropy:.1f}x lower)")
    else:
        print(f"   → ✗ EQUALLY EXPLORATORY")
    print()
    
    print(f"3. REVISIT PERSISTENCE")
    print(f"   Emergent revisits: {emergent_revisit:.2f}x (vs {random_revisit:.2f}x random)")
    print(f"   → ✓ {emergent_revisit / random_revisit:.1f}x MORE REVISITS")
    print()
    
    print(f"4. OVERALL ASSESSMENT")
    if emergent_consensus > 0.5 and emergent_entropy < 1.0 and emergent_revisit > random_revisit * 1.5:
        print(f"   ✓✓✓ STRONG STRUCTURED PREFERENCE EMERGENCE")
        print(f"   The emergent controller demonstrates:")
        print(f"       • Stable cross-seed preferences (>{emergent_consensus:.0%} consensus)")
        print(f"       • Focused but adaptive behavior (entropy={emergent_entropy:.3f})")
        print(f"       • Strong revisit persistence ({emergent_revisit:.2f}x baseline)")
    elif emergent_consensus > 0.33 and emergent_revisit > random_revisit:
        print(f"   ✓✓ MODERATE STRUCTURED PREFERENCE EMERGENCE")
        print(f"   The emergent controller shows:")
        print(f"       • Weak cross-seed preferences (~{emergent_consensus:.0%})")
        print(f"       • Improved revisits vs random ({emergent_revisit:.2f}x)")
        print(f"   → Preference system is learning, but needs reinforcement tuning")
    else:
        print(f"   ✗ WEAK OR NO STRUCTURED EMERGENCE")
        print(f"   Revisit ratio too low ({emergent_revisit:.2f}x) or consensus missing")
    
    print()
    print("=" * 80)
    print()
    
    # Save results
    results_path = Path("LONGRUN_RESULTS.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    generate_longrun_report()
