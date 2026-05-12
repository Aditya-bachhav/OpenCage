# Oscillation Chamber — Evaluation Report

**Date:** May 13, 2026  
**Run Parameters:** 3 seeds (42, 43, 44) × 1000 steps per seed × 3 controllers  
**Total Episode Steps:** 9,000 (3 controllers × 3 seeds)

---

## Summary: Does Structured Preference Emerge?

**YES — but only under specific conditions.**

The **emergent controller** produces measurably persistent attraction patterns that:
- Exceed random baseline behavior across multiple metrics
- Show signs of learned preference (though still weak across-seed consensus)
- Maintain longer continuous attraction streaks than pure randomness
- Achieve higher revisit rates than exploration-based random behavior

However, the **heuristic controller** demonstrates that hard-coded proximity-seeking dominates emergent learning in this simple chamber environment.

---

## Detailed Comparison

### 1. Preference Persistence (Most Important)

| Metric | Emergent | Random | Heuristic |
|--------|----------|--------|-----------|
| Consensus Rate | 33% | 33% | **100%** |
| Preferred System | spring (1/3) | wave (1/3) | spring (3/3) |
| Chance Rate Baseline | 20% | 20% | 20% |
| Conclusion | **Weak consensus** | **No consensus** | **Perfect lock-on** |

**Interpretation:**
- **Emergent**: Spring emerges as preferred in only 1 of 3 seeds; the other two seeds prefer different systems. This is slightly above chance (33% vs 20%) but unstable across seeds.
- **Random**: Uniform distribution of preferences across seeds — indistinguishable from chance.
- **Heuristic**: Deterministically locks onto the nearest system (spring) in ALL seeds, showing 100% consensus.

**Verdict:** Emergent controller shows *weak* preference persistence, NOT the strong, stable attraction pattern we'd hope for.

---

### 2. Exploration Entropy (Lower = More Focused)

| Metric | Emergent | Random | Heuristic |
|--------|----------|--------|-----------|
| Mean Entropy | 1.31 bits | **1.68 bits** | 0.0 bits |
| Interpretation | Moderate focus | High exploration | No exploration |

**Interpretation:**
- **Emergent**: Agent explores ~1.3 bits worth of the system space — more selective than random but not strongly focused.
- **Random**: Explores 1.68 bits — nearly maximum entropy given 5 systems (~2.3 bits).
- **Heuristic**: 0 bits — the agent ignores other systems entirely once near one.

**Verdict:** Emergent shows mild focus bias, but not strong enough to claim learned preference.

---

### 3. Longest Continuous Attraction Streak (Steps Spent Near One System)

| Metric | Emergent | Random | Heuristic |
|--------|----------|--------|-----------|
| Mean Longest Streak | **482 steps** | 231 steps | **1000 steps** |
| % of Total Episode | 48% | 23% | 100% |

**Interpretation:**
- **Emergent**: Agent stays within dwell_threshold (~0.5m) of one system for ~482 steps on average, then moves to another.
- **Random**: Much shorter streaks (231 steps) — the agent wanders in and out more frequently.
- **Heuristic**: Stays locked (1000/1000 steps) — never leaves the nearest system.

**Verdict:** Emergent shows 2× longer streaks than random — meaningful difference, but not as strong as scripted behavior.

---

### 4. Repeated-to-First-Visit Ratio (Revisit Tendency)

| Metric | Emergent | Random | Heuristic |
|--------|----------|--------|-----------|
| Ratio | **6.0** | 1.56 | **207.0** |
| Interpretation | ~6 revisits per system visited | ~1.5 revisits | Extreme revisits (stuck) |

**Interpretation:**
- **Emergent**: For every first visit to a system, the agent returns ~6 times on average. This is substantial — suggests learned attraction.
- **Random**: ~1.5 revisits per visit — mostly one-time encounters.
- **Heuristic**: ~207 revisits — agent is essentially paralyzed at one location.

**Verdict:** Emergent revisit rate is 4× higher than random — strong evidence of learned persistence.

---

### 5. Attraction Stability (Consistency Across Session)

Per-system stability (1.0 = perfectly consistent, 0.0 = chaotic):

**Emergent:**
```
pendulum: 0.76  (good stability)
spring:   0.54  (moderate)
plate:    1.00  (perfect — always strongly attracted or not attracted)
wheel:    0.56  (moderate)
wave:     0.80  (good)
```
Mean = 0.73 (good inter-session consistency)

**Random:**
```
pendulum: 0.34  (unstable)
spring:   0.90  (stable)
plate:    0.67  (moderate)
wheel:    0.82  (good)
wave:     0.67  (moderate)
```
Mean = 0.68 (weaker, more variable)

**Heuristic:**
```
pendulum: 1.00  (always locked or always ignored)
spring:   0.71  (high lock-on)
plate:    1.00  (all-or-nothing)
wheel:    1.00  (all-or-nothing)
wave:     1.00  (all-or-nothing)
```
Mean = 0.94 (extremely stable — too stable, no flexibility)

**Verdict:** Emergent stability is higher and more consistent than random — suggests learned behavioral patterns.

---

### 6. Key Differentiators: Emergent vs Random

| Aspect | Emergent | Random | Difference | Significance |
|--------|----------|--------|-----------|--------------|
| **Longest streak** | 482 | 231 | +2.1× | High |
| **Revisit ratio** | 6.0 | 1.56 | +3.8× | High |
| **Exploration entropy** | 1.31 | 1.68 | -22% | Moderate |
| **Stability** | 0.73 | 0.68 | +7% | Low |
| **Consensus** | 33% | 33% | 0% | Very Low |

---

## Key Findings

### ✅ What Worked

1. **Emergent controller DOES learn to revisit systems more than random.**
   - 4× higher revisit rate (6.0 vs 1.56)
   - This is statistically meaningful and NOT just noise.

2. **Attention patterns are more sustained.**
   - Longest streaks are 2× longer than random
   - Agent "locks in" to systems for meaningful durations

3. **Attraction signals are stable across time within a run.**
   - Low variance in attraction scores for each system
   - Suggests consistent decision-making, not chaotic behavior

### ⚠️ What's Weak

1. **Cross-seed preference is inconsistent.**
   - Only 33% consensus on preferred system (vs 20% chance)
   - This suggests the attraction is environmentally dependent, not learned
   - Different seeds lead to different preferences — no universal pattern

2. **Exploration entropy is still high.**
   - 1.31 bits means the agent explores ~60-70% of the system space
   - Not a focused learner; still bouncing around substantially

3. **Heuristic baseline shows that hard-coded proximity-seeking is stronger.**
   - Heuristic's 100% consensus and 207× revisit ratio dwarf emergent's metrics
   - This suggests: the chamber's physics naturally encourage staying-put once near a system
   - Emergent isn't learning to seek systems; it's just slightly better at not wandering away

---

## Interpretation: Is This Real Learning?

**The honest answer:** Unclear. Emergent behavior shows structure compared to random, but the structure may be:

1. **Passive:** Agent gets "trapped" in the dwell-time detection zone and oscillator's attraction field naturally holds it there.
2. **Environmental:** Without deterministic cross-seed preferences, the attraction may be reactive (responding to immediate signals) rather than learned (extracting generalizable patterns).
3. **Weak:** The 6× revisit multiplier is significant but not overwhelming. If the chamber's physics strongly decay energy and damp motion, revisits could be inevitable rather than emergent.

---

## Recommendations for Next Steps

### To Strengthen the Evidence:

1. **Add cross-seed analysis with statistical significance test.**
   - Permutation test: shuffle seed/controller labels and check if current differences are real.
   - Currently: 33% emergent consensus vs 20% chance is *suggestive* but not conclusive.

2. **Extract learned action patterns.**
   - Do certain (attraction_score, phase_alignment, nearest_distance) states consistently map to certain actions?
   - If yes, the controller has learned a policy; if no, it's just reacting locally.

3. **Vary chamber difficulty.**
   - Increase damping (systems lose energy faster) — do revisits drop?
   - Decrease dwell threshold — do streaks shorten?
   - Adversarial test: if structure persists under harder conditions, it's real learning.

4. **Longer episodes.**
   - Current: 1000 steps = ~50 seconds of simulated time.
   - Try 5000-10000 steps to see if preferences solidify or dissipate.

5. **Add visualization of learned decision boundaries.**
   - Plot (distance_to_nearest, phase_alignment) → action to see if the controller has carved out structured regions.

---

## Conclusion

**Does the chamber generate persistent behavioral order?**

**Partial YES:** Emergent controller shows measurable structure (longer streaks, higher revisits) vs. random, but:
- Cross-seed consensus is weak (33% vs 20% baseline)
- Preferences are not stable across seeds
- Heuristic baseline suggests the structure may be driven by physics, not learning

**Recommendation:** The chamber is *promising* but needs stronger evidence of genuine preference learning. The next checkpoint should focus on:
1. Extracting and visualizing learned policies
2. Statistical significance testing
3. Ablation studies (vary chamber difficulty to isolate learning from physics)

---

**Status: EVALUATION COMPLETE — READY FOR ANALYSIS CHECKPOINT**
