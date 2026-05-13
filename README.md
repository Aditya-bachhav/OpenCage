# Cage — Oscillation Chamber

Purpose
-------
The Oscillation Chamber ("Cage") is a minimal, physically grounded testbed for studying persistent attraction and preference formation in a persistent world whose dynamics are dominated by oscillatory motion. The environment focuses on raw, reproducible signals (trajectories, attraction scores, phase alignment, energy transfers) rather than hand-labeled behaviors.

Repository layout
-----------------
- `main.py` — top-level launcher: runs controller comparisons and (optionally) opens the visualizer.
- `requirements.txt` — runtime dependencies used by this project.
- `README.md` — this document.
- `.gitignore` — ignore patterns for generated artifacts.
- `tests/` — pytest-based unit tests validating determinism and the experiment workflow.
- `cage_env/` — core package:
	- `cage_env/env.py` — `OscillationChamberEnv` (Gymnasium-compatible environment). Entry point for simulation steps and observations.
	- `cage_env/objects.py` — `OscillatingSystem` dataclass and factories (`make_pendulum`, `make_spring`, `make_resonance_plate`, `make_rotating_wheel`, `make_wave_emitter`, `make_agent`).
	- `cage_env/physics.py` — low-level physics: phase update, positional transforms, energy decay, `agent_interact`, and utility functions like `phase_difference`.
	- `cage_env/measurements.py` — `MeasurementEngine` and `SystemMetrics`: computes raw signals and summary metrics (visits, dwell time, revisit intervals, phase alignment, attraction scores).
	- `cage_env/policy.py` — `Policy` protocol plus `RandomPolicy`, `HeuristicPolicy`, and `EmergentPolicy`.
	- `cage_env/runner.py` — policy-driven session runner, summarization, and cross-seed comparison harness.
	- `cage_env/session_log.py` — `SessionLogger`: append-only JSONL session records (per-step + summary).
	- `cage_env/visualize.py` — live visualizer and replay UI (pygame) showing systems, trajectory, attraction bars and metrics.
	- `cage_env/experiment.py` — compatibility wrapper that re-exports the runner API.
	- `cage_env/evaluate.py` — CLI wrapper for the experiment harness (convenience entrypoint).

Quick start
-----------
1. Create and activate a Python 3.10+ virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Run a short reproducible session (writes a JSONL log):

```bash
python -m cage_env.evaluate --policy emergent --seeds 42 --steps 240 --log-dir cage_env/logs/emergent
```

3. Compare controllers across multiple seeds (aggregates metrics):

```bash
python -m cage_env.evaluate --policy all --seeds 42 43 44 --steps 240
```

4. Run the launcher which performs the comparison then opens the visualizer:

```bash
python main.py
```

5. Replay a recorded run in the visualizer:

```bash
python -m cage_env.visualize --replay cage_env/logs/emergent/emergent_seed_42.jsonl
```

Logging and reproducibility
---------------------------
- Sessions are written as JSONL via `cage_env/session_log.py`. Each step writes a `type: "step"` record containing `obs`, `info`, `action`, and `reward`. A final `type: "summary"` row contains aggregated metrics.
- Determinism is implemented by seeding both `random` and `numpy` (see `OscillationChamberEnv.reset`). Experiments use fixed seeds per-run to allow exact replay.

Core signals and metrics
------------------------
- Raw signals (exposed per-step in `info["signals"]`):
	- `position`: agent coordinates
	- `distance_to_nearest`: distance to nearest oscillating system
	- `velocity`: instantaneous velocity estimate
	- `attraction_scores`: per-system attraction (derived from visits × dwell time)
	- `revisit_frequencies`: visits per 100 steps
	- `phase_alignment`: empirical alignment between visit-time phase samples and the system's phase
	- `system_energies` / `system_phases`: direct physical state

- Per-system summary metrics (from `MeasurementEngine.summary()`):
	- `visits`, `dwell_time`, `revisit_count`, `phase_alignment`, `synchronization_attempts`, `attraction_score`.

Experiment analysis
-------------------
- `cage_env/experiment.py` implements:
	- `run_session(seed, steps, log_path, controller_name)` — run one seeded session and write a JSONL log.
	- `run_multi_seed(controller_name, seeds, steps, log_dir)` — run the same controller across multiple seeds.
	- `summarize_session(rows)` — fold a JSONL session into interpretable metrics (exploration entropy, longest attraction streak, stability per system, repeated-to-first-visit ratio, preferred system and consensus rate).
	- `aggregate_runs(results)` — combine multiple seeded runs into an aggregate dictionary used for controller comparison.

How to interpret results
------------------------
- Preference persistence / consensus rate — how often a given system emerges as the preferred target across seeds.
- Attraction stability — how stable the attraction score for a system is across the session (low stddev → high stability).
- Exploration entropy — how widely the agent samples different systems.
- Dwell time & revisit counts — direct evidence of persistent attraction.

Development notes & extending the project
---------------------------------------
- Add new policies by implementing `reset()`, `act(observation)`, and optional `update(feedback)` in `cage_env/policy.py`, then register them in `cage_env/runner.py::make_policy`.
- Add new systems by creating a factory in `cage_env/objects.py` and ensuring `physics.tick` interprets its `kind`.
- To attach richer rewards, edit `OscillationChamberEnv._compute_reward` in `cage_env/env.py`; keep in mind experiments rely on deterministic seeds.

Testing
-------
- Run the test suite:

```bash
pytest -q
```

The project contains deterministic tests (`tests/test_session_logging.py`, `tests/test_experiment_workflow.py`) which validate that identical seeds produce identical session summaries and logs.

Troubleshooting
---------------
- If the visualizer is blank or pygame fails to start, ensure a display is available and that `pygame` is installed in the active environment.
- If runs appear nondeterministic, confirm the same seed is passed and no external state modifies RNGs; `OscillationChamberEnv.reset` seeds both `random` and `numpy`.

Contact & next steps
--------------------
If you'd like, I can:
- Add automated statistical tests (permutation / bootstrap) to quantify significance of controller differences.
- Export aggregated CSV/JSON summaries for external analysis.
- Add a small notebook demonstrating analysis and plots of attraction persistence across seeds.

Happy to continue — tell me which follow-up you want next.
