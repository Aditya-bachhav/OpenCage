# Cage: Oscillation Chamber

The project now centers on one physically grounded phenomenon: periodic motion in a persistent world.

## Active modules

- `cage_env/env.py`: the `OscillationChamberEnv` Gymnasium environment
- `cage_env/objects.py`: oscillating systems and agent factory
- `cage_env/physics.py`: phase, damping, and energy transfer
- `cage_env/measurements.py`: raw signals and physically grounded metrics
- `cage_env/controller.py`: simple deterministic-probabilistic controller
- `cage_env/evaluate.py`: long-run deterministic session runner and JSONL logger
- `cage_env/visualize.py`: live view and replay tool for recorded sessions
- `cage_env/session_log.py`: JSONL session log helper
- `tests/`: reproducibility/logging test

## Run a long session

```bash
python -m cage_env.evaluate --seed 42 --steps 1000
```

## Visualize live

```bash
python -m cage_env.visualize
```

## Replay a recorded run

```bash
python -m cage_env.visualize --replay cage_env/logs/session_42_YYYYMMDD_HHMMSS.jsonl
```

## Test

```bash
pytest
```

## Goal

Prove whether the agent develops persistent attraction patterns through revisit count, dwell time, phase alignment, and synchronization attempts.
