from __future__ import annotations

from pathlib import Path

from cage_env.experiment import compare_controllers, load_session_rows, run_session


def _step_obs(rows: list[dict]) -> list[dict]:
    return [row["obs"] for row in rows if row.get("type") == "step"]


def test_reproducible_runs_and_controller_comparison(tmp_path: Path):
    log_a = tmp_path / "run_a.jsonl"
    log_b = tmp_path / "run_b.jsonl"

    result_a = run_session(seed=21, steps=80, log_path=log_a, controller_name="emergent")
    result_b = run_session(seed=21, steps=80, log_path=log_b, controller_name="emergent")

    rows_a = load_session_rows(log_a)
    rows_b = load_session_rows(log_b)

    assert _step_obs(rows_a) == _step_obs(rows_b)
    assert result_a["summary"] == result_b["summary"]

    comparison = compare_controllers(seeds=[21, 22], steps=40, log_dir=tmp_path / "comparison")
    assert set(comparison) == {"emergent", "random", "heuristic"}
    assert all("aggregate" in payload for payload in comparison.values())
    assert all("preference_persistence" in payload["aggregate"] for payload in comparison.values())