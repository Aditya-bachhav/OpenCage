from __future__ import annotations

from pathlib import Path

from cage_env.runner import run_session
from cage_env.session_log import SessionLogger


def test_deterministic_session_logging(tmp_path: Path):
    log_a = tmp_path / "session_a.jsonl"
    log_b = tmp_path / "session_b.jsonl"

    result_a = run_session(seed=17, steps=120, log_path=log_a)
    result_b = run_session(seed=17, steps=120, log_path=log_b)

    assert log_a.exists()
    assert log_b.exists()

    rows_a = SessionLogger(log_a).read_all()
    rows_b = SessionLogger(log_b).read_all()

    assert rows_a[-1]["type"] == "summary"
    assert rows_b[-1]["type"] == "summary"
    assert rows_a[-1]["summary"] == rows_b[-1]["summary"]
    assert result_a["summary"] == result_b["summary"]

    step_rows = [row for row in rows_a if row["type"] == "step"]
    assert len(step_rows) > 0
    assert all("info" in row and "signals" in row["info"] for row in step_rows)
