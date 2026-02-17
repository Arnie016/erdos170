from __future__ import annotations

import json
from pathlib import Path

from sparse_ruler.e_hunt import build_endpoint_window_candidates, run_e_hunt_from_config_path
from sparse_ruler.search import RulerState


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _tiny_search(enabled_stop_global: bool = False) -> dict:
    return {
        "log_interval": 0,
        "save_best_every": 0,
        "skip_championship_if_complete": True,
        "stop_on_first_complete_global": enabled_stop_global,
        "resume_existing_candidates": True,
        "stages": {
            "screen": {
                "enabled": True,
                "seeds": 1,
                "iterations_per_seed": 0,
                "promote_top_k": 1,
            },
            "deep": {"enabled": False, "seeds": 1, "iterations_per_seed": 0, "promote_top_k": 1},
            "championship": {"enabled": False, "seeds": 1, "iterations_per_seed": 0},
        },
    }


def test_allowlist_filters_candidates_and_preserves_slot_order(tmp_path: Path) -> None:
    root_dir = tmp_path / "out"
    cfg = {
        "targets": [
            {
                "N": 10,
                "m_try": 5,
                "start_marks": [0, 1, 2, 4, 7, 10],
                "candidate_allowlist_deleted_marks": [7, 1],
            }
        ],
        "search": _tiny_search(enabled_stop_global=False),
        "outputs": {"root_dir": str(root_dir), "save_trace": False},
    }

    summary = run_e_hunt_from_config_path(_write_config(tmp_path, cfg))
    target = summary["targets"][0]
    assert target["deletion_candidates_count"] == 2

    rows = target["stages"][0]["rows"]
    deleted_marks = {int(row["deleted_mark"]) for row in rows}
    assert deleted_marks == {1, 7}

    seeds_by_mark = {int(row["deleted_mark"]): int(row["evaluated_seeds"][0]) for row in rows}
    assert seeds_by_mark[7] - seeds_by_mark[1] == 1000


def test_global_early_stop_halts_remaining_targets(tmp_path: Path) -> None:
    root_dir = tmp_path / "out"
    cfg = {
        "targets": [
            {
                "N": 6,
                "m_try": 6,
                "start_marks": [0, 1, 2, 3, 4, 5, 6],
                "candidate_allowlist_deleted_marks": [1],
            },
            {
                "N": 7,
                "m_try": 7,
                "start_marks": [0, 1, 2, 3, 4, 5, 6, 7],
                "candidate_allowlist_deleted_marks": [1],
            },
        ],
        "search": _tiny_search(enabled_stop_global=True),
        "outputs": {"root_dir": str(root_dir), "save_trace": False},
    }

    summary = run_e_hunt_from_config_path(_write_config(tmp_path, cfg))
    assert summary["overall"]["stopped_early_global"] is True
    assert summary["overall"]["target_count"] == 1
    assert summary["overall"]["success_targets"] == [6]


def test_endpoint_window_candidate_pool_targets_tail_and_excludes_occupied() -> None:
    state = RulerState(100, [0, 1, 2, 100])
    candidates = build_endpoint_window_candidates(
        state,
        tail_missing=[90, 95, 99],
        tail_window_k=16,
        low_anchor_max=12,
    )

    assert 10 in candidates
    assert 5 in candidates
    assert 99 in candidates
    assert 0 not in candidates
    assert 100 not in candidates
    assert 1 not in candidates
    assert 2 not in candidates


def test_smoke_outputs_include_new_cycle2_metrics(tmp_path: Path) -> None:
    root_dir = tmp_path / "out"
    cfg = {
        "targets": [
            {
                "N": 12,
                "m_try": 8,
                "start_marks": [0, 1, 2, 3, 4, 5, 9, 11, 12],
                "candidate_allowlist_deleted_marks": [1],
            }
        ],
        "moves": {
            "endpoint_window_mode": {
                "enabled": True,
                "trigger_best_missing_at_most": 12,
                "trigger_stall_steps": 0,
                "tail_window_k": 6,
                "low_anchor_max": 4,
                "proposal_prob": 1.0,
            }
        },
        "search": {
            "log_interval": 0,
            "save_best_every": 0,
            "skip_championship_if_complete": True,
            "stop_on_first_complete_global": False,
            "resume_existing_candidates": True,
            "stages": {
                "screen": {
                    "enabled": True,
                    "seeds": 1,
                    "iterations_per_seed": 5,
                    "promote_top_k": 1,
                },
                "deep": {"enabled": False, "seeds": 1, "iterations_per_seed": 0, "promote_top_k": 1},
                "championship": {"enabled": False, "seeds": 1, "iterations_per_seed": 0},
            },
        },
        "outputs": {"root_dir": str(root_dir), "save_trace": False},
    }

    summary = run_e_hunt_from_config_path(_write_config(tmp_path, cfg))
    assert (root_dir / "summary.json").exists()
    assert summary["config"]["search"]["stop_on_first_complete_global"] is False

    row = summary["targets"][0]["stages"][0]["rows"][0]
    for key in ("endpoint_window_attempts", "endpoint_window_accepts", "tail_trigger_count"):
        assert key in row
        assert isinstance(row[key], int)

    best_paths = list(root_dir.rglob("best.json"))
    assert best_paths, "expected at least one best.json artifact"
    seed_payload = json.loads(best_paths[0].read_text())
    for key in ("endpoint_window_attempts", "endpoint_window_accepts", "tail_trigger_count"):
        assert key in seed_payload
        assert isinstance(seed_payload[key], int)
