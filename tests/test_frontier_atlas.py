from __future__ import annotations

import json
from pathlib import Path

from sparse_ruler.frontier_atlas import build_frontier_atlas_payload, generate_frontier_atlas


def _write_summary(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2))
    return path


def test_frontier_atlas_captures_terminal_run_geometry(tmp_path: Path) -> None:
    cycle1 = {
        "targets": [
            {
                "N": 20,
                "m_try": 10,
                "best_overall": {
                    "stage": "screen",
                    "candidate_id": "cand-a",
                    "deleted_mark": 19,
                    "stage_best_missing_count": 1,
                    "stage_best_missing_list": [11],
                    "stage_best_marks": [0, 1, 2, 3, 4, 5, 6, 12, 14, 15, 16, 17, 18, 19, 20],
                },
            }
        ]
    }
    cycle2 = {
        "targets": [
            {
                "N": 20,
                "m_try": 10,
                "best_overall": {
                    "stage": "deep",
                    "candidate_id": "cand-b",
                    "deleted_mark": 19,
                    "stage_best_missing_count": 1,
                    "stage_best_missing_list": [11],
                    "stage_best_marks": [0, 1, 2, 3, 4, 5, 6, 12, 14, 15, 16, 17, 18, 19, 20],
                },
            }
        ]
    }

    cycle1_path = _write_summary(tmp_path / "cycle1.json", cycle1)
    cycle2_path = _write_summary(tmp_path / "cycle2.json", cycle2)

    payload = build_frontier_atlas_payload(
        cycle1_summary_path=cycle1_path,
        cycle2_summary_path=cycle2_path,
    )
    record = payload["records"][0]
    assert record["terminal_run"] == [14, 15, 16, 17, 18, 19]
    assert record["endpoint_gap"] == 1
    assert record["run_prefix_gap"] == 2
    assert record["furthest_missing_gap"] == 9

    out_md = tmp_path / "atlas.md"
    out_json = tmp_path / "atlas.json"
    generate_frontier_atlas(
        cycle1_summary_path=cycle1_path,
        cycle2_summary_path=cycle2_path,
        out_path=out_md,
        json_out_path=out_json,
    )

    assert out_md.exists()
    assert out_json.exists()
    saved = json.loads(out_json.read_text())
    assert saved["motifs"][0]["shared_missing"] is True
    assert saved["motifs"][0]["shared_terminal_run"] is True
