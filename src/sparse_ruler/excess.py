from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

from .wichmann import generate_extended_wichmann, missing_distances, wichmann_upper_bound


@dataclass(frozen=True)
class ExcessRow:
    N: int
    m: int
    ratio: float
    m_sqrtN: float
    bound: int
    base_term: int
    E: int
    source: str
    family: str | None
    r: int | None
    s: int | None
    missing_count: int


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def base_term(length: int) -> int:
    if length < 1:
        raise ValueError("length must be >= 1")
    return round_half_up(math.sqrt(3 * length + 2.25))


def excess_from_m(length: int, m: int) -> int:
    return m - base_term(length)


def evaluate_length(length: int) -> ExcessRow:
    result = generate_extended_wichmann(length, validate=True)
    m = len(result.marks)
    ratio = m / math.sqrt(length)
    missing_count = len(missing_distances(result.marks, length))
    return ExcessRow(
        N=length,
        m=m,
        ratio=ratio,
        m_sqrtN=ratio,
        bound=wichmann_upper_bound(length),
        base_term=base_term(length),
        E=excess_from_m(length, m),
        source=result.source,
        family=result.family,
        r=result.r,
        s=result.s,
        missing_count=missing_count,
    )


def build_rows(start: int, end: int, step: int) -> List[ExcessRow]:
    if start < 1:
        raise ValueError("start must be >= 1")
    if end < start:
        raise ValueError("end must be >= start")
    if step < 1:
        raise ValueError("step must be >= 1")
    return [evaluate_length(length) for length in range(start, end + 1, step)]


def _row_to_csv_dict(row: ExcessRow) -> dict[str, object]:
    return {
        "N": row.N,
        "m": row.m,
        "ratio": row.ratio,
        "m/sqrt(N)": row.m_sqrtN,
        "bound": row.bound,
        "base_term": row.base_term,
        "E": row.E,
        "source": row.source,
        "family": row.family,
        "r": row.r,
        "s": row.s,
        "missing_count": row.missing_count,
    }


def run_excess_baseline(
    *,
    start: int,
    end: int,
    step: int,
    fixed_points: Sequence[int],
    out_dir: Path,
) -> dict[str, object]:
    rows = build_rows(start, end, step)
    fixed = [evaluate_length(N) for N in fixed_points]

    ratios = [row.ratio for row in rows]
    e_counts: dict[int, int] = {}
    for row in rows:
        e_counts[row.E] = e_counts.get(row.E, 0) + 1

    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "excess_table.csv"
    summary_path = out_dir / "excess_summary.json"

    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "N",
                "m",
                "ratio",
                "m/sqrt(N)",
                "bound",
                "base_term",
                "E",
                "source",
                "family",
                "r",
                "s",
                "missing_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_csv_dict(row))

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "range": {
            "start": start,
            "end": end,
            "step": step,
            "count": len(rows),
        },
        "fixed_points": [asdict(row) for row in fixed],
        "ratio": {
            "min": min(ratios),
            "max": max(ratios),
            "mean": sum(ratios) / len(ratios),
        },
        "E_distribution": dict(sorted(e_counts.items(), key=lambda x: x[0])),
        "complete_failures": [row.N for row in rows if row.missing_count != 0],
        "files": {
            "table": str(table_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def parse_fixed_points(values: Iterable[str]) -> List[int]:
    points: List[int] = []
    for value in values:
        for chunk in value.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            points.append(int(chunk))
    if not points:
        raise ValueError("at least one fixed point is required")
    return points
