import argparse
import csv
import json
import math
import cmath
from pathlib import Path
from typing import Dict, List, Tuple

from sparse_ruler.metrics import compute_W
from exp1a_pilot_summary import (
    ap_cover_scores,
    collision_anatomy,
    endpoint_participation,
    spike_mass,
)


def fourier_peaks(A: List[int], N: int, top_k: int = 10) -> List[Dict[str, float]]:
    indicator = [0] * (N + 1)
    for x in A:
        indicator[x] = 1
    m = len(A)
    peaks = []
    for k in range(1, N + 1):
        total = 0j
        for x in range(N + 1):
            if indicator[x]:
                angle = -2j * math.pi * k * x / (N + 1)
                total += cmath.exp(angle)
        magnitude = abs(total)
        peaks.append({"k": k, "magnitude": magnitude, "normalized": magnitude / m if m else 0.0})
    peaks.sort(key=lambda item: item["magnitude"], reverse=True)
    return peaks[:top_k]


def defect_localization(W: List[int], N: int) -> Tuple[int, float]:
    defects = [d for d in range(1, N + 1) if W[d] >= 3]
    if not defects:
        return 0, 0.0
    L = sum(min(d, N - d) for d in defects) / len(defects)
    return len(defects), L


def load_results(result_path: Path) -> Dict[str, str]:
    with result_path.open() as f:
        lines = [line.strip() for line in f if line.strip()]
    header = lines[0].split(",")
    row = lines[1].split(",")
    return dict(zip(header, row))


def pick_best_complete(rows: List[Tuple[Dict[str, str], Path]]) -> Tuple[Dict[str, str], Path]:
    def key(item):
        row, _ = item
        return (
            int(row["missing_count"]),
            int(row["E"]) - int(row["Emin"]),
            int(row["peak"]),
            int(row["count_w3"]) + int(row["count_w4"]) + int(row["count_w5"]) + int(row["count_w6plus"]),
            float(row["rho"]),
        )

    return sorted(rows, key=key)[0]


def summarize(root: Path, out_csv: Path, out_collisions: Path) -> None:
    summary_rows = []
    collisions = {}
    for N_dir in sorted(root.glob("N*_m*/")):
        N = int(N_dir.name.split("_")[0][1:])
        m = int(N_dir.name.split("_")[1][1:])
        rows = []
        for seed_dir in sorted(N_dir.glob("seed*/")):
            result_path = seed_dir / f"results_N{N}_m{m}.csv"
            if not result_path.exists():
                continue
            rows.append((load_results(result_path), seed_dir))
        if not rows:
            continue
        best_row, best_seed_dir = pick_best_complete(rows)
        gap = int(best_row["E"]) - int(best_row["Emin"])
        best_row["gap"] = gap
        summary_rows.append({"N": N, "m": m, **best_row})
        solution_path = best_seed_dir / "solutions" / f"solution_N{N}_m{m}_1.json"
        if solution_path.exists():
            data = json.loads(solution_path.read_text())
            W = compute_W(data["A"], N)
            d3_size, defect_L = defect_localization(W, N)
            collisions[f"N{N}_m{m}"] = {
                "A": data["A"],
                "spike_mass": spike_mass(W),
                "endpoint_participation": endpoint_participation(data["A"], N),
                "collisions": collision_anatomy(data["A"], N),
                "ap_cover_Lmin5": ap_cover_scores(data["A"], N, min_len=5),
                "ap_cover_Lmin7": ap_cover_scores(data["A"], N, min_len=7),
                "defect_D3_size": d3_size,
                "defect_L": defect_L,
                "fourier_peaks": fourier_peaks(data["A"], N),
            }

    if summary_rows:
        fields = [
            "N",
            "m",
            "missing_count",
            "peak",
            "gap",
            "rho",
            "spike_mass",
            "D3_size",
            "defect_L",
            "APcover_k3_Lmin5",
            "APcover_k3_Lmin7",
        ]
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in summary_rows:
                N = int(row["N"])
                m = int(row["m"])
                solution = collisions.get(f"N{N}_m{m}")
                d3_size = 0
                defect_L = 0.0
                ap5 = 0.0
                ap7 = 0.0
                spike = 0
                if solution:
                    d3_size = solution["defect_D3_size"]
                    defect_L = solution["defect_L"]
                    ap5 = solution["ap_cover_Lmin5"].get("APcover_k3", 0.0)
                    ap7 = solution["ap_cover_Lmin7"].get("APcover_k3", 0.0)
                    spike = solution["spike_mass"]
                writer.writerow(
                    {
                        "N": N,
                        "m": m,
                        "missing_count": row["missing_count"],
                        "peak": row["peak"],
                        "gap": row["gap"],
                        "rho": row["rho"],
                        "spike_mass": spike,
                        "D3_size": d3_size,
                        "defect_L": defect_L,
                        "APcover_k3_Lmin5": ap5,
                        "APcover_k3_Lmin7": ap7,
                    }
                )

    out_collisions.write_text(json.dumps(collisions, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-collisions", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.out_csv, args.out_collisions)


if __name__ == "__main__":
    main()
