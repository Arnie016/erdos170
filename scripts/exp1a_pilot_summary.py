import csv
import importlib.util
import json
import math
import cmath
from pathlib import Path
from typing import Dict, List, Tuple

from sparse_ruler.metrics import compute_W

BASE = Path("outputs/exp1a_pilot")
SUMMARY_PATH = BASE / "summary.csv"
COLLISIONS_PATH = BASE / "collisions.json"
PLOTS_DIR = BASE / "plots"


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
            int(row["peak"]),
            int(row["E"]),
            float(row["rho"]),
        )

    return sorted(rows, key=key)[0]


def collision_anatomy(A: List[int], N: int) -> Dict[str, Dict[str, List[List[int]]]]:
    W = compute_W(A, N)
    collisions = {}
    for d in range(1, N + 1):
        if W[d] >= 3:
            pairs = []
            for i, a in enumerate(A):
                for b in A[:i]:
                    if a - b == d:
                        pairs.append([a, b])
            U = sorted({a for a, _ in pairs})
            V = sorted({b for _, b in pairs})
            collisions[str(d)] = {
                "pairs": pairs,
                "U": U,
                "V": V,
            }
    return collisions


def spike_mass(W: List[int]) -> int:
    return sum(max(w - 2, 0) for w in W[1:])


def endpoint_participation(A: List[int], N: int) -> Dict[str, List[int]]:
    W = compute_W(A, N)
    participation = {x: set() for x in A}
    for d in range(1, N + 1):
        if W[d] >= 3:
            for i, a in enumerate(A):
                for b in A[:i]:
                    if a - b == d:
                        participation[a].add(d)
                        participation[b].add(d)
    ranked = sorted(participation.items(), key=lambda kv: len(kv[1]), reverse=True)
    top5 = [{"mark": mark, "spike_distances": sorted(list(ds))} for mark, ds in ranked[:5]]
    return {"top5": top5}


def greedy_ap_cover(A: List[int], N: int, k: int, min_len: int = 2) -> Dict[str, object]:
    marks = sorted(A)
    mark_set = set(marks)
    uncovered = set(marks)
    selections = []
    for _ in range(k):
        best = {"count": 0, "start": None, "step": None, "covered": []}
        if not uncovered:
            break
        marks_list = sorted(uncovered)
        for i in range(len(marks_list)):
            for j in range(i + 1, len(marks_list)):
                step = marks_list[j] - marks_list[i]
                if step == 0:
                    continue
                start = marks_list[i]
                covered = [start]
                next_val = start + step
                while next_val in mark_set:
                    if next_val in uncovered:
                        covered.append(next_val)
                    next_val += step
                if len(covered) > best["count"]:
                    best = {"count": len(covered), "start": start, "step": step, "covered": covered}
        if best["count"] < min_len:
            break
        selections.append(best)
        uncovered -= set(best["covered"])
    covered_count = len(marks) - len(uncovered)
    return {
        "covered": covered_count,
        "fraction": covered_count / len(marks) if marks else 0.0,
        "selections": selections,
    }


def ap_cover_scores(A: List[int], N: int, min_len: int = 2) -> Dict[str, object]:
    scores = {}
    selections = None
    for k in range(1, 5):
        result = greedy_ap_cover(A, N, k, min_len=min_len)
        scores[f"APcover_k{k}"] = result["fraction"]
        if k == 3:
            selections = result["selections"]
    if selections is None:
        selections = []
    ap_params = [
        {
            "start": sel["start"],
            "step": sel["step"],
            "count": sel["count"],
        }
        for sel in selections
    ]
    scores["AP_params_k3"] = ap_params
    return scores


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


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    collisions = {}
    for N_dir in sorted(BASE.glob("N*_m*/")):
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
        summary_rows.append({"N": N, "m": m, **best_row})
        solution_path = best_seed_dir / "solutions" / f"solution_N{N}_m{m}_1.json"
        if solution_path.exists():
            data = json.loads(solution_path.read_text())
            W = compute_W(data["A"], N)
            ap_scores = ap_cover_scores(data["A"], N)
            fourier = fourier_peaks(data["A"], N)
            collisions[f"N{N}_m{m}"] = {
                "A": data["A"],
                "spike_mass": spike_mass(W),
                "endpoint_participation": endpoint_participation(data["A"], N),
                "collisions": collision_anatomy(data["A"], N),
                "ap_cover": ap_scores,
                "fourier_peaks": fourier,
            }

    if summary_rows:
        fields = [
            "N",
            "m",
            "S",
            "pair_excess",
            "missing_count",
            "peak",
            "H3",
            "H4",
            "E",
            "Emin",
            "rho",
            "count_w1",
            "count_w2",
            "count_w3",
            "count_w4",
            "count_w5",
            "count_w6plus",
            "spike_mass",
            "spike_mass_per_N",
            "spike_mass_per_excess",
            "gap",
            "APcover_k1",
            "APcover_k2",
            "APcover_k3",
            "APcover_k4",
        ]
        with SUMMARY_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in summary_rows:
                try:
                    N = int(row["N"])
                    m = int(row["m"])
                    pair_excess = int(row["S"]) - N
                    row["pair_excess"] = pair_excess
                    row["gap"] = int(row["E"]) - int(row["Emin"])
                except Exception:
                    row["pair_excess"] = ""
                    row["gap"] = ""
                if "spike_mass" not in row:
                    try:
                        N = int(row["N"])
                        m = int(row["m"])
                        solution_path = BASE / f"N{N}_m{m}" / "seed1" / "solutions" / f"solution_N{N}_m{m}_1.json"
                        if solution_path.exists():
                            data = json.loads(solution_path.read_text())
                            row["spike_mass"] = spike_mass(compute_W(data["A"], N))
                    except Exception:
                        row["spike_mass"] = ""
                try:
                    spike_value = float(row["spike_mass"])
                    N = int(row["N"])
                    row["spike_mass_per_N"] = spike_value / N if N > 0 else ""
                    excess = int(row.get("pair_excess", 0))
                    row["spike_mass_per_excess"] = spike_value / excess if excess > 0 else ""
                except Exception:
                    row["spike_mass_per_N"] = ""
                    row["spike_mass_per_excess"] = ""
                if "APcover_k1" not in row:
                    try:
                        N = int(row["N"])
                        m = int(row["m"])
                        solution_path = BASE / f"N{N}_m{m}" / "seed1" / "solutions" / f"solution_N{N}_m{m}_1.json"
                        if solution_path.exists():
                            data = json.loads(solution_path.read_text())
                            ap_scores = ap_cover_scores(data["A"], N)
                            row.update(ap_scores)
                    except Exception:
                        pass
                writer.writerow({key: row.get(key, "") for key in fields})

    COLLISIONS_PATH.write_text(json.dumps(collisions, indent=2))

    if importlib.util.find_spec("matplotlib") is None:
        return
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    grouped: Dict[int, List[Dict[str, str]]] = {}
    for row in summary_rows:
        grouped.setdefault(int(row["m"]), []).append(row)

    for m, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda r: int(r["N"]))
        Ns = [int(r["N"]) for r in rows_sorted]
        rho = [float(r["rho"]) for r in rows_sorted]
        peak = [int(r["peak"]) for r in rows_sorted]
        frac3plus = [
            (
                int(r["count_w3"]) + int(r["count_w4"]) + int(r["count_w5"]) + int(r["count_w6plus"])
            )
            / int(r["N"])
            for r in rows_sorted
        ]

        plt.figure()
        plt.plot(Ns, rho, marker="o")
        plt.xlabel("N")
        plt.ylabel("rho")
        plt.title(f"rho vs N (m={m})")
        plt.savefig(PLOTS_DIR / f"rho_m{m}.png", dpi=150)
        plt.close()

        plt.figure()
        plt.plot(Ns, peak, marker="o")
        plt.xlabel("N")
        plt.ylabel("peak")
        plt.title(f"peak vs N (m={m})")
        plt.savefig(PLOTS_DIR / f"peak_m{m}.png", dpi=150)
        plt.close()

        plt.figure()
        plt.plot(Ns, frac3plus, marker="o")
        plt.xlabel("N")
        plt.ylabel("frac3plus")
        plt.title(f"frac3plus vs N (m={m})")
        plt.savefig(PLOTS_DIR / f"frac3plus_m{m}.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
