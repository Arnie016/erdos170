from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class CNFBundle:
    n_vars: int
    clauses: List[List[int]]
    x_vars: Dict[int, int]


def _new_var(counter: List[int]) -> int:
    counter[0] += 1
    return counter[0]


def build_sparse_ruler_cnf(N: int, m: int) -> CNFBundle:
    # x_i indicates whether mark i is selected.
    counter = [0]
    x_vars: Dict[int, int] = {}
    for i in range(N + 1):
        x_vars[i] = _new_var(counter)

    clauses: List[List[int]] = []

    # Endpoint constraints: 0 and N must be marks.
    clauses.append([x_vars[0]])
    clauses.append([x_vars[N]])

    # Coverage auxiliaries: y_{d,i} <-> (x_i and x_{i-d}) for i=d..N.
    # For each distance d, require OR_i y_{d,i}.
    for d in range(1, N + 1):
        y_vars: List[int] = []
        for i in range(d, N + 1):
            y = _new_var(counter)
            y_vars.append(y)
            xi = x_vars[i]
            xj = x_vars[i - d]
            # y -> xi
            clauses.append([-y, xi])
            # y -> xj
            clauses.append([-y, xj])
            # (xi and xj) -> y
            clauses.append([y, -xi, -xj])
        clauses.append(y_vars)

    # Exact-cardinality constraint Sum x_i == m.
    try:
        from pysat.card import CardEnc
    except Exception as exc:
        raise RuntimeError(
            "python-sat is required for exact-cardinality encoding. "
            "Install with: pip install python-sat[pblib,aiger]"
        ) from exc

    card = CardEnc.equals(
        lits=[x_vars[i] for i in range(N + 1)],
        bound=m,
        top_id=counter[0],
    )
    clauses.extend(card.clauses)
    counter[0] = max(counter[0], card.nv)

    return CNFBundle(n_vars=counter[0], clauses=clauses, x_vars=x_vars)


def write_dimacs(bundle: CNFBundle, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"p cnf {bundle.n_vars} {len(bundle.clauses)}\n")
        for clause in bundle.clauses:
            f.write(" ".join(str(l) for l in clause) + " 0\n")


def solve_with_pysat(bundle: CNFBundle, solver_name: str = "glucose4") -> Tuple[bool, List[int]]:
    try:
        from pysat.solvers import Solver
    except Exception as exc:
        raise RuntimeError(
            "python-sat solver backend is unavailable. "
            "Install with: pip install python-sat[pblib,aiger]"
        ) from exc

    with Solver(name=solver_name, bootstrap_with=bundle.clauses) as s:
        sat = s.solve()
        if not sat:
            return False, []
        model = set(l for l in s.get_model() if l > 0)
        marks = [i for i, v in bundle.x_vars.items() if v in model]
        marks.sort()
        return True, marks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAT feasibility for complete sparse rulers (fixed N,m)"
    )
    parser.add_argument("--N", type=int, default=500)
    parser.add_argument("--m", type=int, default=39)
    parser.add_argument(
        "--mode",
        choices=["solve", "dimacs"],
        default="solve",
        help="solve directly with python-sat or only emit DIMACS",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="glucose4",
        help="python-sat solver name for --mode solve",
    )
    parser.add_argument(
        "--dimacs-out",
        type=Path,
        default=Path("results/sat_feasibility/sparse_ruler_N500_m39.cnf"),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("results/sat_feasibility/sparse_ruler_N500_m39_result.json"),
    )
    args = parser.parse_args()

    bundle = build_sparse_ruler_cnf(args.N, args.m)
    write_dimacs(bundle, args.dimacs_out)

    result = {
        "N": args.N,
        "m": args.m,
        "n_vars": bundle.n_vars,
        "n_clauses": len(bundle.clauses),
        "dimacs_path": str(args.dimacs_out),
        "mode": args.mode,
    }

    if args.mode == "solve":
        sat, marks = solve_with_pysat(bundle, solver_name=args.solver)
        result["sat"] = sat
        result["marks"] = marks
        result["message"] = "SAT model found" if sat else "UNSAT (no ruler with this N,m)"
    else:
        result["sat"] = None
        result["marks"] = []
        result["message"] = "DIMACS written; solve externally for certificate."

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
