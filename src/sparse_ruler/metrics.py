from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


def compute_W(A: Sequence[int], N: int) -> List[int]:
    """Compute unordered weight counts W(d) for d=1..N.

    W is returned as a list of length N+1 with index 0 unused.
    """
    W = [0] * (N + 1)
    m = len(A)
    for i in range(m - 1):
        ai = A[i]
        for j in range(i + 1, m):
            d = A[j] - ai
            if d <= 0 or d > N:
                raise ValueError("A must be sorted with endpoints 0 and N")
            W[d] += 1
    return W


def energy_floor(S: int, N: int) -> int:
    q = S // N
    r0 = S - q * N
    return (N - r0) * q * q + r0 * (q + 1) * (q + 1)


def energy_ratio(E: int, Emin: int) -> float:
    if Emin == 0:
        return 0.0
    return E / Emin


@dataclass(frozen=True)
class MetricSummary:
    N: int
    m: int
    S: int
    missing_count: int
    peak: int
    H3: int
    H4: int
    E: int
    Emin: int
    rho: float
    count_w1: int
    count_w2: int
    count_w3: int
    count_w4: int
    count_w5: int
    count_w6plus: int

    def to_csv_row(self) -> str:
        return (
            f"{self.N},{self.m},{self.S},{self.missing_count},{self.peak},{self.H3},{self.H4},"
            f"{self.E},{self.Emin},{self.rho:.6f},{self.count_w1},{self.count_w2},"
            f"{self.count_w3},{self.count_w4},{self.count_w5},{self.count_w6plus}"
        )


@dataclass(frozen=True)
class Solution:
    A: List[int]
    metrics: MetricSummary


@dataclass(frozen=True)
class ImpossibleResult:
    N: int
    m: int
    S: int

    def to_csv_row(self) -> str:
        Emin = energy_floor(self.S, self.N)
        return (
            f"{self.N},{self.m},{self.S},{self.N},0,0,0,0,{Emin},0.000000,"
            "0,0,0,0,0,0"
        )


def summarize_metrics(N: int, A: Sequence[int], W: Sequence[int]) -> MetricSummary:
    m = len(A)
    S = m * (m - 1) // 2
    if len(W) != N + 1:
        raise ValueError("W must be length N+1")
    total_pairs = sum(W[1:])
    if total_pairs != S:
        raise ValueError("sum W must equal S")
    missing_count = sum(1 for d in range(1, N + 1) if W[d] == 0)
    peak = max(W[1:]) if N > 0 else 0
    H3 = sum(1 for d in range(1, N + 1) if W[d] >= 3)
    H4 = sum(1 for d in range(1, N + 1) if W[d] >= 4)
    E = sum(w * w for w in W[1:])
    Emin = energy_floor(S, N)
    rho = energy_ratio(E, Emin)
    count_w1 = sum(1 for d in range(1, N + 1) if W[d] == 1)
    count_w2 = sum(1 for d in range(1, N + 1) if W[d] == 2)
    count_w3 = sum(1 for d in range(1, N + 1) if W[d] == 3)
    count_w4 = sum(1 for d in range(1, N + 1) if W[d] == 4)
    count_w5 = sum(1 for d in range(1, N + 1) if W[d] == 5)
    count_w6plus = sum(1 for d in range(1, N + 1) if W[d] >= 6)
    return MetricSummary(
        N=N,
        m=m,
        S=S,
        missing_count=missing_count,
        peak=peak,
        H3=H3,
        H4=H4,
        E=E,
        Emin=Emin,
        rho=rho,
        count_w1=count_w1,
        count_w2=count_w2,
        count_w3=count_w3,
        count_w4=count_w4,
        count_w5=count_w5,
        count_w6plus=count_w6plus,
    )


def format_csv_header() -> str:
    return (
        "N,m,S,missing_count,peak,H3,H4,E,Emin,rho,count_w1,count_w2,"
        "count_w3,count_w4,count_w5,count_w6plus"
    )


def validate_complete(W: Sequence[int], N: int) -> bool:
    return all(W[d] >= 1 for d in range(1, N + 1))


def sample_distance_check(A: Sequence[int], N: int, samples: Iterable[int]) -> bool:
    diffs = set()
    m = len(A)
    for i in range(m - 1):
        for j in range(i + 1, m):
            diffs.add(A[j] - A[i])
    return all(d in diffs for d in samples if 1 <= d <= N)
