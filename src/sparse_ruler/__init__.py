"""Sparse ruler rigidity experiment suite."""

from .e_hunt import (
    generate_deletion_candidates,
    load_e_hunt_config,
    run_e_hunt,
    run_e_hunt_from_config_path,
)
from .excess import (
    base_term,
    excess_from_m,
    run_excess_baseline,
)
from .metrics import (
    compute_W,
    energy_floor,
    energy_ratio,
    summarize_metrics,
)
from .search import anneal_run, run_experiments
from .research_report_cycle2 import generate_research_report_cycle2
from .wichmann import (
    ExtendedWichmannRuler,
    generate_extended_wichmann,
    is_complete,
    missing_distances,
    wichmann_upper_bound,
)

__all__ = [
    "compute_W",
    "energy_floor",
    "energy_ratio",
    "summarize_metrics",
    "anneal_run",
    "run_experiments",
    "generate_research_report_cycle2",
    "generate_deletion_candidates",
    "load_e_hunt_config",
    "run_e_hunt",
    "run_e_hunt_from_config_path",
    "base_term",
    "excess_from_m",
    "run_excess_baseline",
    "ExtendedWichmannRuler",
    "generate_extended_wichmann",
    "is_complete",
    "missing_distances",
    "wichmann_upper_bound",
]
