# Sparse Ruler Rigidity Experiment Suite

This project runs simulated annealing experiments for Erdős Problem #170 (sparse rulers / restricted difference bases). It builds candidate mark sets `A ⊆ {0..N}`, tracks distance multiplicities `W(d)`, and searches for complete rulers with near-optimal energy.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Run a single experiment

```bash
sparse-ruler run 29 9 --runs 50 --steps 200000 --processes 4 --out-dir outputs
```

Outputs:
- `outputs/results_N29_m9.csv` with one CSV row of metrics.
- `outputs/solutions/solution_N29_m9_*.json` with the best 5 solutions.

### Run the default scan

Small scan (N=1..120 with m0, m0+1, m0+2):

```bash
sparse-ruler small-scan --runs 200 --steps 200000 --processes 8 --out-dir outputs
```

Large scan (N in {1000,2000,5000,10000,20000,50000}):

```bash
sparse-ruler large-scan --runs 200 --steps 200000 --processes 8 --out-dir outputs
```

### Compute metrics for a specific ruler

```bash
sparse-ruler metrics 13 0,1,6,9,11,13
```

## Modal execution

If you are running on Modal, the CLI works the same way. Mount the repo, then run:

```bash
python -m sparse_ruler.cli small-scan --runs 200 --steps 200000 --processes 8 --out-dir /mnt/outputs
```

## Notes

- The code keeps `W(d)` updated incrementally when marks move.
- The annealer uses `H = alpha*M + beta*E + gamma*peak` with `alpha=1_000_000`, `beta=1`, `gamma=1000`.
- If `C(m,2) < N`, the run is declared impossible and a CSV row is still written.

## Tests

```bash
pytest
```

## Paper Automation (GitHub + Overleaf)

Generate the latest paper bundle locally:

```bash
cd /Users/hema/Desktop/erdos170
export PYTHONPATH=src
python3 scripts/generate_research_report_to_date.py
python3 scripts/build_paper_artifacts.py --repo-root .
```

This writes:

- `paper/main.tex`
- `paper/generated/tables.tex`
- `paper/generated/status_snapshot.json`
- copies of generated markdown/json reports into `paper/generated/`

CI workflow: `.github/workflows/paper-automation.yml`

- Auto-runs on `main` pushes, nightly, and manual dispatch.
- Rebuilds paper artifacts and uploads `erdos170-paper-bundle` artifact.
- Optional Overleaf sync if repository secrets are set:
  - `OVERLEAF_GIT_URL` (required for sync; use Overleaf project Git URL, no token in URL)
  - `OVERLEAF_GIT_TOKEN` (required for sync; generated in Overleaf account Git settings)
  - `OVERLEAF_GIT_BRANCH` (optional, default `master`)

## Cycle-3 Tail-Focused Run

```bash
cd /Users/hema/Desktop/erdos170
source .venv/bin/activate
export PYTHONPATH=src
python3 scripts/run_e_hunt_breakthrough_cycle3.py \
  --config configs/e_hunt_breakthrough_cycle3_tailfocus.json
```

Outputs land under:

- `results/e_hunt_breakthrough_cycle3_tailfocus/`

## SAT Feasibility Lane (N=500, m=39)

Build DIMACS + attempt direct solve (requires `python-sat`):

```bash
cd /Users/hema/Desktop/erdos170
source .venv/bin/activate
pip install "python-sat[pblib,aiger]"
export PYTHONPATH=src
python3 scripts/run_sat_sparse_ruler_feasibility.py \
  --N 500 --m 39 \
  --mode solve \
  --dimacs-out results/sat_feasibility/sparse_ruler_N500_m39.cnf \
  --json-out results/sat_feasibility/sparse_ruler_N500_m39_result.json
```

DIMACS-only mode (external solver):

```bash
python3 scripts/run_sat_sparse_ruler_feasibility.py \
  --N 500 --m 39 --mode dimacs
```
