# Paper Automation

This folder is Overleaf-ready.

- `main.tex`: entrypoint.
- `generated/`: auto-created artifacts from experiment outputs.

Local refresh:

```bash
cd /Users/hema/Desktop/erdos170
export PYTHONPATH=src
python3 scripts/generate_research_report_to_date.py
python3 scripts/build_paper_artifacts.py --repo-root .
```
