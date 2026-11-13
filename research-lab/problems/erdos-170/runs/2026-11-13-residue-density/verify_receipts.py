#!/usr/bin/env python3
"""Deterministic acceptance check for two frozen outputs. Does not execute workers."""
import json
from pathlib import Path

x=json.loads(Path("worker_formula.ci.json").read_text())
y=json.loads(Path("worker_independent.ci.json").read_text())
expected=[10,10,12,12,10,12,14,14,12,10,11,12,13,14]
assert x["status"]==y["status"]=="PASS"
assert x["negative_control"]==y["negative_control"]=="PASS"
assert [row["M"] for row in x["rows"]]==expected
assert [(row["a"],row["M"]) for row in x["rows"]]==[(row["a"],row["M"]) for row in y["rows"]]
assert y["independent_exhaustive_domain"]=={"r":3,"t":12,"a_min":1,"a_max":5,"subsets_per_a":4096}
assert [(p["a"],p["minimum"]) for p in y["small"]]==[(1,4),(2,4),(3,6),(4,4),(5,5)]
assert all(p["number_of_minimal_R"]>0 for p in y["small"])
print("PASS: independently reconstructed minima, full declared domains and controls")
