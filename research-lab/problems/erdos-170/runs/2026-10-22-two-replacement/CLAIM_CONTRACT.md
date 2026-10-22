# Claim contract frozen before both workers (2026-10-22)

For the exact A in candidate.json and all B differing by at most two removed and two added marks, decide whether B has every difference 1,...,500. The mandatory 0 and 500 endpoints cannot be removed. The missing baseline 493 demands one new mark in S={7,493,...,499}. Exactly two replacements reduce to choose(37,2)*(8*454+choose(8,2))=2,437,560 candidates. Zero and one replacements are separately checked.

Competing explanations: a coordinated pair fixes the missing tail distance without sacrificing other essential differences, or local rigidity persists through radius two. Falsifier: one fully checked complete B. Acceptance: two independent deterministic exact integer checkers agree on complete-domain size and no-candidate count, with <=35-second hard limits per worker. Stop on a verified complete witness or completed exclusion. No claim about general F(500) or Erdős #170.
