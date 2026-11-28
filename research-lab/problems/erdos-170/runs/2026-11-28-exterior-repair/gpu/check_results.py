"""Audit domains, witnesses, and the precommitted prediction in the recorded table."""
import csv
from math import comb
from pathlib import Path

from check_template import marks_from_gaps, missing


def main():
    path = Path(__file__).with_name("results.csv")
    with path.open(newline="") as f:
        rows = [{key: int(value) for key, value in row.items()} for row in csv.DictReader(f)]
    if [row["r"] for row in rows] != list(range(1, 33)):
        raise ValueError("Expected one ordered row for every radius through 32")
    for row in rows:
        r = row["r"]
        marks = marks_from_gaps(r)
        n = marks[-1]
        pairs = comb(len(marks), 2)
        if (row["N"], row["marks"], row["deletion_pairs"], row["fresh_triples"]) != (
            n, len(marks), pairs, pairs * (n + 1 - len(marks))
        ):
            raise ValueError(f"Domain count disagrees at radius {r}")
        x, y, z = (row[key] for key in ("delete_x", "delete_y", "insert_z"))
        if not (x < y and x in marks and y in marks and 0 <= z <= n and z not in marks):
            raise ValueError(f"Invalid witness at radius {r}")
        edit = [p for p in marks if p not in (x, y)] + [z]
        expected_minimum = 4 if r == 1 else r + 5
        if len(missing(edit, n)) != row["min_missing"] or row["min_missing"] != expected_minimum:
            raise ValueError(f"Wrong witness or minimum at radius {r}")
        a, b, q = 2*r + 1, 2*r + 2, 4*r + 3
        left = (r + 1) * a
        if r >= 3 and (x, y, z) != (left, left + 2*r*q, left + b):
            raise ValueError(f"Wrong first minimizer at radius {r}")
    print("PASS exact domains, literal witness coverage and eight held-out predictions")
    print("Legal edit triples:", sum(row["fresh_triples"] for row in rows))


if __name__ == "__main__":
    main()
