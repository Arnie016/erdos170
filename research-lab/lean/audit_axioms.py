"""Replay theorem-specific audits under the project pin and enforce the allowed axiom set."""
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
MODULES = ("Certificates", "OneNewMark", "ExteriorRepair", "GraphCovers", "DeficitBounds")
ALLOWED = {"propext", "Classical.choice", "Quot.sound"}
REQUIRED = {
    "OpenMath.cliqueA_checked",
    "OpenMath.SparseRuler.too_many_missing_for_one_mark",
    "OpenMath.ExteriorRepair.one_block_capacity",
    "OpenMath.ExteriorRepair.cap_congruence_impossible",
    "OpenMath.GraphCovers.binary_clique_cover_domination",
    "OpenMath.SparseRuler.left_ap_distance_gt_of_stepB_dvd",
    "OpenMath.SparseRuler.right_ap_distance_gt_of_stepA_dvd",
    "OpenMath.SparseRuler.normalized_deficit_error_eventually_lt",
}
REPORT = re.compile(r"^'([^']+)' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)$")


def lean(*args):
    result = subprocess.run(["lake", "env", "lean", *args], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="", flush=True)
    result.check_returncode()
    return result.stdout


def main():
    version = lean("--version")
    if not re.search(r"Lean \(version 4\.19\.0,", version):
        raise ValueError("The research package requires Lean 4.19.0")
    lean("OpenMath/StatementAudit.lean")
    audited = set()
    for module in MODULES:
        source = Path("OpenMath") / f"{module}.lean"
        declared = re.findall(r"^#print axioms (\S+)$", (ROOT / source).read_text(), re.MULTILINE)
        matched = []
        for line in lean(str(source)).splitlines():
            record = REPORT.fullmatch(line)
            if record:
                name, listed = record.groups()
                axioms = {item.strip() for item in listed.split(",")} if listed else set()
                if axioms - ALLOWED:
                    raise ValueError(f"{name} depends on unapproved axioms {sorted(axioms - ALLOWED)}")
                if name in audited:
                    raise ValueError(f"Duplicate audit record for {name}")
                audited.add(name)
                matched.append(name)
        if not declared or len(matched) != len(declared):
            raise ValueError(f"Incomplete dependency report for {module}")
        for name in declared:
            if not any(qualified == name or qualified.endswith("." + name) for qualified in matched):
                raise ValueError(f"Missing report for {module}.{name}")
    if REQUIRED - audited:
        raise ValueError(f"Missing required declarations {sorted(REQUIRED - audited)}")
    print(f"PASS {len(audited)} theorem-specific axiom audits and the frozen mixed statement comparison")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        sys.exit(1)
