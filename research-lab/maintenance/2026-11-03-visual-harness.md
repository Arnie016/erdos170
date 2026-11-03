# Visual notebook and selective verification | 3 November 2026

**INFRASTRUCTURE. No new mathematical claim.**

This release follows the project owner's request for a wider mathematical portfolio and tasteful, intuitive GitHub documentation. It creates an exploration board, four original editable SVGs, a reusable design guide and a selective finite-certificate checker. It preserves the original sparse-ruler README guide and historical CI jobs. The existing catalog is unchanged: unstarted candidate topics and this engineering port are not promoted into completed mathematical results.

## Baseline and scope

Inspected main: `52ad6442ba03695d9e2c00efaed0e4d96e62c915`. The original workflow blob was verified byte-for-byte against Git blob `53a2a61a58941ae8f2389a42857d028bdbf85a15` before adding the new job. The public #5 branch was inspected at `4ce3f329cf604b3b5124aaaeab236085db04e820`; it was reported nonmergeable and is not edited by this distinct release. No other repository or automation schedule is changed.

New code uses a fixed verifier allowlist and data-only JSON. It checks the existing field witness, writes scoped execution receipts and refuses unknown verifier types, forged formalization labels or paths outside the lab. It does not implement general proof search, automatic theorem discovery or outreach.

## Actual local validation

Two sequential validation processes, each with a 35-second hard timeout and kill/wait cleanup:

| Process | Domain | Exit | External wall time |
|---|---|---|---|
| First | One registered certificate plus 25 frozen harness tests | 0 | 0.975 s |
| Second | Independent repeated-multiplication scan of all 1129 residues; SVG safety/render checks; original guide/workflow preservation | 0 | 1.467 s |

Both passed. The separate field scan returned exactly roots 45 and 276. Negative controls test verifier sensitivity, not mathematical novelty or universal security. The complete historical archive and Lean project were not rerun locally.

Four SVGs were rendered at original width and 420 pixels. Visual inspection found missing arrow/product glyphs in the local fonts; these were replaced with plain text, then re-rendered. Arithmetic code and acceptance criteria were not changed. Diagrams have descriptive titles/alt text and text equivalents. No scripts, external font files or third-party artwork are embedded.

## Source and evidence depth

The original field-witness proof was recovered from the 3 November local research package and read in full. Its theorem is unchanged. Primary starting sources for the expanded candidates are recorded with reading-depth caveats in [EXPLORE.md](../EXPLORE.md). This is not a current-status review of every field.

The new `focused-certificates` CI job executes this checker and its regression suite. Remote CI must be observed for the resulting commit before claiming a remote pass; this note records local validation only. Local source hashes and receipts accompany the release package. Workflow checks are not peer review or Lean formalization.

No email, private correspondence, draft, contact ledger or unpublished third-party work is included. No mathematical outreach is performed by this code. No generated research result was added to the catalog.
