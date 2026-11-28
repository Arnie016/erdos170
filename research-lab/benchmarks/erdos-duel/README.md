# Erdős proof duel

Open [the replay board](index.html) in a browser from a local checkout. It opens at the latest recorded stage. Select **Start replay** to return to the frozen baseline, then step through submissions, challenges, responses, and decisions. The timeline jumps to any stage. Each decision links to the evidence behind its points.

The replay uses only the rules in the [frozen brief](PROJECT_BRIEF.md) and the recorded referee decisions in [match.json](match.json). The mathematical scope of each result is explained in the relevant decision.

| Desk | Target | Extension | Lean | Independent check | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Group · Erdős #117 | 8 | 2 | 2 | 1 | **13** |
| Ruler · Erdős #170 | 8 | 2 | 1 | 1 | **12** |

Group wins the match. Its weighted dominating-support formula gives two exact optima across every finite simple graph and finite field, and Lean checks the full binary case. Ruler closes every two-delete, one-insert edit in the canonical family at its original length by combining the general argument and an exact finite boundary. Lean checks the residue argument used by the Ruler proof. [Group proof](../../problems/erdos-117/runs/2026-11-28-graph-covers/PROOF.md) · [Ruler proof](../../problems/erdos-170/runs/2026-11-28-exterior-repair/PROOF.md)

## After the final score

Compute exposed a sharper question for the ruler. Across 1,657,494,000 legal edits the observed optimum missed exactly the radius plus five distances for every radius 2 through 32. A two-junction construction attains that count for all larger radii too. A separate lower bound puts the best achievable deficit within 13 of the construction, giving a limit of one for the minimum deficit divided by the radius. The frozen match score predates this follow-up. [Quantitative theorem](../../problems/erdos-170/runs/2026-11-28-exterior-repair/DEFICIT-BOUNDS.md) · [Full search and held-out predictions](../../problems/erdos-170/runs/2026-11-28-exterior-repair/gpu/README.md)

## Replay and verify

From this directory, use Python 3.10 or newer. The recorded checks used Python 3.14.8.

```sh
python3 -B scoreboard.py --check-evidence --check-html
python3 -B scoreboard.py --stage baseline
python3 -B scoreboard.py --stage group-decision --json
python3 -B -m unittest -v
```

`scoreboard.py` recomputes every stage from the event log. It awards each desk a single base result and one award per bonus. An explicit revision replaces the previous decision. Equal totals produce a tie. `--check-evidence` checks local receipt files and permits version-pinned paper or commit links. `--check-html` detects an out-of-date embedded replay.

## Independent referee checks

```sh
LC_ALL=C python3 -B referee/check_ruler.py --max-r 16 --any-pair
LC_ALL=C python3 -B referee/check_odd_cover.py
```

The ruler check encodes each mark as one integer bit. Shifting the retained mark bits by each retained mark yields their positive differences. A reflection of the retained bits supplies the differences on the other side of each fresh insertion. The checker enumerates every original-mark deletion pair and every fresh insertion in the fixed interval for \(1\le r\le16\). It tests a known repair, a broken repair, and a missing endpoint. [Ruler replay](referee/ruler.log)

For the graph with edges 01 and 12 and isolated vertex 3, the group check computes signed determinants over \(\mathbb F_3\), searches for a clique, and constructs isotropic spans independently of the dominating-support formula. It validates the matching ten-element clique and ten-space cover against all 81 vectors. The checker rejects both a broken clique and a broken cover. [Odd-field replay](referee/odd_cover.log)

## Record another match

1. For a separate match, copy the board, reducer, and test into a sibling benchmark directory. Freeze two targets, extensions, and exact prior evidence in its `PROJECT_BRIEF.md` before either desk starts. Generate a fresh book with the opening commit and fill its two desk descriptions.
2. Add each desk's opening `submission`, using its desk ID as `actor`. Record a referee `challenge` with `desk` and `reply_to` identifying the submission. The desk's next submission names the challenge in `reply_to`.
3. Append a referee `decision` naming the submission it judges. Use the frozen base level, a Boolean full-extension award, the Lean level, and a Boolean independent-check award. Every awarded category needs a typed evidence receipt with a link, domain, and method. Explain the reasoning and any limit in the decision summary. Record a revised award with `supersedes` and the previous decision ID.
4. Run the validation and tests, then update that match's board with `python3 -B scoreboard.py --check-evidence --write-html`. Store the book, board, receipts, and baseline together.

From the new benchmark directory, set `opening_commit` to its frozen brief commit and generate the unscored book.

```sh
python3 -B scoreboard.py --template "$opening_commit" > match.json
python3 -B scoreboard.py --stage baseline
```

## Board review

The reader is someone evaluating a mathematical result who wants the shortest path from score to proof or replay. The board uses the same light paper treatment throughout, with typography for emphasis and distinct desk colors. It uses native buttons, links, and disclosure controls. The HTML contains its data and code and has no remote runtime dependencies.

The local Chrome smoke test exercised the standalone file at 1360, 375, and 320 pixels wide, including keyboard stage selection, the reset and next actions, the final receipt links, and the unreadable-data fallback. Human comprehension review is `NOT_TESTED`.
