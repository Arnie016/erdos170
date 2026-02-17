from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .metrics import compute_W
from .wichmann_sporadic_data import SPORADIC_RECIPES

Recipe = Tuple[List[int], List[int]]
PatternFn = Callable[[int, int], Recipe]


@dataclass(frozen=True)
class ExtendedWichmannRuler:
    """Extended Wichmann construction result."""

    length: int
    marks: List[int]
    diffs: List[int]
    counts: List[int]
    source: str
    family: Optional[str] = None
    r: Optional[int] = None
    s: Optional[int] = None


def wichmann_upper_bound(length: int) -> int:
    if length < 1:
        raise ValueError("length must be >= 1")
    return math.isqrt(3 * length) + 3


def missing_distances(marks: Sequence[int], length: int) -> List[int]:
    W = compute_W(marks, length)
    return [d for d in range(1, length + 1) if W[d] == 0]


def is_complete(marks: Sequence[int], length: int) -> bool:
    return len(missing_distances(marks, length)) == 0


def _pattern_w1(r: int, s: int) -> Recipe:
    return ([1, 1 + r, 1 + 2 * r, 3 + 4 * r, 2 + 2 * r, 1], [r, 1, r, s, 1 + r, r])


def _pattern_w2(r: int, s: int) -> Recipe:
    return ([1, 1 + r, 1 + 2 * r, 3 + 4 * r, 2 + 2 * r, 1], [r, 1, 1 + r, s, r, r])


def _pattern_w3(r: int, s: int) -> Recipe:
    return ([1, 1 + r, 2 + 2 * r, 5 + 4 * r, 3 + 2 * r, 1], [1 + r, 1, r, s, 1 + r, 1 + r])


def _pattern_w6(r: int, s: int) -> Recipe:
    return (
        [1, 2 + r, 9 + 4 * r, 4 + 2 * r, 9 + 4 * r, 5 + 2 * r, 3 + r, 2 + r, 5 + 2 * r, 1],
        [2 + r, 1, 1, r, s, 1 + r, 1, 1, 1, 1 + r],
    )


def _pattern_w7(r: int, s: int) -> Recipe:
    return ([1, 1 + r, 2 + 2 * r, 5 + 4 * r, 3 + 2 * r, 1], [1 + r, 1, 1 + r, s, r, 1 + r])


def _pattern_w10(r: int, s: int) -> Recipe:
    return ([1, 1 + r, 2 + 2 * r, 5 + 4 * r, 3 + 2 * r, 1], [1 + r, 1, 2 + r, s, 1 + r, r])


def _pattern_w11(r: int, s: int) -> Recipe:
    return (
        [1, 1 + r, 1, 1 + 2 * r, 2 + 2 * r, 5 + 4 * r, 3 + 2 * r, 1],
        [1 + r, 1, 1, 1, r, s, r, r],
    )


def _pattern_w23(r: int, s: int) -> Recipe:
    return (
        [1, 2 + r, 3 + 2 * r, 7 + 4 * r, 4 + 2 * r, 1, 2 + r],
        [1 + r, 1, r, s, 1 + r, 2 + r, 1],
    )


def _pattern_w24(r: int, s: int) -> Recipe:
    return (
        [1, 1 + r, 1 + 2 * r, 3 + 4 * r, 2 + 2 * r, 1, 1 + r],
        [r, 1, r, s, 1 + r, r, 1],
    )


def _pattern_w35(r: int, s: int) -> Recipe:
    return (
        [1, 1 + r, 1 + 2 * r, 3 + 4 * r, 2 + 2 * r, 3 + 4 * r, 2 + 2 * r, 1],
        [r, 1, r, s, 1, s, 1 + r, r],
    )


def _pattern_w100(r: int, s: int) -> Recipe:
    return (
        [1, 3 + r, 1, 4 + r, 4 + 2 * r, 9 + 4 * r, 5 + 2 * r, 1, 5 + 2 * r, 1, 2],
        [1, 1, r, 1, 2 + r, s, 2 + r, 1, 1, r, 1],
    )


def _pattern_w101(r: int, s: int) -> Recipe:
    return (
        [1, 2 + r, 2 + 2 * r, 5 + 4 * r, 3 + 2 * r, 1, 2 + r],
        [1 + r, 1, r, s, 1 + r, 1 + r, 1],
    )


def _pattern_w103(r: int, s: int) -> Recipe:
    return ([1, 1 + r, 1 + 2 * r, 3 + 4 * r, 2 + 2 * r, 1], [r, 1, 3 + r, s, r, r])


def _pattern_w120(r: int, s: int) -> Recipe:
    return (
        [1, 1 + r, 1 + 2 * r, 3 + 4 * r, 4 + 6 * r, 2 + 2 * r, 1 + 2 * r, 1],
        [r, 1, 3 + r, s, 1, r, 1, r],
    )


def _pattern_w224(r: int, s: int) -> Recipe:
    return (
        [1, 1 + r, 1, 2 + 3 * r, 2 + 4 * r, 5 + 8 * r, 3 + 4 * r, 1, 3 + 4 * r, 1],
        [r, 2, r, 1, 1 + 2 * r, s, 1 + 2 * r, r, 1, r],
    )


def _pattern_w225(r: int, s: int) -> Recipe:
    return (
        [1, 3 + 2 * r, 1, 4 + 3 * r, 4 + 4 * r, 9 + 8 * r, 5 + 4 * r, 1, 3 + 3 * r, 2 + r, 1],
        [1 + r, 1, r, 1, 2 + 2 * r, s, 2 + 2 * r, 1 + r, 1, 1, r],
    )


def _pattern_w226(r: int, s: int) -> Recipe:
    return (
        [1, 1 + r, 4 + 3 * r, 1, 5 + 4 * r, 9 + 8 * r, 4 + 4 * r, 3 + 3 * r, 1, 3 + 2 * r, 1],
        [1 + r, 1, 1, r, 2 + 2 * r, s, 2 + 2 * r, 1, 1 + r, 1, r],
    )


def _pattern_w270(r: int, s: int) -> Recipe:
    return (
        [1, 1 + r, 2 + 5 * r, 1, 3 + 6 * r, 5 + 12 * r, 2 + 6 * r, 2 + 4 * r, 1, 2 + 3 * r, 1],
        [2 * r, 1, 1, r, 1 + 3 * r, s, 1 + 3 * r, 1, 2 * r, 1, r],
    )


def _pattern_w310(r: int, s: int) -> Recipe:
    return (
        [1, 3 + 8 * r, 1, 3 + 8 * r, 5 + 16 * r, 2 + 8 * r, 2 + 5 * r, 1, 1 + 2 * r, 1],
        [3 * r, 1, r, 1 + 4 * r, s, 1 + 4 * r, 1, 3 * r, 2, r],
    )


_FALLBACK_PATTERN_ORDER: Dict[int, Tuple[str, PatternFn]] = {
    1: ("W1", _pattern_w1),
    2: ("W2", _pattern_w2),
    3: ("W2", _pattern_w2),
    4: ("W3", _pattern_w3),
    5: ("W7", _pattern_w7),
    6: ("W10", _pattern_w10),
    7: ("W23", _pattern_w23),
    8: ("W24", _pattern_w24),
    9: ("W35", _pattern_w35),
    10: ("W100", _pattern_w100),
    11: ("W101", _pattern_w101),
    12: ("W103", _pattern_w103),
    13: ("W120", _pattern_w120),
    14: ("W224", _pattern_w224),
    15: ("W225", _pattern_w225),
    16: ("W226", _pattern_w226),
    17: ("W270", _pattern_w270),
    18: ("W310", _pattern_w310),
}


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _wichmann_value(column_index: int) -> int:
    residue = (column_index % 6) - 3
    return (column_index * column_index - residue * residue) // 3 + column_index


def _recipe_length(recipe: Recipe) -> int:
    diffs, counts = recipe
    return sum(d * c for d, c in zip(diffs, counts))


def _normalize_recipe(recipe: Recipe) -> Recipe:
    diffs, counts = recipe
    out_diffs: List[int] = []
    out_counts: List[int] = []
    for diff, count in zip(diffs, counts):
        if diff > 0 and count > 0:
            out_diffs.append(int(diff))
            out_counts.append(int(count))
    return out_diffs, out_counts


def _recipe_to_marks(recipe: Recipe) -> List[int]:
    diffs, counts = recipe
    marks = [0]
    for diff, count in zip(diffs, counts):
        for _ in range(count):
            marks.append(marks[-1] + diff)
    return marks


def _solve_pattern_with_linear_constraint(
    *,
    length: int,
    column: int,
    constant: int,
    r_coeff: int,
    s_coeff: int,
    target_is_column_plus_one: bool,
    length_expr: Callable[[int, int], int],
) -> Optional[Tuple[int, int]]:
    target = column + 1 if target_is_column_plus_one else column
    rhs_max = target - constant - s_coeff
    if rhs_max < r_coeff:
        return None
    r_upper = rhs_max // r_coeff
    for r in range(1, max(1, r_upper) + 1):
        rhs = target - constant - r_coeff * r
        if rhs <= 0 or rhs % s_coeff != 0:
            continue
        s = rhs // s_coeff
        if s <= 0:
            continue
        if length_expr(r, s) == length:
            return r, s
    return None


def generate_extended_wichmann(length: int, *, validate: bool = True) -> ExtendedWichmannRuler:
    """Construct a complete ruler using extended Wichmann recipes."""
    if length < 1:
        raise ValueError("length must be >= 1")

    if length in SPORADIC_RECIPES:
        diffs, counts = _normalize_recipe(SPORADIC_RECIPES[length])
        marks = _recipe_to_marks((diffs, counts))
        result = ExtendedWichmannRuler(
            length=length,
            marks=marks,
            diffs=diffs,
            counts=counts,
            source="sporadic",
        )
        if validate:
            _validate_result(result)
        return result

    column = _round_half_up(math.sqrt(3 * length + 2.25))
    values = [_wichmann_value(n) for n in range(column - 2, column + 2)]
    diffs = [values[i + 1] - values[i] for i in range(3)]
    if values[1] >= length:
        height = values[1] - length
        column_height = diffs[0]
    else:
        height = values[2] - length
        column += 1
        column_height = diffs[1]

    excess_fraction = 1.0 - (height + 1) / (column_height + 1)
    column_mod = ((column - 2) % 6) + 1

    rule_upper = [
        ((column - 2) // 6, (column - 8) // 3),
        ((column - 3) // 6, (column - 6) // 3),
        ((column - 4) // 6, (column - 4) // 3),
        ((column - 5) // 6, (column - 2) // 3),
        ((column - 6) // 6, column // 3),
        ((column - 7) // 6, (column + 2) // 3),
    ]
    r_upper, s_upper = rule_upper[column_mod - 1]
    bot_prev = _pattern_w1(r_upper, s_upper)
    bot_this = _pattern_w1(r_upper, s_upper + 1)
    bot_this_m1 = _pattern_w2(r_upper, s_upper + 1)

    rule_lower = [
        (_pattern_w6, (column - 14) // 6, (column_height - 3) // 2),
        (_pattern_w6, (column - 15) // 6, (column_height - 1) // 2),
        (_pattern_w35, (column - 4) // 6, (column - 4) // 6),
        (_pattern_w6, (column - 11) // 6, (column_height - 5) // 2),
        (_pattern_w6, (column - 12) // 6, (column_height - 3) // 2),
        (_pattern_w6, (column - 13) // 6, (column_height - 1) // 2),
    ]
    mid_pattern, r_lower, s_lower = rule_lower[column_mod - 1]
    mid_this = mid_pattern(r_lower, s_lower)

    column_places: List[Recipe] = [bot_prev, mid_this, bot_this_m1, bot_this]
    if column_mod == 6 and (math.floor(column_height / 2) - height) in {0, 1}:
        case11 = _pattern_w11(column_height - 1, 2 * column_height)
        case11p = (case11[0][:], case11[1][:])
        case11p[1][-1] += 1
        column_places.extend([case11, case11p])

    column_vals = [_recipe_length(recipe) for recipe in column_places]
    covered = column_vals[1] + mid_this[1][0] + 1

    chosen: Optional[Recipe] = None
    source = "unknown"
    family: Optional[str] = None
    r_value: Optional[int] = None
    s_value: Optional[int] = None

    if length in column_vals:
        chosen = column_places[column_vals.index(length)]
        source = "column_place"
    elif excess_fraction < 0.25:
        chosen = (bot_prev[0] + [length - column_vals[0]], bot_prev[1] + [1])
        source = "bot_prev_append_1"
    elif excess_fraction <= 0.5:
        extender = bot_prev[1][0] + 1
        chosen = (
            bot_prev[0] + [extender, length - column_vals[0] - extender],
            bot_prev[1] + [1, 1],
        )
        source = "bot_prev_append_2"
    elif column_vals[1] < length <= covered:
        chosen = (mid_this[0] + [length - column_vals[1]], mid_this[1] + [1])
        source = "mid_append_1"
    elif excess_fraction < 1.0:
        max_val = column_vals[1] if length < 100000 else max(column_vals[1], length - 10 * int(length ** 0.25))
        solutions: List[Tuple[int, int]] = []
        for r in range(1, column):
            s = column - 3 - 4 * r
            if s <= 0:
                continue
            value = 3 + 8 * r + 4 * r * r + 3 * s + 4 * r * s
            if max_val < value <= length:
                solutions.append((r, s))

        if solutions:
            candidates = [_pattern_w1(r, s) for r, s in solutions] + [_pattern_w2(r, s) for r, s in solutions]
            candidates.sort(key=lambda recipe: (_recipe_length(recipe), recipe[1][0]))
            candidate_lengths = [_recipe_length(recipe) for recipe in candidates]
            if length in candidate_lengths:
                idx = candidate_lengths.index(length)
                chosen = candidates[idx]
                source = "w1_w2_exact"
            else:
                last = candidates[-1]
                if length - candidate_lengths[-1] <= last[1][0] + 1:
                    chosen = (last[0] + [length - candidate_lengths[-1]], last[1] + [1])
                    source = "w1_w2_append_1"

    if chosen is None:
        fallback_equations = [
            (1, 3, 4, 1, True, lambda r, s: 3 + 8 * r + 4 * r * r + 3 * s + 4 * r * s),
            (2, 3, 4, 1, True, lambda r, s: 2 + 8 * r + 4 * r * r + 3 * s + 4 * r * s),
            (3, 3, 4, 1, False, lambda r, s: 2 + 8 * r + 4 * r * r + 3 * s + 4 * r * s),
            (4, 5, 4, 1, False, lambda r, s: 6 + 10 * r + 4 * r * r + 5 * s + 4 * r * s),
            (5, 5, 4, 1, False, lambda r, s: 5 + 10 * r + 4 * r * r + 5 * s + 4 * r * s),
            (6, 6, 4, 1, True, lambda r, s: 9 + 14 * r + 4 * r * r + 5 * s + 4 * r * s),
            (7, 7, 4, 1, True, lambda r, s: 11 + 13 * r + 4 * r * r + 7 * s + 4 * r * s),
            (8, 4, 4, 1, True, lambda r, s: 4 + 9 * r + 4 * r * r + 3 * s + 4 * r * s),
            (9, 4, 4, 2, True, lambda r, s: 5 + 10 * r + 4 * r * r + 6 * s + 8 * r * s),
            (10, 11, 4, 1, True, lambda r, s: 34 + 23 * r + 4 * r * r + 9 * s + 4 * r * s),
            (11, 6, 4, 1, True, lambda r, s: 9 + 11 * r + 4 * r * r + 5 * s + 4 * r * s),
            (12, 5, 4, 1, True, lambda r, s: 4 + 12 * r + 4 * r * r + 3 * s + 4 * r * s),
            (13, 7, 4, 1, True, lambda r, s: 9 + 20 * r + 4 * r * r + 3 * s + 4 * r * s),
            (14, 7, 8, 1, True, lambda r, s: 12 + 31 * r + 16 * r * r + 5 * s + 8 * r * s),
            (15, 11, 8, 1, True, lambda r, s: 32 + 47 * r + 16 * r * r + 9 * s + 8 * r * s),
            (16, 11, 8, 1, True, lambda r, s: 31 + 47 * r + 16 * r * r + 9 * s + 8 * r * s),
            (17, 7, 12, 1, True, lambda r, s: 12 + 46 * r + 36 * r * r + 5 * s + 12 * r * s),
            (18, 7, 16, 1, True, lambda r, s: 12 + 61 * r + 64 * r * r + 5 * s + 16 * r * s),
        ]
        for pattern_idx, constant, r_coeff, s_coeff, plus_one, length_expr in fallback_equations:
            solution = _solve_pattern_with_linear_constraint(
                length=length,
                column=column,
                constant=constant,
                r_coeff=r_coeff,
                s_coeff=s_coeff,
                target_is_column_plus_one=plus_one,
                length_expr=length_expr,
            )
            if solution is None:
                continue
            r_value, s_value = solution
            family, pattern_fn = _FALLBACK_PATTERN_ORDER[pattern_idx]
            chosen = pattern_fn(r_value, s_value)
            source = "fallback_equation"
            break

    if chosen is None:
        raise ValueError(f"no extended Wichmann construction found for length={length}")

    chosen = _normalize_recipe(chosen)
    marks = _recipe_to_marks(chosen)
    result = ExtendedWichmannRuler(
        length=length,
        marks=marks,
        diffs=chosen[0],
        counts=chosen[1],
        source=source,
        family=family,
        r=r_value,
        s=s_value,
    )
    if validate:
        _validate_result(result)
    return result


def _validate_result(result: ExtendedWichmannRuler) -> None:
    if result.marks[0] != 0:
        raise ValueError("constructed ruler does not start at 0")
    if result.marks[-1] != result.length:
        raise ValueError("constructed ruler length mismatch")
    if any(a >= b for a, b in zip(result.marks, result.marks[1:])):
        raise ValueError("constructed marks are not strictly increasing")
    missing = missing_distances(result.marks, result.length)
    if missing:
        raise ValueError(f"construction is incomplete, missing distances count={len(missing)}")
