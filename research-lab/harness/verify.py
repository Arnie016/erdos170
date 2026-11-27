#!/usr/bin/env python3
"""Data-only, selective certificate verification. Standard library; no network.

This is a verifier of finite arithmetic obligations, not a proof-search engine,
formal theorem prover, permission system or external-action agent.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

LAB = Path(__file__).resolve().parents[1]
REGISTRY = LAB / 'harness/registry.json'
MAX_FIELD = 65535


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def integer(value: object, name: str, lo: int, hi: int) -> int:
    require(type(value) is int and lo <= value <= hi,
            f'{name}: integer outside [{lo}, {hi}]')
    return value


def prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


def factors(n: int) -> list[int]:
    answer = []
    for d in range(2, math.isqrt(n) + 1):
        if n % d == 0:
            answer.append(d)
            while n % d == 0:
                n //= d
    if n > 1:
        answer.append(n)
    return answer


def safe_file(relative: str, lab: Path = LAB) -> Path:
    require(isinstance(relative, str) and bool(relative), 'missing path')
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, 'unsafe path')
    resolved = (lab / p).resolve()
    require(resolved.is_relative_to(lab.resolve()), 'path escapes lab')
    require(resolved.is_file(), 'evidence file missing')
    require(resolved.stat().st_size <= 100_000, 'evidence file too large')
    return resolved


def read_json(path: Path) -> object:
    require(path.stat().st_size <= 100_000, 'JSON file too large')
    return json.loads(path.read_text(encoding='utf-8'))


def validate_registry(data: object, lab: Path = LAB) -> list[dict]:
    require(type(data) is dict and set(data) == {'schema_version', 'certificates'},
            'invalid registry fields')
    integer(data['schema_version'], 'schema_version', 1, 1)
    entries = data['certificates']
    require(type(entries) is list and 1 <= len(entries) <= 32, 'invalid registry length')
    seen = set()
    fields = {'id', 'verifier', 'certificate', 'proof', 'scope', 'literature', 'formalization'}
    for entry in entries:
        require(type(entry) is dict and set(entry) == fields, 'invalid entry fields')
        key = entry['id']
        require(isinstance(key, str) and 1 <= len(key) <= 80 and
                all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in key), 'invalid id')
        require(key not in seen, 'duplicate certificate id')
        seen.add(key)
        require(entry['verifier'] == 'odd_character_root_v1', 'unknown verifier')
        require(entry['literature'] in {'KNOWN', 'PRIORITY_UNCHECKED', 'CANDIDATE_NOVELTY'},
                'invalid literature label')
        require(entry['formalization'] == 'UNFORMALIZED',
                'this runner does not establish formal verification')
        require(isinstance(entry['scope'], str) and bool(entry['scope'].strip()), 'missing scope')
        require(isinstance(entry['certificate'], str) and entry['certificate'].endswith('.json'),
                'certificate must be JSON')
        require(isinstance(entry['proof'], str) and entry['proof'].endswith('.md'),
                'proof must be Markdown')
        safe_file(entry['certificate'], lab)
        safe_file(entry['proof'], lab)
    return entries


def check_character(data: object) -> dict:
    fields = {'schema_version', 'p', 'r', 'order', 'exponents', 'base', 'offset', 'n0', 'period'}
    require(type(data) is dict and set(data) == fields, 'invalid certificate fields')
    integer(data['schema_version'], 'schema_version', 1, 1)
    p = integer(data['p'], 'p', 3, MAX_FIELD)
    r = integer(data['r'], 'r', 2, p - 1)
    order = integer(data['order'], 'order', 3, p - 1)
    base = integer(data['base'], 'base', 2, 2)
    offset = integer(data['offset'], 'offset', 1, 64)
    n0 = integer(data['n0'], 'n0', 1, 256)
    period = integer(data['period'], 'period', 1, MAX_FIELD)
    exponents = data['exponents']
    require(type(exponents) is list and len(exponents) == offset, 'hole-count mismatch')
    for e in exponents:
        integer(e, 'exponent', 0, 4096)
    require(len(set(exponents)) == len(exponents), 'duplicate exponents')
    require(max(exponents) < base ** n0 + offset, 'holes not distinct at initial modulus')
    require(prime(p), 'field modulus is not prime')
    require(order % 2 == 1, 'order must be odd')
    require(pow(r, order, p) == 1 and
            all(pow(r, order // q, p) != 1 for q in factors(order)), 'wrong exact order')
    # Two separately structured evaluations, not independent human review.
    direct = sum(pow(r, e, p) for e in exponents) % p
    current = 1
    iterative = 0
    exponent_set = set(exponents)
    for i in range(max(exponents) + 1):
        if i in exponent_set:
            iterative = (iterative + current) % p
        current = current * r % p
    require(direct == iterative == 0, 'hole polynomial does not vanish')
    require(all((1 + pow(r, a, p)) % p for a in range(order)),
            'a subset-sum factor vanished')
    require((pow(base, n0, order) + offset) % order == 0, 'initial progression fails')
    require(pow(base, period, order) == 1, 'progression period fails')
    return {'prime_field': p, 'root': r, 'exact_order': order,
            'polynomial_residue_direct': direct, 'polynomial_residue_iterative': iterative,
            'nonzero_factors_checked': order, 'n0': n0, 'period': period,
            'finite_scope': 'listed arithmetic obligations; universal implication is in the ordinary proof'}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(selected: str | None, registry: Path = REGISTRY, lab: Path = LAB) -> dict:
    start = time.monotonic()
    result = {'status': 'FAIL', 'checker': 'odd_character_root_v1',
              'checker_sha256': sha256(Path(__file__)), 'python': sys.version.split()[0],
              'github_sha': os.environ.get('GITHUB_SHA'), 'selected': selected,
              'records': [], 'formalization': 'UNFORMALIZED'}
    try:
        entries = validate_registry(read_json(registry), lab)
        result['registry_sha256'] = sha256(registry)
        if selected is not None:
            entries = [e for e in entries if e['id'] == selected]
            require(bool(entries), 'unknown selected certificate')
        for e in entries:
            path = safe_file(e['certificate'], lab)
            proof = safe_file(e['proof'], lab)
            checked = check_character(read_json(path))
            result['records'].append({'id': e['id'], 'scope': e['scope'], 'status': 'PASS',
                                      'input_sha256': sha256(path), 'proof_sha256': sha256(proof),
                                      'literature': e['literature'], **checked})
        result['status'] = 'PASS'
    except (ValueError, OSError, TypeError, KeyError) as error:
        result['error'] = str(error)
    result['wall_seconds'] = time.monotonic() - start
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', help='exact registered certificate id')
    parser.add_argument('--list', action='store_true', help='list reviewed registry entries')
    args = parser.parse_args()
    if args.list:
        try:
            for entry in validate_registry(read_json(REGISTRY)):
                print(f"{entry['id']}: {entry['scope']}")
            return 0
        except (ValueError, OSError, TypeError) as error:
            print(f'FAIL: {error}', file=sys.stderr)
            return 1
    result = run(args.only)
    destination = LAB / '_generated/focused/SUMMARY.json'
    # The writer has one fixed workspace output, never a manifest-supplied path.
    require(destination.resolve().is_relative_to(LAB.resolve()), 'receipt path escapes lab')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
