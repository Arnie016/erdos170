import math

import pytest

from sparse_ruler.metrics import compute_W, summarize_metrics, validate_complete


def assert_identity(N, A):
    W = compute_W(A, N)
    m = len(A)
    S = m * (m - 1) // 2
    assert sum(W[1:]) == S
    return W


def test_n1_basic():
    N = 1
    A = [0, 1]
    W = assert_identity(N, A)
    metrics = summarize_metrics(N, A, W)
    assert validate_complete(W, N)
    assert W[1] == 1
    assert metrics.m == 2
    assert metrics.S == 1
    assert metrics.Emin == 1
    assert metrics.E == 1


def test_n6_complete():
    N = 6
    A = [0, 1, 4, 6]
    W = assert_identity(N, A)
    assert validate_complete(W, N)


def test_n13_complete_peak():
    N = 13
    A = [0, 1, 6, 9, 11, 13]
    W = assert_identity(N, A)
    metrics = summarize_metrics(N, A, W)
    assert validate_complete(W, N)
    assert metrics.peak <= 2


def test_n29_complete():
    N = 29
    A = [0, 1, 3, 6, 13, 20, 24, 28, 29]
    W = assert_identity(N, A)
    assert validate_complete(W, N)


def test_must_have_endpoints():
    N = 5
    A = [0, 2, 5]
    W = compute_W(A, N)
    assert math.isclose(sum(W[1:]), 3)


