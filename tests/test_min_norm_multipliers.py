"""``_min_norm_lsq`` / ``_nnls`` (sia.py) against scipy on small problems.

The certificate solver eliminates the free-sign equality block and runs a
warm Lawson-Hanson on the inequality block; these pin it to ``lsq_linear``
(the formulation it replaced) and ``scipy.optimize.nnls`` on problems with
the features the aircraft Jacobians have: columns spanning decades,
duplicate and dependent columns, more columns than rows.
"""
import numpy as np
import pytest
from scipy.optimize import lsq_linear, nnls

from lcsolver.solvers.sequential.sia import _min_norm_lsq, _nnls


def _problem(rng, n, me, mi, dup=False, dep=False, wide=False):
    m = me + mi
    A = rng.standard_normal((n, m)) * 10.0 ** rng.uniform(-4, 4, size=m)
    if dup and mi >= 2:
        A[:, me + 1] = A[:, me]
    if dep and me >= 3:
        A[:, 2] = A[:, 0] - 2.0 * A[:, 1]
    if dep and mi >= 3:
        A[:, me + 2] = A[:, me] + A[:, me + 1]
    g0 = rng.standard_normal(n)
    free = np.zeros(m, dtype=bool)
    free[:me] = True
    return A, g0, free


@pytest.mark.parametrize("seed", range(12))
def test_min_norm_lsq_matches_lsq_linear(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(8, 40))
    me = int(rng.integers(0, n // 2))
    mi = int(rng.integers(1, n))
    A, g0, free = _problem(rng, n, me, mi, dup=seed % 3 == 0,
                           dep=seed % 4 == 0)
    lam = _min_norm_lsq(A, g0, free)
    assert lam[~free].min() >= 0.0
    scale = np.maximum(np.linalg.norm(A, axis=0), 1e-12)
    r_new = np.linalg.norm(g0 + A @ lam)
    # independent reference: numpy SVD projector onto the equality range
    # (numpy lstsq's rank cut) + scipy Lawson-Hanson on the inequality block
    As = A / scale
    E, B = As[:, free], As[:, ~free]
    if E.shape[1]:
        U, S, _ = np.linalg.svd(E, full_matrices=False)
        r = int(np.count_nonzero(S > np.finfo(float).eps * max(A.shape) * S[0]))
        U = U[:, :r]
        P = np.eye(A.shape[0]) - U @ U.T
    else:
        P = np.eye(A.shape[0])
    mu, _ = nnls(P @ B, -P @ g0, maxiter=10000)
    r_ref = np.linalg.norm(P @ (g0 + B @ mu))
    assert r_new <= r_ref * (1.0 + 1e-8) + 1e-10 * np.linalg.norm(g0)
    # and never worse than the lsq_linear formulation it replaced, unless
    # that one ran off along a numerical null vector (duals ~1e14 on an
    # exactly dependent equality column: the failure mode this exists for)
    lo = np.where(free, -np.inf, 0.0)
    ref = lsq_linear(As, -g0, bounds=(lo, np.inf), tol=1e-14,
                     max_iter=5000, lsq_solver="exact")
    if np.abs(ref.x).max() < 1e8:
        r_old = np.linalg.norm(g0 + A @ (ref.x / scale))
        assert r_new <= r_old * (1.0 + 1e-8) + 1e-10 * np.linalg.norm(g0)
    assert np.abs(lam * scale).max() < 1e8


def test_min_norm_lsq_equalities_only_is_least_squares():
    rng = np.random.default_rng(3)
    A, g0, free = _problem(rng, 20, 7, 0)
    lam = _min_norm_lsq(A, g0, free)
    ref = np.linalg.lstsq(A, -g0, rcond=None)[0]
    assert np.allclose(A @ lam, A @ ref, atol=1e-10)
    assert np.allclose(lam, ref, rtol=1e-8, atol=1e-10)


def test_min_norm_lsq_empty_and_inequalities_only():
    rng = np.random.default_rng(5)
    assert _min_norm_lsq(np.zeros((4, 0)), np.ones(4), np.zeros(0, bool)).size == 0
    A, g0, free = _problem(rng, 15, 0, 6, dup=True)
    lam = _min_norm_lsq(A, g0, free)
    scale = np.linalg.norm(A, axis=0)
    ref, _ = nnls(A / scale, -g0)
    assert lam.min() >= 0.0
    assert np.isclose(np.linalg.norm(g0 + A @ lam),
                      np.linalg.norm(g0 + (A / scale) @ ref), rtol=1e-8)


@pytest.mark.parametrize("seed", range(10))
def test_nnls_matches_scipy(seed):
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(3, 60))
    k = int(rng.integers(1, 80))
    M = rng.standard_normal((n, k)) * 10.0 ** rng.uniform(-3, 3, size=k)
    if seed % 2 and k > 2:
        M[:, 1] = M[:, 0]
    if seed % 3 == 0 and k > 3:
        M[:, 2] = 2 * M[:, 0] - M[:, 1]
    M /= np.linalg.norm(M, axis=0)
    b = rng.standard_normal(n)
    if seed % 4 == 0:
        b = M @ np.abs(rng.standard_normal(k))       # consistent system
    x = _nnls(M, b)
    ref, rn = nnls(M, b, maxiter=10000)
    assert x.min() >= 0.0
    assert np.linalg.norm(M @ x - b) <= rn * (1.0 + 1e-8) + 1e-12 * np.linalg.norm(b)


def test_nnls_zero_rhs_and_no_columns():
    assert _nnls(np.ones((3, 2)), np.zeros(3)).tolist() == [0.0, 0.0]
    assert _nnls(np.zeros((3, 0)), np.ones(3)).size == 0
