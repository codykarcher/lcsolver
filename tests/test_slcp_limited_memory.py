"""The limited-memory Hessian option must match the dense one.

``Options.hessian_memory`` swaps the dense n-by-n BFGS matrix for a
limited-memory representation. It is a toggle: the default (``None``) leaves
the dense path exactly as it was.

The motivation is the sub-problem, not the update. The dense path builds
``0.5 * sum_ij B[i][j] d_i d_j`` as an n^2-term Pyomo expression every
iteration -- 1.4 million terms at n = 1173. Limited memory makes that
``O(n * memory)``.
"""
import numpy as np
import pyomo.environ as pyo
import pytest

from lcsolver import Formulation
from lcsolver.solvers.ipopt.slcp import LimitedMemoryB, Options, _damped_bfgs
from lcsolver.solvers.ipopt.slcp_bridge import solve_slcp
from lcsolver.presolve.structureDetector import structure_detector
from lcsolver.presolve.unitCorrector import unit_corrector

M = pyo.units.m


def test_matches_dense_bfgs_within_the_memory_window():
    """With memory >= the number of updates, the two are the same matrix."""
    rng = np.random.default_rng(0)
    n = 6
    dense = np.eye(n)
    lm = LimitedMemoryB(n, memory=10)
    for _ in range(4):
        s = rng.normal(size=n)
        z = rng.normal(size=n) + 3.0 * s      # keep curvature mostly positive
        dense = _damped_bfgs(dense, s, z)
        lm.update(s, z)
    probe = rng.normal(size=n)
    assert np.allclose(lm.matvec(probe), dense @ probe, rtol=1e-9, atol=1e-9)


def test_quadratic_form_reconstructs_the_matrix():
    """gamma*I + sum sign_k v_k v_k^T must reproduce B exactly."""
    rng = np.random.default_rng(1)
    n = 5
    lm = LimitedMemoryB(n, memory=10)
    for _ in range(3):
        s = rng.normal(size=n)
        lm.update(s, rng.normal(size=n) + 3.0 * s)
    gamma, pairs = lm.quad_terms()
    B = gamma * np.eye(n)
    for sign, v in pairs:
        B = B + sign * np.outer(v, v)
    probe = rng.normal(size=n)
    assert np.allclose(B @ probe, lm.matvec(probe), rtol=1e-9, atol=1e-9)


def test_memory_window_is_bounded():
    lm = LimitedMemoryB(4, memory=2)
    rng = np.random.default_rng(2)
    for _ in range(10):
        s = rng.normal(size=4)
        lm.update(s, rng.normal(size=4) + 3.0 * s)
    assert len(lm.pairs) <= 2 * lm.memory


def test_reset_clears_the_window():
    lm = LimitedMemoryB(3, memory=5)
    s = np.array([1.0, 2.0, 3.0])
    lm.update(s, s * 4.0)
    assert lm.pairs
    lm.reset()
    assert not lm.pairs
    assert np.allclose(lm.matvec(s), s)      # back to gamma*I


def _box():
    f = Formulation()
    h = f.Variable(name="h", guess=1.0, units="m", description="")
    w = f.Variable(name="w", guess=1.0, units="m", description="")
    d = f.Variable(name="d", guess=1.0, units="m", description="")
    f.Objective(2 * (h * w + h * d + w * d))
    f.ConstraintList([h * w * d >= 8.0 * M ** 3, h <= 4.0 * M, w <= 4.0 * M])
    return f


@pytest.mark.parametrize("memory", [1, 3, 10])
def test_solve_agrees_with_the_dense_default(memory):
    dense = solve_slcp(structure_detector(unit_corrector(_box())))
    lm = solve_slcp(structure_detector(unit_corrector(_box())),
                    options=Options(hessian_memory=memory))
    assert lm.objective is not None, lm.status
    assert lm.objective == pytest.approx(dense.objective, rel=1e-5)


def test_default_is_still_the_dense_matrix():
    assert Options().hessian_memory is None
