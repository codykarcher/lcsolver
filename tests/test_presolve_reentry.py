"""Re-entry safety of the presolve pipeline, and honest attach failures.

Every solve-in-a-loop workflow (Monte Carlo, sweeps, continuation)
re-corrects and re-detects. These pin the three failure modes that made
that impossible, found on the lcjetliner b737 anchor (2026-08-29):

1. `unit_corrector` on an already-corrected model crashed deleting the
   detector's bracketed-name bound rows ('FS_M[0]_lowerBound') -- and
   `sensitivities()` re-corrects internally, so every post-solve
   sensitivity read on a pre-corrected model failed.
2. `structure_detector` re-detection collided with its own bound rows,
   and the collision fallback raised even when its random-suffix retry
   SUCCEEDED (the raise sat outside the success check).
3. `_attach_sensitivities` swallowed such failures silently: an empty
   table at a certified optimum with no message anywhere.
"""

import warnings

import pytest

import lcsolver
from lcsolver import Formulation
from lcsolver.presolve.structureDetector import structure_detector
from lcsolver.presolve.unitCorrector import unit_corrector


def _model():
    f = Formulation()
    F = f.Variable(name="F", guess=10.0, units="N", description="epigraph objective")
    x = f.Variable(name="x", guess=1.0, units="m", description="free length",
                   bounds=[1e-3, 1e3])
    f.Constant(name="c1", value=2.0, units="N/m", description="linear cost")
    f.Constant(name="c2", value=8.0, units="N*m", description="inverse cost")
    f.Objective(F)
    f.ConstraintList([f.c1 * x + f.c2 / x <= F])
    return f


def test_unit_corrector_idempotent():
    f = _model()
    once = unit_corrector(f)
    assert getattr(once, "_lc_unit_corrected", False)
    # re-correcting must neither crash nor hand back the same object
    twice = unit_corrector(once)
    assert twice is not once
    # and the result must still be solvable machinery: detect + count rows
    st1 = structure_detector(unit_corrector(f))
    st2 = structure_detector(twice)
    assert len(st1["variables"]) == len(st2["variables"])


def test_unit_corrector_survives_detected_model():
    # the b737 failure shape: correct a model that ALREADY carries
    # detector bound rows with bracketed/odd names
    f = _model()
    corrected = unit_corrector(f)
    structure_detector(corrected)          # adds bound rows to `corrected`
    again = unit_corrector(corrected)      # crashed before the fix
    assert again is not corrected


def test_structure_detector_reentry():
    f = _model()
    corrected = unit_corrector(f)
    st1 = structure_detector(corrected)
    n1 = st1["info"]["N_cons_total"]
    # second pass on the SAME component: replaces its own rows, same count
    st2 = structure_detector(corrected)
    assert st2["info"]["N_cons_total"] == n1


def test_detector_user_owned_name_gets_suffix_not_raise():
    import pyomo.environ as pyo

    f = _model()
    corrected = unit_corrector(f)
    # a USER constraint squatting on the detector's preferred key: the
    # detector must fall back to a random-suffix key -- and must not raise
    # when that fallback succeeds (it used to raise unconditionally)
    corrected.x_lowerBound = pyo.Constraint(expr=corrected.x >= 1e-3)
    st = structure_detector(corrected)
    assert st["info"]["N_cons_total"] > 0


def test_attach_reports_sensitivity_failure(monkeypatch):
    f = _model()

    def _boom(model, **kwargs):
        raise RuntimeError("synthetic sensitivity failure")

    import lcsolver.postsolve.sensitivity as sensmod

    monkeypatch.setattr(sensmod, "sensitivities", _boom)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = lcsolver.solve(f)
    assert res.optimality_status  # the solve itself must survive
    detail = res.get("sensitivity_detail") or {}
    assert detail.get("method") == "failed"
    assert "synthetic sensitivity failure" in detail.get("error", "")
    # solve() captures warnings into res.messages; accept either surface
    surfaced = [str(w.message) for w in caught] + [str(m) for m in res.messages]
    assert any("LC-W302" in s for s in surfaced)
