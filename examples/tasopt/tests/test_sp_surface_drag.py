"""The signomial form of surfcd must reproduce the exact TASOPT routine.

This makes the York claim checkable: if TASOPT's physics can be recast as a
signomial program, solving the SP with the geometry fixed has to return what
the Fortran-matching port returns. Both asymptotic branches of the spanwise
integral are covered, since that is where the two formulations could most
easily diverge -- and where the SP form has to divide by a sum, which is what
motivated teaching EDI's detector to add signomial fractions.
"""
from __future__ import annotations

import pytest

from sp.surface_drag import DESIGNS, verify


@pytest.mark.parametrize("case", range(len(DESIGNS)))
def test_sp_matches_exact_surfcd(case):
    name, got, exact, rel = verify(designs=[DESIGNS[case]])[0]
    assert rel < 1e-6, f"{name}: SP {got!r} vs TASOPT {exact!r} (rel {rel:.2e})"


def test_all_designs_agree_to_seven_figures():
    worst = max(rel for _, _, _, rel in verify())
    assert worst < 1e-6, worst
