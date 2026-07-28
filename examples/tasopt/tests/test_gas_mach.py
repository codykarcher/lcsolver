"""gas_mach (gascalc.f) against the compiled Fortran.

State change across a Mach number change, with variable cp. Needed by
``tfsize``, which calls it six times.

Five cases: static to M0.8, a hot decelerating flow, a deceleration to rest,
a compression at 90% polytropic efficiency, and an expansion (epol > 1)
through M1.

Reference regenerated with::

    gfortran -fdefault-real-8 -O0 -o drv_gasmach drv_gasmach.f gascalc.f gasfun.f
    ./drv_gasmach > tests/data/gasmach_ref.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.gas.mixture import gas_mach, gassum

DATA = Path(__file__).parent / "data"
RTOL = 1e-12

AIR = [0.7570, 0.2320, 0.0006, 0.0104, 0.0]
N = 5
CASES = {
    1: dict(to=288.0, po=101325.0, mo=0.0, m=0.8, epol=1.0),
    2: dict(to=800.0, po=101325.0, mo=0.0, m=0.35, epol=1.0),
    3: dict(to=250.0, po=101325.0, mo=0.8, m=0.0, epol=1.0),
    4: dict(to=1400.0, po=900000.0, mo=0.0, m=0.6, epol=0.90),
    5: dict(to=288.0, po=101325.0, mo=0.3, m=1.0, epol=1.05),
}
FIELDS = ["p", "t", "h", "s", "cp", "r"]


def _solve(kw):
    st = gassum(AIR, N, kw["to"])
    return gas_mach(AIR, N, kw["po"], kw["to"], st.h, st.s, st.cp, st.r,
                    kw["mo"], kw["m"], kw["epol"])


def test_matches_fortran():
    got = {ic: _solve(kw) for ic, kw in CASES.items()}
    n = 0
    with (DATA / "gasmach_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            ic, name, ref = int(row[0]), row[1].strip(), float(row[2])
            actual = got[ic][FIELDS.index(name)]
            rel = abs(actual - ref) / max(abs(ref), 1e-300)
            assert rel <= RTOL, f"case {ic} {name}: {actual!r} vs {ref!r} rel {rel:.3e}"
            n += 1
    assert n == 30


def test_stagnation_enthalpy_is_conserved():
    """The whole point: h + u^2/2 is unchanged across the Mach change."""
    for kw in CASES.values():
        st0 = gassum(AIR, N, kw["to"])
        p, t, h, s, cp, r = _solve(kw)
        uo2 = kw["mo"] ** 2 * st0.cp * st0.r / (st0.cp - st0.r) * kw["to"]
        u2 = kw["m"] ** 2 * cp * r / (cp - r) * t
        assert h + 0.5 * u2 == pytest.approx(st0.h + 0.5 * uo2, rel=1e-9)


def test_gas_constant_is_unchanged():
    """Composition does not change, so R must not either."""
    for kw in CASES.values():
        st0 = gassum(AIR, N, kw["to"])
        assert _solve(kw)[5] == pytest.approx(st0.r, rel=1e-12)


def test_accelerating_cools_the_flow():
    p, t, *_ = _solve(CASES[1])          # M 0 -> 0.8
    assert t < CASES[1]["to"]
    assert p < CASES[1]["po"]
