"""Fuselage BL driver (fusebl.f) against the compiled Fortran.

The shipped 737 fuselage (737.tas geometry, in metres) at cruise, smooth and
with the excrescence factor, at three other flight conditions, and as a
double bubble with a point tail and with an edge tail.

This is the whole chain -- ``axisol`` -> perimeter and dr/dn -> ``blax`` ->
Squire-Young -- so it measures the composition rather than any one module.
The four outputs come out at 2.8e-14 across all seven cases. That is looser
than ``axisol`` (4.4e-16) or ``blax`` driven with exact inputs (3e-14 worst,
1e-15 on the state variables) because the coupled Newton amplifies whatever
differs upstream of it; the tolerance below is a property of the solve, not of
the port.

Case 1 is not a constructed case
--------------------------------
Instrumenting ``fusebl.f`` to dump its input state and re-running the shipped
737 shows the program calls it exactly twice, with exactly the inputs case 1
sets -- the geometry does not change over the sizing loop, so there is only
one fuselage BL solve in a 737 sizing. Case 1 is that solve, and the port
reproduces it to 1.4e-15; ``test_matches_the_real_737_run`` pins the four
numbers. The instrumented source is kept in
``fortran_ref/fusebl_instrumented.f``; the shipped file was restored
afterwards and still sizes the 737 to WTO = 174979.1499 lbf in 18 iterations.

Reference regenerated with::

    gfortran -fdefault-real-8 -O0 -fdollar-ok -I. -o drv_fusebl \\
        drv_fusebl.f fusebl.f blax.f blsys.f axisol.f gaussn.f atmos.f
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.aero.fusebl import fusebl
from tasopt_py.model import Aircraft
from tasopt_py.model import indices as I

DATA = Path(__file__).parent / "data"
RTOL = 1e-12

FT, IN = 0.3048, 0.0254

NAMES = ["DAfsurf", "DAfwake", "KAfTE", "PAfinf"]
IDX = [I.IADAFSURF, I.IADAFWAKE, I.IAKAFTE, I.IAPAFINF]


def _setup(icase):
    """Rebuild one driver case. Mirrors drv_fusebl.f exactly."""
    ac = Aircraft()
    pari, parg, para = ac.pari, ac.parg, ac.para.column(1)

    pari[I.IIFCLOSE] = 0
    parg[I.IGXNOSE] = 0.0 * FT
    parg[I.IGXEND] = 124.0 * FT
    parg[I.IGXBLEND1] = 20.0 * FT
    parg[I.IGXBLEND2] = 97.0 * FT
    parg[I.IGANOSE] = 1.65
    parg[I.IGBTAIL] = 2.0
    parg[I.IGRFUSE] = 77.0 * IN
    parg[I.IGDRFUSE] = 15.0 * IN
    parg[I.IGWFB] = 0.0

    para[I.IAMACH] = 0.80
    para[I.IAALT] = 35000.0 * FT
    para[I.IAFEXCDF] = 1.03

    if icase == 2:
        para[I.IAFEXCDF] = 1.0
    if icase == 3:
        para[I.IAMACH], para[I.IAALT] = 0.60, 20000.0 * FT
    if icase == 4:
        para[I.IAMACH], para[I.IAALT] = 0.20, 0.0
    if icase == 5:
        parg[I.IGWFB], parg[I.IGDRFUSE] = 0.40, 30.0 * IN
    if icase == 6:
        pari[I.IIFCLOSE] = 1
        parg[I.IGWFB], parg[I.IGDRFUSE] = 0.40, 30.0 * IN
    if icase == 7:
        parg[I.IGRFUSE] = 90.0 * IN
        parg[I.IGANOSE] = 2.20
        parg[I.IGBTAIL] = 1.40
    return pari, parg, para


def _run(icase):
    pari, parg, para = _setup(icase)
    fusebl(pari, parg, para)
    return para


RESULTS = {ic: _run(ic) for ic in range(1, 8)}


def test_matches_fortran():
    n = 0
    with (DATA / "fusebl_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            ic = int(row[0])
            para = RESULTS[ic]
            for name, idx, want in zip(NAMES, IDX, map(float, row[1:5])):
                assert para[idx] == pytest.approx(want, rel=RTOL), \
                    f"case {ic} {name}"
                n += 1
    assert n == 28


#: What the shipped program actually computed, dumped out of the one fuselage
#: BL solve in a converged 737 sizing. See the module docstring.
REAL_737 = {I.IADAFSURF: 0.859625359180616333,
            I.IADAFWAKE: 0.0774391139866845712,
            I.IAKAFTE: 0.917384346185931565,
            I.IAPAFINF: 0.950005096504426771}


def test_matches_the_real_737_run():
    """Not a driver case reproduced -- the shipped program's own numbers."""
    para = RESULTS[1]
    for idx, want in REAL_737.items():
        assert para[idx] == pytest.approx(want, rel=1e-14)


def test_all_four_areas_are_positive_and_ordered():
    for ic, para in RESULTS.items():
        surf, wake = para[I.IADAFSURF], para[I.IADAFWAKE]
        kte, pinf = para[I.IAKAFTE], para[I.IAPAFINF]
        assert surf > 0.0 and wake > 0.0 and kte > 0.0 and pinf > 0.0
        # Most of the dissipation happens on the body, not in the wake.
        assert wake < 0.2 * surf, ic


def test_excrescence_factor_raises_every_area():
    """Cases 1 and 2 differ only in fexcdf: 1.03 against 1.0."""
    rough, smooth = RESULTS[1], RESULTS[2]
    for idx in IDX:
        assert rough[idx] > smooth[idx]
    # 3% more skin friction is worth about 2% more fuselage drag.
    ratio = rough[I.IAPAFINF] / smooth[I.IAPAFINF]
    assert 1.01 < ratio < 1.03


def test_fuselage_drag_is_a_sensible_cd():
    """PAfinf is a drag area. cdsum divides it by the wing area."""
    S = 124.0
    assert 0.005 < RESULTS[1][I.IAPAFINF] / S < 0.010


def test_a_double_bubble_costs_drag():
    """Cases 5 and 6 widen the section; only the area reaches axisol, but a
    bigger area means a bigger equivalent round body."""
    assert RESULTS[5][I.IAPAFINF] > RESULTS[1][I.IAPAFINF]


def test_edge_tail_adds_perimeter_and_so_dissipation():
    """Cases 5 and 6 have identical cross-sections; case 6 closes to an edge,
    which adds 4*dy to the perimeter aft of the second blend point."""
    assert RESULTS[6][I.IADAFSURF] > RESULTS[5][I.IADAFSURF]
    assert RESULTS[6][I.IADAFWAKE] > RESULTS[5][I.IADAFWAKE]


def test_fusebl_only_writes_its_four_outputs():
    pari, parg, para = _setup(1)
    before = [para[i] for i in range(1, len(para) + 1)]
    fusebl(pari, parg, para)
    changed = {i for i in range(1, len(para) + 1)
               if para[i] != before[i - 1]}
    assert changed == set(IDX)
