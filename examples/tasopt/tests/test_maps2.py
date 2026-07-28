"""``Ncmap``, ``etmap`` (tfmap.f) and ``gas_mass`` (gascalc.f) vs the Fortran.

The three routines ``tfoper`` needs beyond what ``tfsize`` already used:
corrected wheel speed, turbine efficiency, and the static state at a
specified mass flux.

References regenerated with::

    gfortran -fdefault-real-8 -O0 -o drv_ncmap drv_ncmap.f tfmap.f
    gfortran -fdefault-real-8 -O0 -o drv_maps2 \\
        drv_maps2.f tfmap.f gascalc.f gasfun.f
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tasopt_py.engine.maps import (CMAPF, CMAPHC, CMAPLC, TMAPH, TMAPL,
                                   MapSpeedError, Ncmap, etmap)
from tasopt_py.gas.mixture import gassum, gas_mass

DATA = Path(__file__).parent / "data"
RTOL = 1e-13

AIR = [0.7532, 0.2315, 0.0006, 0.0020, 0.0127, 0.0]
SPOOLS = {1: (CMAPF, 1.685), 2: (CMAPLC, 1.935), 3: (CMAPHC, 2.100)}


def test_ncmap_matches_fortran():
    n = 0
    with (DATA / "ncmap_ref.csv").open() as fh:
        for row in csv.reader(fh):
            ic, ip, im = int(row[0]), int(row[1]), int(row[2])
            cmap, piD = SPOOLS[ic]
            got = Ncmap(piD + 0.06 * (ip - 2), 34.0 + 3.0 * im,
                        piD, 40.0, 1.05, cmap)
            for a, ref in zip((got.Nb, got.Nb_pi, got.Nb_mb),
                              map(float, row[3:6])):
                assert a == pytest.approx(ref, rel=RTOL)
                n += 1
    assert n == 144


def test_etmap_and_gas_mass_match_fortran():
    o = gassum(AIR, 5, 800.0)
    n = 0
    with (DATA / "maps2_ref.csv").open() as fh:
        for row in csv.reader(fh):
            kind, i = row[0].strip(), int(row[1])
            if kind == "etmap":
                got = [etmap(-2.6e5 - 3.0e4 * i, 8.0 + i, 0.9 + 0.05 * i,
                             4.0, 11.0, 1.05, 0.889, TMAPH,
                             1400.0 + 20.0 * i, 1250.0, 288.0)]
                ref = [float(row[2])]
            else:
                got = list(gas_mass(AIR, 5, 4.0e5, 800.0, o.h, o.s, o.cp,
                                    o.r, 120.0 + 40.0 * i, 0.5)[:4])
                ref = list(map(float, row[2:6]))
            for a, b in zip(got, ref):
                assert a == pytest.approx(b, rel=RTOL)
                n += 1
    assert n == 26


def test_ncmap_recovers_design_speed():
    """At the design point the map must give back the design speed exactly."""
    for cmap, piD in SPOOLS.values():
        got = Ncmap(piD, 40.0, piD, 40.0, 1.05, cmap)
        assert got.Nb == pytest.approx(1.05, rel=1e-9)


def test_ncmap_speed_rises_with_pressure_ratio():
    cmap, piD = SPOOLS[1]
    speeds = [Ncmap(piD + 0.04 * k, 40.0, piD, 40.0, 1.05, cmap).Nb
              for k in range(4)]
    assert speeds == sorted(speeds)


def test_ncmap_off_map_raises():
    """Far above the surge line the log argument goes negative."""
    cmap, piD = SPOOLS[1]
    with pytest.raises((MapSpeedError, ValueError)):
        Ncmap(piD + 3.0, 12.0, piD, 40.0, 1.05, cmap)


def test_etmap_peaks_at_design_point():
    """Both penalties are quadratic and centred on design, so the peak is there."""
    ept0, cpt, Rt, Tt = 0.889, 1250.0, 288.0, 1420.0
    # Choose dh so prat lands exactly on piD, and mb*Nb on its design value.
    piD, mbD, NbD = 4.0, 11.0, 1.05
    gex = cpt / (Rt * ept0)
    Trat = piD ** (1.0 / gex)
    dh = cpt * Tt * (1.0 / Trat - 1.0)
    peak = etmap(dh, mbD, NbD, piD, mbD, NbD, ept0, TMAPH, Tt, cpt, Rt)
    assert peak == pytest.approx(ept0, rel=1e-12)
    for off in (0.9, 1.1):
        assert etmap(dh * off, mbD, NbD, piD, mbD, NbD, ept0, TMAPH,
                     Tt, cpt, Rt) < peak


def test_turbine_maps_are_identical():
    """tfmap.inc gives the HP and LP turbines the same two constants."""
    assert TMAPL == TMAPH == (0.15, 0.15)


def test_gas_mass_inverts_the_mass_flux():
    """rho*u recovered from the returned static state must equal mflux."""
    o = gassum(AIR, 5, 800.0)
    for mflux in (160.0, 240.0, 300.0):
        p, t, h, s, cp, r = gas_mass(AIR, 5, 4.0e5, 800.0, o.h, o.s,
                                     o.cp, o.r, mflux, 0.5)
        u = (2.0 * (o.h - h)) ** 0.5
        assert p / (r * t) * u == pytest.approx(mflux, rel=1e-6)


def test_gas_mass_branch_is_picked_by_the_guess():
    """A supersonic guess lands on the other root of the same mass flux."""
    o = gassum(AIR, 5, 800.0)
    sub = gas_mass(AIR, 5, 4.0e5, 800.0, o.h, o.s, o.cp, o.r, 200.0, 0.5)
    sup = gas_mass(AIR, 5, 4.0e5, 800.0, o.h, o.s, o.cp, o.r, 200.0, 1.8)
    assert sup[1] < sub[1]      # supersonic branch is colder
