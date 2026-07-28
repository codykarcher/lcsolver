"""On-design turbofan sizing (tfsize.f) against the compiled Fortran.

Three cases on a CFM56-class engine at top of climb: no cooling, cooling with
the metal temperature specified, and cooling plus mass and power offtakes --
the last of which exercises the multi-pass convergence loop rather than the
single-pass shortcut.

Reference regenerated with::

    gfortran -fdefault-real-8 -O0 -o drv_tfsize \\
        drv_tfsize.f tfsize.f tfcool.f tfmap.f gascalc.f gasfun.f
    ./drv_tfsize > tests/data/tfsize_ref.csv
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tasopt_py.engine.tfsize import EngineSizingError, tfsize

DATA = Path(__file__).parent / "data"
RTOL = 1e-11

T0 = 219.43
BASE = dict(gee=9.81, M0=0.80, T0=T0, p0=23842.0,
            a0=math.sqrt(1.4 * 287.0 * T0), M2=0.60, M25=0.60,
            Feng=25000.0, Phiinl=0.0, Kinl=0.0, iBLIc=0,
            BPR=5.1, pif=1.685, pilc=1.935, pihc=9.369,
            pid=0.998, pib=0.94, pifn=0.98, pitn=0.989,
            Tt4=1450.0, Ttf=280.0, ifuel=24, etab=0.985,
            epf0=0.8948, eplc0=0.88, ephc0=0.87, epht0=0.889, eplt0=0.899,
            pifK=1.685, epfK=0.0, mofft=0.0, Pofft=0.0,
            Tt9=300.0, pt9=30000.0, epsl=0.0, epsh=0.0, icool=0,
            Mtexit=1.0, dTstrk=200.0, StA=0.09, efilm=0.7, tfilm=0.3,
            M4a=0.9, ruc=0.9, ncrowx=4)
CASES = {
    1: dict(icool=0),
    2: dict(icool=2),
    3: dict(icool=2, mofft=0.5, Pofft=60000.0, Tt4=1550.0),
}


def _solved():
    return {ic: tfsize(**{**BASE, **kw}, Tmrow=[1200.0] * 4)
            for ic, kw in CASES.items()}


def _read(r, name):
    st = r.stations
    direct = {"TSFC": r.TSFC, "Fsp": r.Fsp, "ff": r.ff, "mcore": r.mcore,
              "etaf": r.etaf, "etaht": r.etaht}
    if name in direct:
        return direct[name]
    if name.startswith("Tt"):
        return st[int(name[2:])].Tt
    if name.startswith("pt"):
        return st[int(name[2:])].pt
    if name.startswith("A"):
        return st[int(name[1:])].A
    if name.startswith("u"):
        return st[int(name[1:])].u
    raise KeyError(name)


def test_matches_fortran():
    got = _solved()
    n = 0
    with (DATA / "tfsize_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            ic, name, ref = int(row[0]), row[1].strip(), float(row[2])
            actual = _read(got[ic], name)
            rel = abs(actual - ref) / max(abs(ref), 1e-300)
            assert rel <= RTOL, f"case {ic} {name}: {actual!r} vs {ref!r} rel {rel:.3e}"
            n += 1
    assert n == 54


def test_thrust_is_actually_delivered():
    """Momentum balance: the sized mass flow must produce the thrust asked for."""
    for ic, r in _solved().items():
        st = r.stations
        u0, u6, u8 = st[0].u, st[6].u, st[8].u
        fo = 0.0 if ic == 1 else BASE["mofft"] if ic != 3 else 0.5 / r.mcore
        F = r.mcore * ((1.0 - fo + r.ff) * u6 - u0
                       + BASE["BPR"] * (u8 - u0) + fo * st[9].u)
        target = {**BASE, **CASES[ic]}["Feng"]
        assert F == pytest.approx(target, rel=2e-2), (ic, F, target)


def test_cooling_lowers_turbine_inlet_temperature():
    """Mixing coolant in drops Tt41 below Tt4; without cooling they are equal."""
    got = _solved()
    assert got[1].stations[41].Tt == pytest.approx(BASE["Tt4"], rel=1e-12)
    assert got[2].stations[41].Tt < BASE["Tt4"]
    assert got[2].ncrow > 0


def test_cooling_reduces_fuel_burn_per_core_flow():
    got = _solved()
    assert got[2].ff < got[1].ff


def test_areas_are_positive_and_ordered():
    """The fan duct passes far more air than the core, so A7 >> A5."""
    for r in _solved().values():
        for stn in (2, 25, 5, 7):
            assert r.stations[stn].A > 0.0
        assert r.stations[7].A > r.stations[5].A


def test_impossible_expansion_raises():
    """A fan pressure ratio below ambient recovery cannot make thrust."""
    with pytest.raises(EngineSizingError, match="plume velocity"):
        tfsize(**{**BASE, "pif": 1.001, "pilc": 1.001, "pihc": 1.001,
                  "pid": 0.5, "pib": 0.5}, Tmrow=[1200.0] * 4)
