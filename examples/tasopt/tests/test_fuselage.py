"""Fuselage weight port (fusew.f) against the compiled Fortran.

Three configurations, chosen to exercise the branches:

  case 1  single tube with a floor extension (wfb = 0 -> full-width floor)
  case 2  multi-bubble (wfb > 0 -> centre-supported floor, ksum loop runs)
  case 3  larger tube, heavier payload, higher cabin pressure
"""
from __future__ import annotations

import csv
from pathlib import Path

from tasopt_py.structures.fuselage import fusew

DATA = Path(__file__).parent / "data"
RTOL = 1e-13

BASE = dict(
    gee=9.81, Nland=6.0, Wfix=13000.0, Wpay=175000.0, Wpadd=30000.0,
    Wseat=25000.0, Wapu=5000.0, Weng=0.0,
    fstring=0.35, fframe=0.25, ffadd=0.20, deltap=55000.0,
    Wpwindow=435.0, Wppinsul=22.0, Wppfloor=60.0,
    Whtail=12000.0, Wvtail=9000.0, rMh=0.4, rMv=0.7,
    Lhmax=200000.0, Lvmax=180000.0, bv=8.0, lambdav=0.3, nvtail=1.0,
    Rfuse=1.9, dRfuse=0.3, wfb=0.0, nfweb=0.0, lambdac=0.3,
    xnose=0.0, xshell1=5.0, xshell2=31.0, xconend=36.0,
    xhtail=34.0, xvtail=33.0, xwing=18.0, xwbox=18.0, cbox=3.0,
    xfix=3.0, xapu=35.0, xeng=16.0, hfloor=0.13,
    sigskin=1.5e8, sigbend=1.5e8, rhoskin=2700.0, rhobend=2700.0,
    Eskin=6.9e10, Ebend=6.9e10, Gskin=2.4e10,
)

CASES = {
    1: {},
    2: dict(wfb=0.4, nfweb=1.0, Rfuse=1.7),
    3: dict(Rfuse=2.6, dRfuse=0.0, Wpay=350000.0, deltap=60000.0,
            xshell2=45.0, xconend=52.0, xhtail=49.0, xvtail=48.0,
            xwing=26.0, xwbox=26.0),
}


def test_fuselage_matches_fortran():
    with (DATA / "fusew_ref.csv").open() as fh:
        rdr = csv.DictReader(fh)
        fields = [f.strip() for f in rdr.fieldnames]
        n = 0
        for raw in rdr:
            row = {k.strip(): v for k, v in raw.items()}
            ic = int(row["case"])
            kw = dict(BASE)
            kw.update(CASES[ic])
            res = fusew(**kw)
            for f in fields[1:]:
                ref = float(row[f])
                got = getattr(res, f)
                rel = abs(got - ref) / max(abs(ref), 1e-300)
                assert rel <= RTOL, f"case {ic} {f}: {got!r} vs {ref!r} rel {rel:.3e}"
                n += 1
    assert n == 66


def test_weight_breakdown_sums_to_total():
    """Wfuse must equal the sum of its components (property, not a value)."""
    res = fusew(**BASE)
    parts = (BASE["Wfix"] + BASE["Wapu"] + BASE["Wpadd"] + BASE["Wseat"]
             + res.Wshell + res.Wcone + res.Wwindow + res.Winsul
             + res.Wfloor + res.Whbend + res.Wvbend)
    assert abs(res.Wfuse - parts) / res.Wfuse < 1e-14


def test_multibubble_floor_is_lighter_per_unit_load():
    """A centre-supported floor carries the same payload with less material.

    The centre support cuts the floor bending moment from P*w/4 to 9P*w/256,
    so for the same payload the multi-bubble floor beams must be lighter.
    """
    single = fusew(**BASE)
    multi = fusew(**{**BASE, "wfb": 0.4, "nfweb": 1.0})
    assert multi.Wfloor < single.Wfloor
