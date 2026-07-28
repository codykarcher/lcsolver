"""Off-design turbofan operation (tfoper.f) against the compiled Fortran.

A CFM56-class engine is sized on design, its map anchors are built exactly as
``tfcalc.f`` builds them, and it is then run off design four ways: burner
temperature specified and thrust specified, each with and without turbine
cooling plus mass/power offtakes.

Tolerance is 1e-8 rather than the 1e-13 the closed-form modules reach. That is
inherent, not sloppiness: this is a nine-variable Newton solve whose
termination test is on the size of the step, and the port differentiates
numerically where the Fortran differentiates analytically. The two runs stop at
slightly different points inside the same convergence ball. Observed worst
disagreement across all 124 compared values is 9e-10.

Reference regenerated with::

    gfortran -fdefault-real-8 -fdollar-ok -O0 -o drv_tfoper drv_tfoper.f \\
        tfoper.f tfsize.f tfcool.f tfmap.f gascalc.f gasfun.f gaussn.f

``-fdollar-ok`` is needed because tfoper.f declares ``res$``/``a$`` for a
debug block. The driver also supplies a ``compare`` stub: ``tfoper`` references
``compare``, which lives in ``compare.f`` -- a file that is not a dependency of
any numerical module, so a minimal link does not pull it in. Linking
``compare.f`` instead works equally well.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tasopt_py.engine.tfoper import TFOperError, tfoper
from tasopt_py.engine.tfsize import tfsize

DATA = Path(__file__).parent / "data"
RTOL = 1e-8

T0D, P0D, M0D = 219.43, 23842.0, 0.80
TREF, PREF, GEARF, BPRD = 288.2, 101320.0, 1.0, 5.1

CASES = {
    1: dict(icool=0, iTFspec=1, Tt4=1450.0),
    2: dict(icool=0, iTFspec=2, Feng=23000.0, Tt4=1450.0),
    3: dict(icool=2, iTFspec=1, Tt4=1500.0, mofft=0.5, Pofft=60000.0),
    4: dict(icool=2, iTFspec=2, Feng=26000.0, Tt4=1450.0,
            mofft=0.5, Pofft=60000.0),
}

_COMMON = dict(Ttf=280.0, ifuel=24, etab=0.985, epf0=0.8948, eplc0=0.88,
               ephc0=0.87, epht0=0.889, eplt0=0.899, pifK=1.685, epfK=0.0,
               Tt9=300.0, pt9=30000.0, epsl=0.0, epsh=0.0, Mtexit=1.0,
               dTstrk=200.0, StA=0.09, efilm=0.7, tfilm=0.3, M4a=0.9,
               ruc=0.9, ncrowx=4)


def _design(icool=0, mofft=0.0, Pofft=0.0):
    """Size on design and build the map anchors the way tfcalc.f does."""
    r = tfsize(gee=9.81, M0=M0D, T0=T0D, p0=P0D,
               a0=math.sqrt(1.4 * 287.0 * T0D), M2=0.60, M25=0.60,
               Feng=25000.0, Phiinl=0.0, Kinl=0.0, iBLIc=0, BPR=BPRD,
               pif=1.685, pilc=1.935, pihc=9.369, pid=0.998, pib=0.94,
               pifn=0.98, pitn=0.989, Tt4=1450.0, mofft=mofft, Pofft=Pofft,
               icool=icool, Tmrow=[1200.0] * 4, **_COMMON)
    s, mc, ff = r.stations, r.mcore, r.ff
    fo = mofft / mc

    def cm(stn, extra=1.0):
        return mc * math.sqrt(s[stn].Tt / TREF) / (s[stn].pt / PREF) * extra

    A = dict(mbfD=cm(2, BPRD), mblcD=cm(19), mbhcD=cm(25, 1 - fo),
             mbhtD=cm(41, 1 - fo + ff), mbltD=cm(45, 1 - fo + ff),
             NbfD=(1.0 / GEARF) / math.sqrt(s[2].Tt / TREF),
             NblcD=1.0 / math.sqrt(s[19].Tt / TREF),
             NbhcD=1.0 / math.sqrt(s[25].Tt / TREF),
             NbhtD=1.0 / math.sqrt(s[41].Tt / TREF),
             NbltD=1.0 / math.sqrt(s[45].Tt / TREF),
             pifD=1.685, pilcD=1.935, pihcD=9.369,
             A2=s[2].A, A25=s[25].A, A5=s[5].A, A7=s[7].A)
    # Turbine design pressure ratios in the approximate temperature form that
    # etmap expects -- NOT pt45/pt41. See maps.etmap.
    Trh = s[41].Tt / (s[41].Tt + (s[45].ht - s[41].ht) / s[41].cpt)
    Trl = s[45].Tt / (s[45].Tt + (s[49].ht - s[45].ht) / s[45].cpt)
    A["pihtD"] = Trh ** (s[41].cpt / (s[41].Rt * 0.889))
    A["piltD"] = Trl ** (s[45].cpt / (s[45].Rt * 0.899))
    return r, A


def _run(A, *, M0=M0D, T0=T0D, p0=P0D, Tt4=1450.0, iTFspec=1, Feng=0.0,
         icool=0, mofft=0.0, Pofft=0.0, warm=None):
    kw = dict(gee=9.81, M0=M0, T0=T0, p0=p0, a0=math.sqrt(1.4 * 287.0 * T0),
              Tref=TREF, pref=PREF, Phiinl=0.0, Kinl=0.0, iBLIc=0, pid=0.998,
              pib=0.94, pifn=0.98, pitn=0.989, Gearf=GEARF, iTFspec=iTFspec,
              mofft=mofft, Pofft=Pofft, icool=icool, ncrow=0,
              Tmrow=[1200.0] * 4, Tt4=Tt4, Feng=Feng, **_COMMON, **A)
    if warm:
        kw.update(warm)
    return tfoper(**kw)


def _warm_state(r):
    """The starting guess a marching caller would hand the next point."""
    s2 = r.stations[2]
    return dict(pif=r.pif, pilc=r.pilc, pihc=r.pihc, mbf=r.mbf, mblc=r.mblc,
                mbhc=r.mbhc, pt5=r.stations[5].pt,
                M2=s2.u / math.sqrt(s2.T * s2.cp * s2.R / (s2.cp - s2.R)))


def _solve(ic):
    kw = dict(CASES[ic])
    _, A = _design(icool=kw["icool"], mofft=kw.get("mofft", 0.0),
                   Pofft=kw.get("Pofft", 0.0))
    warm = _run(A, icool=kw["icool"], mofft=kw.get("mofft", 0.0),
                Pofft=kw.get("Pofft", 0.0), Tt4=1450.0)
    return _run(A, warm=_warm_state(warm), **kw)


def _read(r, name):
    st = r.stations
    direct = {"TSFC": r.TSFC, "Fsp": r.Fsp, "ff": r.ff, "Feng": r.Feng,
              "mcore": r.mcore, "pif": r.pif, "pilc": r.pilc, "pihc": r.pihc,
              "mbf": r.mbf, "mblc": r.mblc, "mbhc": r.mbhc, "Nbf": r.Nbf,
              "Nblc": r.Nblc, "Nbhc": r.Nbhc, "etaf": r.etaf,
              "etaht": r.etaht, "eplt": r.eplt}
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
    if name.startswith("M"):
        s = st[int(name[1:])]
        return s.u / math.sqrt(s.T * s.cp * s.R / (s.cp - s.R))
    raise KeyError(name)


def test_matches_fortran():
    got = {ic: _solve(ic) for ic in CASES}
    n = 0
    with (DATA / "tfoper_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            ic, name, ref = int(row[0]), row[1].strip(), float(row[2])
            actual = _read(got[ic], name)
            rel = abs(actual - ref) / max(abs(ref), 1e-300)
            assert rel <= RTOL, f"case {ic} {name}: {actual!r} vs {ref!r} rel {rel:.3e}"
            n += 1
    assert n == 124


def test_design_point_reproduces_tfsize():
    """Run at its own design condition, tfoper must give tfsize back."""
    sized, A = _design()
    r = _run(A)
    assert r.mcore == pytest.approx(sized.mcore, rel=1e-10)
    assert r.TSFC == pytest.approx(sized.TSFC, rel=1e-10)
    assert r.pif == pytest.approx(1.685, rel=1e-9)
    assert r.pilc == pytest.approx(1.935, rel=1e-9)
    assert r.pihc == pytest.approx(9.369, rel=1e-9)


def test_ambient_pressure_only_scales_the_engine():
    """Every corrected quantity is invariant to p0; mass flow scales with it.

    Holding M0 and T0 fixed, pt0 is proportional to p0 and so is every total
    pressure in the gas path. The corrected mass flows, pressure ratios and
    Mach numbers are all normalised by those pressures, so they cannot change;
    only the physical mass flow does, in exact proportion. TSFC, being a ratio,
    is unchanged too.

    This is a property of the equations, not of any particular solver, and it
    holds here to 1e-9. The shipped Fortran cannot reach this solution at all
    -- its Newton iteration drifts away from it. See DISCREPANCIES.md.
    """
    _, A = _design()
    warm = _warm_state(_run(A))
    base = _run(A, warm=warm)
    scale = 1.2582837010317926          # 30000/23842
    hi = _run(A, warm=warm, p0=P0D * scale)

    assert hi.mcore / base.mcore == pytest.approx(scale, rel=1e-9)
    assert hi.TSFC == pytest.approx(base.TSFC, rel=1e-9)
    for a, b in ((base.pif, hi.pif), (base.pilc, hi.pilc),
                 (base.pihc, hi.pihc), (base.mblc, hi.mblc),
                 (base.Nblc, hi.Nblc)):
        assert b == pytest.approx(a, rel=1e-9)


def test_hotter_burner_gives_more_thrust():
    _, A = _design()
    warm = _warm_state(_run(A))
    thrust = [_run(A, warm=warm, Tt4=T).Feng for T in (1350.0, 1450.0, 1550.0)]
    assert thrust == sorted(thrust)


def test_thrust_spec_and_temperature_spec_agree():
    """Solving for Tt4 at a given thrust must invert solving for thrust at Tt4."""
    _, A = _design()
    warm = _warm_state(_run(A))
    a = _run(A, warm=warm, Tt4=1400.0)
    b = _run(A, warm=warm, iTFspec=2, Feng=a.Feng)
    assert b.stations[4].Tt == pytest.approx(1400.0, rel=1e-7)
    assert b.mcore == pytest.approx(a.mcore, rel=1e-7)


def test_cooling_costs_thrust_at_fixed_burner_temperature():
    """Bleeding air around the first turbine rows takes work out of the cycle."""
    _, Ah = _design()
    hot = _run(Ah, warm=_warm_state(_run(Ah)), Tt4=1500.0)
    _, Ac = _design(icool=2)
    cooled = _run(Ac, warm=_warm_state(_run(Ac, icool=2)), Tt4=1500.0,
                  icool=2)
    assert cooled.ncrow > 0
    assert cooled.Feng < hot.Feng


def test_offtakes_reduce_net_thrust():
    """One fixed engine, run with and without bleed and shaft power drawn off.

    Sizing a second engine *around* the offtakes instead would just give back
    the thrust it was sized for, which tests nothing.
    """
    _, A = _design()
    warm = _warm_state(_run(A))
    without = _run(A, warm=warm)
    with_off = _run(A, warm=warm, mofft=0.5, Pofft=60000.0)
    assert with_off.Feng < without.Feng


def test_fan_face_mach_stays_under_the_cap():
    for ic in CASES:
        r = _solve(ic)
        s2 = r.stations[2]
        M2 = s2.u / math.sqrt(s2.T * s2.cp * s2.R / (s2.cp - s2.R))
        assert 0.0 < M2 <= 0.98 + 1e-12


def test_thrust_balances_momentum():
    """The reported thrust must equal the momentum flux it is built from."""
    for ic in CASES:
        r = _solve(ic)
        st = r.stations
        fo = CASES[ic].get("mofft", 0.0) / r.mcore
        F = ((1.0 - fo + r.ff) * st[6].u - st[0].u
             + r.BPR * (st[8].u - st[0].u) + fo * st[9].u) * r.mcore
        assert F == pytest.approx(r.Feng, rel=1e-9)


def test_no_operating_envelope_is_enforced():
    """Asking for 16x the design thrust converges -- to a nonsense engine.

    Neither this routine nor the source it ports has any envelope check: the
    maps are smooth extrapolations and the gas model is a polynomial fit, so
    Newton happily lands on a burner at 4240 K and a fan pressure ratio of 7.7
    and reports success. Callers are responsible for bounding what they ask
    for; ``converged`` means the residuals are zero, not that the answer is
    physical.
    """
    _, A = _design()
    r = _run(A, warm=_warm_state(_run(A)), iTFspec=2, Feng=400000.0)
    assert r.converged
    assert r.Feng == pytest.approx(400000.0, rel=1e-8)
    assert r.stations[4].Tt > 4000.0      # far past any real material limit
    assert r.pif > 7.0                    # far past any single-stage fan
