"""Pitch trim, CG limits and tail sizing (balance.f) vs the compiled Fortran.

Six cases: ``balance`` with each of its four ``itrim`` modes, then ``htsize``
with fixed tail volume + cruise-trim wing placement, and with trim-power tail
sizing + stability-margin wing placement.

``constants.inc`` is a COMMON block that ``tasopt.f`` fills at startup, so the
driver has to fill it too -- otherwise ``pi`` is zero and the engine inlet's
contribution to the neutral point silently vanishes.

Reference regenerated with::

    gfortran -fdefault-real-8 -fdollar-ok -O0 -o drv_balance \\
        drv_balance.f balance.f
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tasopt_py.model import Aircraft
from tasopt_py.model import indices as I
from tasopt_py.sizing.balance import balance, cglpay, htsize

DATA = Path(__file__).parent / "data"
RTOL = 1e-12

GEOM = {
    I.IGWMTO: 7.5e5, I.IGWPAY: 1.7e5, I.IGWFUEL: 1.6e5, I.IGWFUSE: 1.9e5,
    I.IGWWING: 9.5e4, I.IGWSTRUT: 0.0, I.IGWHTAIL: 1.4e4, I.IGWVTAIL: 1.1e4,
    I.IGWENG: 5.3e4, I.IGFHPESYS: 0.010, I.IGFLGNOSE: 0.011,
    I.IGFLGMAIN: 0.044, I.IGXSHELL1: 5.2, I.IGXSHELL2: 31.0,
    I.IGXWBOX: 17.0, I.IGXWING: 17.6, I.IGXHBOX: 34.0, I.IGXVBOX: 33.0,
    I.IGXHTAIL: 34.6, I.IGXVTAIL: 33.5, I.IGXENG: 15.0, I.IGXHPESYS: 19.0,
    I.IGXLGNOSE: 5.0, I.IGDXLGMAIN: 0.8, I.IGXWFUSE: 3.4e6,
    I.IGDXWFUEL: 0.0, I.IGDXWWING: 2.0e4, I.IGDXWSTRUT: 0.0,
    I.IGDXWHTAIL: 1.5e3, I.IGDXWVTAIL: 1.2e3, I.IGS: 124.0, I.IGSH: 31.0,
    I.IGCO: 5.8, I.IGCOH: 3.1, I.IGCMA: 4.2, I.IGSWEEP: 26.0,
    I.IGCMVF1: 60.0, I.IGCLMF0: 0.185, I.IGDCLHDCL: 0.55,
    I.IGDCLNDCL: 0.015, I.IGNENG: 2.0, I.IGDFAN: 1.7, I.IGVH: 1.45,
    I.IGSMMIN: 0.05, I.IGCLHSPEC: -0.02, I.IGCLHCGFWD: -0.65}

AERO = {I.IACMW0: -0.09, I.IACMW1: 0.015, I.IACMH0: -0.02, I.IACMH1: -0.30,
        I.IACL: 0.57, I.IACLH: -0.05}


def _build(**over):
    ac = Aircraft(npoint=1)
    ac.pari[I.IIENGLOC] = 1
    for k, v in GEOM.items():
        ac.parg[k] = v
    for k, v in AERO.items():
        ac.para[k, 1] = v
    for k, v in over.items():
        ac.parg[getattr(I, k)] = v
    return ac


def _case(ic):
    ac = _build()
    para = ac.para.column(1)
    if ic <= 4:
        balance(ac.pari, ac.parg, para, 0.65, 1.0, 0.5, ic - 1)
        lim = cglpay(ac.parg)
        return {"xCG": para[I.IAXCG], "xCP": para[I.IAXCP],
                "xNP": para[I.IAXNP], "CLh": para[I.IACLH],
                "Sh": ac.parg[I.IGSH], "xwbox": ac.parg[I.IGXWBOX],
                "xwing": ac.parg[I.IGXWING], "rpayF": lim.rpayF,
                "xcgF": lim.xcgF, "rpayB": lim.rpayB, "xcgB": lim.xcgB}
    ac.pari[I.IIHTSIZE], ac.pari[I.IIXWMOVE] = (1, 1) if ic == 5 else (2, 2)
    pF, pB, pC = (ac.para.column(1) for _ in range(3))
    pF[I.IACLPMAX] = 1.25
    pC[I.IAFRACW] = 0.90
    htsize(ac.pari, ac.parg, pF, pB, pC)
    return {"Sh": ac.parg[I.IGSH], "xwbox": ac.parg[I.IGXWBOX],
            "xwing": ac.parg[I.IGXWING], "xCGfw": ac.parg[I.IGXCGFWD],
            "xCGaf": ac.parg[I.IGXCGAFT], "CLhfw": ac.parg[I.IGCLHCGFWD]}


def test_matches_fortran():
    got = {ic: _case(ic) for ic in range(1, 7)}
    n = 0
    with (DATA / "balance_ref.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip() == "case":
                continue
            ic, name, ref = int(row[0]), row[1].strip(), float(row[2])
            assert got[ic][name] == pytest.approx(ref, rel=RTOL), \
                f"case {ic} {name}"
            n += 1
    assert n == 56


def test_itrim_zero_changes_nothing():
    ac = _build()
    para = ac.para.column(1)
    balance(ac.pari, ac.parg, para, 0.65, 1.0, 0.5, 0)
    assert para[I.IACLH] == AERO[I.IACLH]
    assert ac.parg[I.IGSH] == GEOM[I.IGSH]
    assert ac.parg[I.IGXWBOX] == GEOM[I.IGXWBOX]
    # Untrimmed, so the centres of pressure and gravity do not coincide.
    assert para[I.IAXCP] != pytest.approx(para[I.IAXCG], rel=1e-6)


@pytest.mark.parametrize("itrim", [1, 3])
def test_trim_makes_cp_and_cg_coincide(itrim):
    """One Newton step is exact -- the residual is linear in each variable."""
    ac = _build()
    para = ac.para.column(1)
    balance(ac.pari, ac.parg, para, 0.65, 1.0, 0.5, itrim)
    assert para[I.IAXCP] == pytest.approx(para[I.IAXCG], rel=1e-12)


def test_trim_by_area_leaves_cp_and_cg_apart():
    """itrim = 2 updates the weight moment for the new tail area but not the
    weight, so xCG = xW_new/W_old and the two centres do not quite meet.

    This is the source's behaviour, reproduced deliberately -- see the note
    in balance().
    """
    ac = _build()
    para = ac.para.column(1)
    balance(ac.pari, ac.parg, para, 0.65, 1.0, 0.5, 2)
    assert para[I.IAXCP] != pytest.approx(para[I.IAXCG], rel=1e-9)
    # The gap is the tail weight change divided through: a few percent here.
    assert para[I.IAXCP] == pytest.approx(para[I.IAXCG], rel=5e-2)


def test_each_trim_mode_moves_only_its_own_variable():
    base = _build()
    for itrim, moved in ((1, "CLh"), (2, "Sh"), (3, "xwbox")):
        ac = _build()
        balance(ac.pari, ac.parg, ac.para.column(1), 0.65, 1.0, 0.5, itrim)
        assert (ac.parg[I.IGSH] != base.parg[I.IGSH]) == (moved == "Sh")
        assert (ac.parg[I.IGXWBOX] != base.parg[I.IGXWBOX]) \
            == (moved == "xwbox")


def test_wing_moves_rigidly_with_its_box():
    """xwing - xwbox is held fixed through a trim change."""
    ac = _build()
    offset = ac.parg[I.IGXWING] - ac.parg[I.IGXWBOX]
    balance(ac.pari, ac.parg, ac.para.column(1), 0.65, 1.0, 0.5, 3)
    assert ac.parg[I.IGXWING] - ac.parg[I.IGXWBOX] == pytest.approx(
        offset, rel=1e-14)


def test_cg_limits_bracket_the_loaded_cg():
    ac = _build()
    lim = cglpay(ac.parg)
    para = ac.para.column(1)
    balance(ac.pari, ac.parg, para, 0.65, 1.0, 0.5, 0)
    assert lim.xcgF < lim.xcgB
    assert 0.0 < lim.rpayF < 1.0
    assert 0.0 < lim.rpayB <= 1.0


def test_cg_limits_assume_zero_fuel():
    """Stated as a bare assumption in the source; both limits use rfuel = 0."""
    ac = _build()
    lim = cglpay(ac.parg)
    assert lim.rfuelF == 0.0 and lim.rfuelB == 0.0
    # Changing the fuel weight therefore cannot move either limit.
    ac2 = _build(IGWFUEL=2.4e5)
    lim2 = cglpay(ac2.parg)
    assert lim2.xcgF == pytest.approx(lim.xcgF, rel=1e-14)
    assert lim2.xcgB == pytest.approx(lim.xcgB, rel=1e-14)


def test_fixed_volume_sizing_delivers_the_volume_coefficient():
    ac = _build()
    ac.pari[I.IIHTSIZE], ac.pari[I.IIXWMOVE] = 1, 1
    pF, pB, pC = (ac.para.column(1) for _ in range(3))
    pF[I.IACLPMAX] = 1.25
    pC[I.IAFRACW] = 0.90
    htsize(ac.pari, ac.parg, pF, pB, pC)
    lhtail = ac.parg[I.IGXHTAIL] - ac.parg[I.IGXWING]
    Vh = ac.parg[I.IGSH] * lhtail / (ac.parg[I.IGS] * ac.parg[I.IGCMA])
    assert Vh == pytest.approx(GEOM[I.IGVH], rel=1e-6)


def _size_for_margin(SMmin):
    ac = _build(IGSMMIN=SMmin)
    ac.pari[I.IIHTSIZE], ac.pari[I.IIXWMOVE] = 2, 2
    pF, pB, pC = (ac.para.column(1) for _ in range(3))
    pF[I.IACLPMAX] = 1.25
    pC[I.IAFRACW] = 0.90
    htsize(ac.pari, ac.parg, pF, pB, pC)
    return ac


def test_stability_margin_sizing_leaves_the_cg_ahead_of_the_np():
    """The converged aircraft is statically stable at its aft CG limit.

    Note the margin is enforced against the CG limits computed on *entry* to
    htsize, before the wing moves, so it does not close exactly against the
    limits recomputed afterwards. That is the source's structure, not a
    convergence failure -- htsize does not re-run cglpay inside its loop.
    """
    ac = _size_for_margin(0.05)
    para = ac.para.column(1)
    lim = cglpay(ac.parg)
    balance(ac.pari, ac.parg, para, 0.0, lim.rpayB, 1.0, 0)
    assert para[I.IAXNP] > para[I.IAXCG]


def test_demanding_more_margin_moves_the_wing_aft():
    """More margin means the CG must sit further ahead of the neutral point,
    which the solver buys by moving the wing (and its CG) back."""
    a = _size_for_margin(0.02)
    b = _size_for_margin(0.10)
    assert b.parg[I.IGXWBOX] > a.parg[I.IGXWBOX]


def test_engine_inlet_moves_the_neutral_point_forward():
    """The inlet normal force is destabilising; zeroing it moves xNP aft."""
    ac = _build()
    para = ac.para.column(1)
    balance(ac.pari, ac.parg, para, 0.65, 1.0, 0.5, 0)
    with_eng = para[I.IAXNP]

    ac0 = _build(IGDCLNDCL=0.0)
    para0 = ac0.para.column(1)
    balance(ac0.pari, ac0.parg, para0, 0.65, 1.0, 0.5, 0)
    assert para0[I.IAXNP] > with_eng

    # And the size of the shift is neng * xengcp * dCLndCL * Afan / S.
    Afan = 0.25 * math.pi * GEOM[I.IGDFAN] ** 2
    xengcp = GEOM[I.IGXENG] - 0.25 * GEOM[I.IGDFAN]
    shift = (GEOM[I.IGNENG] * xengcp * GEOM[I.IGDCLNDCL] * Afan
             / GEOM[I.IGS])
    assert para0[I.IAXNP] - with_eng == pytest.approx(shift, rel=1e-12)
