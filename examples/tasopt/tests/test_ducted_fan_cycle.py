"""Ducted fan gas path -- against TASOPT.jl."""
import math

import pytest

from tasopt_py.atmosphere import atmos
from tasopt_py.engine_v3.ducted_fan_cycle import (ducted_fan_size,
                                                  ducted_fan_operate,
                                                  V3_CMAPF)

TREF, PREF = 288.2, 101320.0
GEE = 9.81


def _design():
    a = atmos(11.0)
    r = ducted_fan_size(GEE, 0.8, a.T, a.p, a.a, 0.6, 1e4, 0, 0, 0,
                        1.5, 1.0, 1.0, 0.9)
    mbfD = r.mfan * math.sqrt(r.Tt[2] / TREF) / (r.pt[2] / PREF)
    NbfD = 1.0 / math.sqrt(r.Tt[2] / TREF)
    return a, r, mbfD, NbfD


SIZE_REF = (
    327.11210797234, 0.4556101382778294, 3.2711210797234005e6,
    92.71080094936762,
    245.96522147425617, -84555.91953868531, 34660.08023796853,
    1004.6979001778267, 287.482334,
    245.96522147425617, -84555.91953868531, 34660.08023796853,
    1004.6979001778267, 287.482334,
    245.96522147425617, -84555.91953868531, 34660.08023796853,
    1004.6979001778267, 287.482334,
    281.0643310206969, -49272.85600965954, 51990.120356952786,
    1005.7936500187604, 287.482334,
    281.0643310206969, -49272.85600965954, 51990.120356952786,
    1005.7936500187604, 287.482334,
    236.74253162629373,
    229.40455573526395, 182.38072225868564, 27167.2752649851,
    1004.3726660891869, 287.482334, 1.234009545340459,
    234.14733320991527, 307.08902200062846, 27455.81621405862,
    1004.4531614909928, 287.482334, 0.7401709919109498,
    221.89872511133328, 344.82581602886535, 22756.739147503034,
    1004.2637990691651, 287.482334, 0.7536791452295546,
    0.8693488619036531, 0.8616487982671662)


def test_ductedfansize_matches_tasopt_jl():
    _, r, _, _ = _design()
    got = (r.TSEC, r.Fsp, r.Pfan, r.mfan)
    for st in (0, 18, 2, 21, 7):
        got += (r.Tt[st], r.ht[st], r.pt[st], r.cpt[st], r.Rt[st])
    got += (r.u0,)
    for st in (2, 7, 8):
        got += (r.T[st], r.u[st], r.p[st], r.cp[st], r.R[st], r.A[st])
    got += (r.epf, r.etaf)
    assert len(got) == len(SIZE_REF)
    for a, b in zip(got, SIZE_REF):
        assert a == pytest.approx(b, rel=1e-13, abs=1e-13)


def test_operating_at_the_design_point_returns_the_design():
    """The strongest self-consistency check the two routines admit."""
    a, r, mbfD, NbfD = _design()
    o = ducted_fan_operate(0.8, a.T, a.p, a.a, TREF, PREF, 0, 0, 0,
                           1.0, 1.0, 1.5, mbfD, NbfD, r.A[2], r.A[7], 0.9,
                           Feng=1e4, M2=0.6, pif=1.5)
    assert o.Feng == pytest.approx(1e4, rel=1e-12)
    assert o.Pfan == pytest.approx(r.Pfan, rel=1e-12)
    assert o.mfan == pytest.approx(r.mfan, rel=1e-12)
    assert o.pif == pytest.approx(1.5, rel=1e-12)


OPER_REF = (
    140.74270917523936, 0.0, 56841.31026666052, 8.000000000000149e6,
    225.39505852577363, 1.4330295431713083, 225.39505852577363,
    1.0012505100170295,
    288.2, -42093.61205685147, 101320.0, 1006.0696710363969, 287.482334,
    288.2, -42093.61205685147, 101320.0, 1006.0696710363969, 287.482334,
    288.2, -42093.61205685147, 101320.0, 1006.0696710363969, 287.482334,
    323.4458621045641, -6600.376081204027, 145194.55331411696,
    1007.9218437076298, 287.482334,
    323.4458621045641, -6600.376081204027, 145194.55331411696,
    1007.9218437076298, 287.482334,
    0.0,
    273.9002717202924, 169.61337292744685, 84794.83629183592,
    1005.535918910043, 287.482334, 0.5107847532243659,
    291.8712909624618, 252.18525480744196, 101320.0, 1006.2219870034098,
    287.482334, 0.7357958274607429,
    291.8712909624618, 252.18525480744196, 101320.0, 1006.2219870034098,
    287.482334, 0.7357958274607429, 0.7401709919114717,
    0.8903649191567354, 0.8846551852455233)


def test_ductedfanoper_at_takeoff_matches_tasopt_jl():
    """Sea-level static on specified shaft power."""
    _, r, mbfD, NbfD = _design()
    a = atmos(0.0)
    o = ducted_fan_operate(0.0, a.T, a.p, a.a, TREF, PREF, 0, 0, 0,
                           1.0, 1.0, 1.5, mbfD, NbfD, r.A[2], r.A[7], 0.9,
                           Feng=0, Peng=8e6, M2=0.6, pif=1.5, iPspec=True)
    got = (o.TSEC, o.Fsp, o.Feng, o.Pfan, o.mfan, o.pif, o.mbf, o.Nbf)
    for st in (0, 18, 2, 21, 7):
        got += (o.Tt[st], o.ht[st], o.pt[st], o.cpt[st], o.Rt[st])
    got += (o.u0,)
    for st in (2, 7):
        got += (o.T[st], o.u[st], o.p[st], o.cp[st], o.R[st], o.M[st])
    got += (o.T[8], o.u[8], o.p[8], o.cp[8], o.R[8], o.M[8], o.A[8],
            o.epf, o.etaf)
    assert len(got) == len(OPER_REF)
    for a_, b in zip(got, OPER_REF):
        assert a_ == pytest.approx(b, rel=1e-11, abs=1e-11)


def test_sea_level_static_makes_corrected_and_actual_flow_equal():
    """A check the reference's own numbers happen to contain.

    At M0 = 0 on a standard sea-level day with a lossless diffuser, the fan
    face sits exactly at the reference conditions, so the corrected mass
    flow and the actual one must be the same number. They are: 225.395 both.
    """
    _, r, mbfD, NbfD = _design()
    a = atmos(0.0)
    o = ducted_fan_operate(0.0, a.T, a.p, a.a, TREF, PREF, 0, 0, 0,
                           1.0, 1.0, 1.5, mbfD, NbfD, r.A[2], r.A[7], 0.9,
                           Feng=0, Peng=8e6, M2=0.6, pif=1.5, iPspec=True)
    assert o.Tt[2] == pytest.approx(TREF, rel=1e-12)
    assert o.pt[2] == pytest.approx(PREF, rel=1e-12)
    assert o.mfan == pytest.approx(o.mbf, rel=1e-14)


def test_thrust_and_power_specification_agree():
    """Ask for a thrust, then ask for the power it needed. Same point.

    This is what makes an electrically driven fan usable from either end,
    and it is the only difference between the two residual sets.
    """
    _, r, mbfD, NbfD = _design()
    a = atmos(0.0)
    by_power = ducted_fan_operate(0.0, a.T, a.p, a.a, TREF, PREF, 0, 0, 0,
                                  1.0, 1.0, 1.5, mbfD, NbfD, r.A[2], r.A[7],
                                  0.9, Peng=8e6, M2=0.6, pif=1.5,
                                  iPspec=True)
    by_thrust = ducted_fan_operate(0.0, a.T, a.p, a.a, TREF, PREF, 0, 0, 0,
                                   1.0, 1.0, 1.5, mbfD, NbfD, r.A[2],
                                   r.A[7], 0.9, Feng=by_power.Feng, M2=0.6,
                                   pif=1.5)
    assert by_thrust.Pfan == pytest.approx(8e6, rel=1e-9)
    assert by_thrust.pif == pytest.approx(by_power.pif, rel=1e-9)
    assert by_thrust.mfan == pytest.approx(by_power.mfan, rel=1e-9)


def test_the_speed_map_constants_are_v3s_not_216s():
    """v3 refitted Cmapf. The efficiency comes from the pyCycle table but
    the *speed* still comes from 2.16's analytic Ncmap -- with a different
    constant set than 2.16 ships. Two fan models in one routine.
    """
    from tasopt_py.engine.maps import CMAPF
    assert V3_CMAPF != CMAPF
    assert V3_CMAPF[0] == pytest.approx(3.31140687)
    assert CMAPF[0] == pytest.approx(3.50)


def test_more_power_gives_more_thrust_but_less_efficiently():
    """Propulsive efficiency falls as the jet speeds up."""
    _, r, mbfD, NbfD = _design()
    a = atmos(0.0)
    out = []
    for P in (4e6, 8e6, 1.2e7):
        o = ducted_fan_operate(0.0, a.T, a.p, a.a, TREF, PREF, 0, 0, 0,
                               1.0, 1.0, 1.5, mbfD, NbfD, r.A[2], r.A[7],
                               0.9, Peng=P, M2=0.6, pif=1.5, iPspec=True)
        out.append((o.Feng, o.Feng / P))
    thrusts = [t for t, _ in out]
    per_watt = [e for _, e in out]
    assert thrusts == sorted(thrusts)
    assert per_watt == sorted(per_watt, reverse=True)
