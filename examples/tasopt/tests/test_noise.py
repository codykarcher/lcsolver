"""Certification-point noise (noise.f) against the real TASOPT 737 run.

``noise.f`` does two separable things, and only one needs an acoustic model.
It sets the **takeoff and cutback engine operating points** -- the only
routine in the program that runs the engine there -- and it works out the
**three certification observer positions**. Both are checked here against the
reference program's ``737.out``, which reports all of them.

The decibels themselves come from ``tfnoise.f``, ported in
:mod:`tasopt_py.acoustics` and checked in ``tests/test_acoustics.py``. What
this file shows is that everything *around* the acoustic model is right: the
engine states at the two points, their flight-path angles, and where the
observers are -- and that the model is injected, so leaving it out leaves the
decibels alone rather than defaulting them.

The full byte-for-byte comparison is in ``tests/test_output.py``; this file
pins the individual numbers so a failure says which one moved.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from tasopt_py.model import indices as I
from tasopt_py.run import run_case
from tasopt_py.sizing.noise import (EXHAUST_UNMIXED, FAN, L_OBSERVER,
                                    METHOD_HEIDMANN, SIDELINE_Y)

TAS = Path("/Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.tas")
OUT = Path("/Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.out")

pytestmark = pytest.mark.skipif(
    not TAS.exists() or not OUT.exists(), reason="737 case not present")


@pytest.fixture(scope="module")
def flown():
    return run_case(TAS, off_design=False)


@pytest.fixture(scope="module")
def reference_noise_rows():
    """The three ``x  z  dB  label`` rows the reference prints."""
    lines = OUT.read_text().split("\n")
    i = lines.index("    x [m]   z [m]     dB")
    rows = {}
    for k in (1, 2, 3):
        parts = lines[i + k].split()
        rows[parts[3]] = (float(parts[0]), float(parts[1]), float(parts[2]))
    return rows


def test_observer_positions_match(flown, reference_noise_rows):
    parm = flown.case.missions[0].parm
    x_sl, z_sl, _ = reference_noise_rows["sideline"]
    x_cb, z_cb, _ = reference_noise_rows["cutback"]
    x_fo, z_fo, _ = reference_noise_rows["flyover"]

    # The sideline x is 6500 m minus the takeoff run, printed by tofwrt.
    assert L_OBSERVER - parm[I.IMLTO] == pytest.approx(x_sl, abs=0.05)
    assert parm[I.IMXCB] == pytest.approx(x_cb, abs=0.05)
    assert parm[I.IMZCB] == pytest.approx(z_cb, abs=0.05)
    assert parm[I.IMXFO] == pytest.approx(x_fo, abs=0.05)
    assert parm[I.IMZFO] == pytest.approx(z_fo, abs=0.05)


def test_the_takeoff_and_cutback_engine_points_are_run(flown):
    """Neither wsize nor mission touches these two points; without noise.f
    they stay at the 'unset' fill value and the report prints garbage."""
    pare = flown.case.missions[0].pare
    for ip in (I.IPTAKEOFF, I.IPCUTBACK):
        for idx in (I.IETT4, I.IETT3, I.IEFE, I.IEMCORE, I.IETSFC,
                    I.IEU6, I.IEU8, I.IEA6, I.IEA8, I.IEETAF):
            v = pare[idx, ip]
            assert 0.0 <= abs(v) < 1e30, (ip, idx)


def test_cutback_is_throttled_back_from_takeoff(flown):
    """The cutback point is set to the thrust that holds the prescribed climb
    angle, which on the 737 is a good deal less than takeoff thrust."""
    pare, para = flown.case.missions[0].pare, flown.case.missions[0].para
    Fto = pare[I.IEFE, I.IPTAKEOFF]
    Fcb = pare[I.IEFE, I.IPCUTBACK]
    assert 0.3 < Fcb / Fto < 0.8
    assert pare[I.IETT4, I.IPCUTBACK] < pare[I.IETT4, I.IPTAKEOFF]
    # And its flight-path angle comes back as the one that was asked for.
    assert para[I.IAGAMV, I.IPCUTBACK] == pytest.approx(
        flown.case.missions[0].parm[I.IMGAMVCB], rel=1e-6)


def test_takeoff_climb_angle_matches_the_report(flown):
    """geowrt prints tan(gam)_TO as a percentage."""
    text = OUT.read_text()
    want = float(re.search(r"tan\(gam\)_TO =\s*([-\d.]+) %", text).group(1))
    parm = flown.case.missions[0].parm
    assert math.tan(parm[I.IMGAMVTO]) * 100.0 == pytest.approx(want,
                                                               abs=0.005)


def test_the_decibels_are_computed(flown):
    """tasopt_py.acoustics is wired in by default, so the three
    certification levels are real -- see tests/test_acoustics.py."""
    parm = flown.case.missions[0].parm
    for idx in (I.IMDBSL, I.IMDBCB, I.IMDBFO):
        assert 50.0 < parm[idx] < 120.0


def test_without_an_acoustic_model_the_decibels_are_left_alone(flown):
    """The model is injected, and omitting it leaves the three parm entries
    exactly as they were rather than defaulting them to something -- so a
    missing acoustic model reads as missing, not as silence."""
    from tasopt_py.sizing.noise import noise

    case = flown.case
    m = case.missions[0]
    before = [m.parm[idx] for idx in (I.IMDBSL, I.IMDBCB, I.IMDBFO)]
    g = noise(case.pari, case.parg, m.parm, m.para, m.pare, initeng=1,
              table=None, tfnoise=None)
    assert [m.parm[idx] for idx in (I.IMDBSL, I.IMDBCB, I.IMDBFO)] == before
    assert (g.dBSL, g.dBCB, g.dBFO) == (None, None, None)
    # ...but the geometry is still worked out.
    assert g.xCB > 0.0 and g.zCB > 0.0


def test_the_hard_wired_fan_geometry():
    """All of it is fixed in the source, from the SAX-40 fan design."""
    assert (FAN.Mt, FAN.Mtr, FAN.Mtrd) == (1.2, 1.38, 1.30)
    assert (FAN.B, FAN.V) == (20.0, 44.0)
    assert SIDELINE_Y == 450.0 and L_OBSERVER == 6500.0
    # Unmixed exhaust and Heidmann's method are the live settings; mixed
    # exhaust and the ESDU and Allied Signal variants are commented out.
    assert (EXHAUST_UNMIXED, METHOD_HEIDMANN) == (0, 2)
