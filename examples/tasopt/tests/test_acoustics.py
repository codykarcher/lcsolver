"""The acoustic model (tfnoise.f) against the real TASOPT 737 run.

The three certification decibel values are the last numbers in ``737.out``,
and with this module ported the port's report is **byte-identical to the
reference program's, all 4565 lines** -- see ``tests/test_output.py``. This
file pins the three numbers individually, and covers the correlations and
edge cases underneath them.

The reference values, from ``737.out``:

===========  ========
sideline     88.912 dB
cutback      73.714 dB
flyover      73.959 dB
===========  ========
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from tasopt_py.acoustics import (BZdirfun, BBdirfun, DTdirfun,
                                 GROUND_REFLECTION_DB, TONE_FLOOR_DB,
                                 convert_tones_to_third_octave, dBAcorrf,
                                 dBattenf, esdu98008combinationtone_total,
                                 esdu98008discretetone_total, fpfun, fsfun,
                                 tfnoise, which_third_octave)
from tasopt_py.acoustics_tables import DBACORR, DBATTEN, FMID, FREQ, NFREQ
from tasopt_py.model import indices as I
from tasopt_py.run import run_case
from tasopt_py.sizing.noise import FAN

TAS = Path("/Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.tas")
OUT = Path("/Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.out")

pytestmark_case = pytest.mark.skipif(
    not TAS.exists() or not OUT.exists(), reason="737 case not present")


@pytest.fixture(scope="module")
def flown():
    return run_case(TAS, off_design=False)


@pytest.fixture(scope="module")
def reference_dB():
    lines = OUT.read_text().split("\n")
    i = lines.index("    x [m]   z [m]     dB")
    return {p[3]: float(p[2]) for p in
            (lines[i + k].split() for k in (1, 2, 3))}


@pytestmark_case
def test_the_three_certification_levels(flown, reference_dB):
    parm = flown.case.missions[0].parm
    assert parm[I.IMDBSL] == pytest.approx(reference_dB["sideline"],
                                           abs=0.0005)
    assert parm[I.IMDBCB] == pytest.approx(reference_dB["cutback"],
                                           abs=0.0005)
    assert parm[I.IMDBFO] == pytest.approx(reference_dB["flyover"],
                                           abs=0.0005)


@pytestmark_case
def test_the_sideline_is_the_loudest_point(flown):
    """It is the closest -- 450 m abeam at low altitude, at full power."""
    parm = flown.case.missions[0].parm
    assert parm[I.IMDBSL] > parm[I.IMDBCB]
    assert parm[I.IMDBSL] > parm[I.IMDBFO]


def test_the_angle_of_attack_is_passed_in_two_different_units():
    """noise.f writes ``alpha = 5.0`` for the sideline and
    ``alpha = 5.0*pi/180`` for the other two points, so the sideline
    calculation gets 5 radians -- 286 degrees -- where the others get 5
    degrees. It reaches the jet correlation as ``cos(alpha)``, which is 0.28
    instead of 0.996, so the sideline jet is computed with a much weaker
    flight-velocity correction. Reproducing it is what makes the sideline
    value come out right."""
    assert FAN.alpha_sideline == 5.0
    assert FAN.alpha_other == pytest.approx(5.0 * math.pi / 180.0)
    assert math.cos(FAN.alpha_sideline) == pytest.approx(0.2837, abs=1e-4)
    assert math.cos(FAN.alpha_other) == pytest.approx(0.9962, abs=1e-4)


# --- the correlations underneath -------------------------------------------

def test_fpfun_is_always_evaluated_at_110_degrees():
    """It clamps its angle to [110, 250] degrees, and its only caller passes
    radians -- never more than about 3.5 -- so the clamp always fires. The
    jet spectrum shape is frozen at the 110-degree column."""
    for theta_rad in (0.0, 0.5, 1.5, 3.14, 3.5):
        assert fpfun(1.0, theta_rad) == fpfun(1.0, 110.0)
    # It does vary with angle if it is actually given degrees.
    assert fpfun(1.0, 200.0) != fpfun(1.0, 110.0)


def test_fpfun_peaks_near_unit_strouhal_number():
    peak = max((fpfun(10 ** e, 110.0), e) for e in
               [x / 10.0 for x in range(-30, 31)])
    assert -1.0 < peak[1] < 1.0
    assert fpfun(1e-4, 110.0) < peak[0] and fpfun(1e4, 110.0) < peak[0]


def test_fsfun_vanishes_with_no_secondary_stream():
    assert fsfun(0.0, 0.5) == pytest.approx(0.0)
    assert 0.0 < fsfun(5.0, 0.64) < 1.0


def test_the_weighting_tables_are_the_formulae_rounded():
    """freq.inc tabulates what dBAcorrf and dBattenf compute; the model uses
    the tables, so both are kept and this pins them together."""
    for f, w, a in zip(FREQ, DBACORR, DBATTEN):
        assert w == pytest.approx(dBAcorrf(f), abs=1e-3)
        assert a == pytest.approx(dBattenf(f), abs=0.02)


def test_a_weighting_peaks_around_2_to_3_kHz():
    best = max(range(NFREQ), key=lambda i: DBACORR[i])
    assert 2000.0 <= FREQ[best] <= 3200.0
    assert DBACORR[0] < -25.0            # 50 Hz is heavily discounted


def test_third_octave_band_assignment():
    assert which_third_octave(FREQ[0]) == 1
    assert which_third_octave(FREQ[-1]) == NFREQ
    assert which_third_octave(FMID[0] - 1.0) == 0        # below the range
    assert which_third_octave(1.0e6) == 0                # above it
    for i, f in enumerate(FREQ, start=1):
        assert which_third_octave(f) == i


def test_tones_add_energetically_within_a_band():
    """Two equal tones in one band make it 3 dB louder; the first tone
    replaces the -40 dB floor rather than adding to it."""
    out = convert_tones_to_third_octave([FREQ[5]], [80.0])
    assert out[5] == 80.0
    assert all(v == TONE_FLOOR_DB for i, v in enumerate(out) if i != 5)

    out = convert_tones_to_third_octave([FREQ[5], FREQ[5]], [80.0, 80.0])
    assert out[5] == pytest.approx(83.0103, abs=1e-3)


def test_tones_outside_the_band_range_are_dropped():
    out = convert_tones_to_third_octave([1.0e6, 1.0], [90.0, 90.0])
    assert all(v == TONE_FLOOR_DB for v in out)


def test_the_directivity_functions_are_bounded_and_distinct():
    for theta in (0.0, 0.5, 1.0, 2.0, math.pi):
        for method in (1, 2, 3):
            for direc in (+1, -1):
                assert -80.0 < BBdirfun(theta, direc, method) < 80.0
                assert -80.0 < DTdirfun(theta, direc, method) < 80.0
        assert -80.0 < BZdirfun(theta) < 80.0
    # Forward and rearward really are different correlations.
    assert BBdirfun(1.0, +1, 2) != BBdirfun(1.0, -1, 2)
    assert DTdirfun(1.0, +1, 2) != DTdirfun(1.0, -1, 2)
    # Method 2 differs from methods 1 and 3, which share coefficients.
    assert DTdirfun(1.0, +1, 1) == DTdirfun(1.0, +1, 3)
    assert DTdirfun(1.0, +1, 1) != DTdirfun(1.0, +1, 2)


def test_the_tone_series_runs_one_harmonic_past_ten_kilohertz():
    """The next harmonic is appended before the limit is tested, so the
    series always ends above 10 kHz. Whether that one contributes depends on
    where it lands: the top third-octave band runs to 11220 Hz, so a harmonic
    between 10 and 11.2 kHz still counts, and one above 11.2 kHz is dropped.
    """
    f, spl = esdu98008discretetone_total(
        +1, 2, Dist=500.0, theta=1.0, dTt=60.0, mfan=200.0, Mtr=1.38,
        Mtrd=1.30, Mt=1.2, RPM=4700.0, rss=300.0, B=20.0, V=44.0,
        bpf=1566.0)
    assert len(f) == len(spl)
    assert f[-1] > 10000.0
    assert which_third_octave(f[-1]) == NFREQ       # inside the top band
    # A higher blade-passing frequency puts it past the top edge instead.
    f2, _ = esdu98008discretetone_total(
        +1, 2, Dist=500.0, theta=1.0, dTt=60.0, mfan=200.0, Mtr=1.38,
        Mtrd=1.30, Mt=1.2, RPM=4700.0, rss=300.0, B=20.0, V=44.0,
        bpf=6000.0)
    assert f2[-1] > FMID[-1] and which_third_octave(f2[-1]) == 0
    # Levels fall off with harmonic number.
    assert spl[0] > spl[-1]


def test_the_buzzsaw_only_has_three_subharmonics():
    f, spl = esdu98008combinationtone_total(
        Dist=500.0, theta=1.0, shocklocation=+1, rss=300.0, dTt=60.0,
        mfan=200.0, Mtr=1.38, bpf=1600.0)
    assert f == [800.0, 400.0, 200.0]
    assert len(spl) == 3


def test_an_ingested_shock_takes_six_decibels_off_the_buzzsaw():
    kw = dict(Dist=500.0, theta=1.0, rss=300.0, dTt=60.0, mfan=200.0,
              Mtr=1.38, bpf=1600.0)
    _, expelled = esdu98008combinationtone_total(shocklocation=+1, **kw)
    _, ingested = esdu98008combinationtone_total(shocklocation=-1, **kw)
    for a, b in zip(expelled, ingested):
        assert a - b == pytest.approx(6.0)


def test_an_unrecognised_shock_location_is_refused():
    with pytest.raises(ValueError, match="shock location"):
        esdu98008combinationtone_total(
            Dist=500.0, theta=1.0, shocklocation=0, rss=300.0, dTt=60.0,
            mfan=200.0, Mtr=1.38, bpf=1600.0)


def test_more_engines_are_louder_by_ten_log_n():
    kw = dict(x=0.0, y=450.0, z=0.0, climb=0.17, alpha=5.0, vector=0.0,
              rho0=1.2, p0=101325.0, T0=288.0, mu0=1.8e-5, c0=340.0,
              A6=0.23, A8=0.74, u6=570.0, u8=316.0, T6=821.0, T8=295.0,
              M0=0.22, etaf=0.885, FPR=1.70, mdot=54.7, BPR=5.11,
              Mtrd=1.30, Mtr=1.38, Mt=1.2, RPM=4700.0, rss=300.0,
              B=20.0, V=44.0, htr=0.29, type_=0, method=2)
    one = tfnoise(neng=1, **kw)
    two = tfnoise(neng=2, **kw)
    assert two.total - one.total == pytest.approx(10.0 * math.log10(2.0),
                                                  abs=1e-9)
    # Every component scales the same way.
    for name in ("jet", "fan_tone_fwd", "fan_broadband_fwd",
                 "fan_tone_rear", "fan_broadband_rear"):
        assert (getattr(two, name) - getattr(one, name)
                == pytest.approx(10.0 * math.log10(2.0), abs=1e-9))


def test_the_total_is_an_energy_sum_of_the_components():
    kw = dict(x=100.0, y=450.0, z=300.0, climb=0.17, alpha=0.087,
              vector=0.0, rho0=1.2, p0=101325.0, T0=288.0, mu0=1.8e-5,
              c0=340.0, A6=0.23, A8=0.74, u6=570.0, u8=316.0, T6=821.0,
              T8=295.0, M0=0.22, etaf=0.885, FPR=1.70, mdot=54.7,
              BPR=5.11, Mtrd=1.30, Mtr=1.38, Mt=1.2, RPM=4700.0,
              rss=300.0, B=20.0, V=44.0, htr=0.29, neng=2, type_=0,
              method=2)
    r = tfnoise(**kw)
    parts = (r.jet, r.fan_tone_fwd, r.fan_broadband_fwd, r.fan_buzzsaw,
             r.fan_tone_rear, r.fan_broadband_rear)
    assert r.total == pytest.approx(
        10.0 * math.log10(sum(10.0 ** (v / 10.0) for v in parts)))
    assert r.total > max(parts)          # the sum beats any one component


def test_the_ground_reflection_is_a_flat_three_decibels():
    assert GROUND_REFLECTION_DB == 3.0
