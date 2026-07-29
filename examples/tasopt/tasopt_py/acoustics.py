"""The fan-and-jet acoustic model -- ``tfnoise.f``.

Estimates the A-weighted sound pressure level at an observer, as the sum of
six contributions:

======================  ====================================================
jet mixing              Stone's correlation (AIAA J. 21(3)), with Low's 1994
                        flight corrections and a coaxial-jet modification
fan tones, forward      ESDU 98008 discrete tone, harmonics of blade passing
fan broadband, forward  ESDU 98008 broadband
fan buzzsaw             ESDU 98008 combination tone, only above ``Mtr = 1``
fan tones, rearward     as forward, different directivity and level
fan broadband, rearward as forward
======================  ====================================================

Each is built as a 24-band third-octave spectrum from 50 Hz to 10 kHz,
A-weighted, given a fixed atmospheric attenuation and a flat +3 dB for ground
reflection, then integrated. The source is explicit that lateral attenuation,
Doppler and shielding are all left out, and that it "is for noise estimates
only, and is not intended to replace ANOPP".

``method`` selects between ESDU (1), Heidmann as in ANOPP (2, the live
setting) and Heidmann with Allied Signal's small-engine modifications (3).
The source notes there is "no significant difference between the methods,
around 1 dBA" on a 737-300/CFM56 and a 777-200/GE90.

Things in the source worth knowing
----------------------------------
**``fpfun`` is always evaluated at 110 degrees.** It takes the corrected
directivity angle in *degrees* -- its own comment says so, and it clamps the
argument to ``[110, 250]``. ``jet_noise`` passes ``thetap``, which is in
*radians* and never exceeds about 3.5. So the clamp always returns 110.0 and
the jet spectral shape is frozen at the 110-degree column of the table it
represents, whatever direction the observer is in. The overall level still
varies with angle -- that is ``UOL``, computed separately -- so this affects
the shape of the spectrum, not the total by very much. Reproduced, because
reproducing it is the point; :func:`fpfun` takes degrees, and
:func:`jet_noise` hands it radians, exactly as the Fortran does.

**A commented-out convective Mach number, flagged by its author.** The line

    ``cc    Mc = 0.62*(u8-u0*cos(alpha))/c0   !%% BUG???``

sits above the live ``Mc = 0.62*(u6-u0*cos(alpha))/c0``. Somebody suspected
the fan jet should drive the convection rather than the core jet and left the
question in. The core-jet form is what ships.

**The buzzsaw floor is -40 dB, not zero.** Bands with no tone in them are set
to -40 dB rather than to silence, which contributes 1e-4 of a unit to the
energy sum -- negligible, but it means the buzzsaw term is never exactly zero
even below the supersonic threshold.

**The tone series stops one harmonic late.** ``esdu98008discretetone_total``
appends the next harmonic *before* testing whether the current one is past
10 kHz, so the returned series always ends above the limit. Whether that last
one counts is then an accident of where it lands: the top third-octave band
runs to 11220 Hz, so a harmonic between 10 and 11.2 kHz is still added in and
one above 11.2 kHz is dropped. Reproduced, so the band assignment sees the
same input.

Verified against the compiled Fortran; see ``tests/test_acoustics.py``.
"""
from __future__ import annotations

import math

from .acoustics_tables import DBACORR, DBATTEN, FMID, FREQ, NFREQ

__all__ = ["tfnoise", "jet_noise", "esdu_fan_noise_total", "fpfun", "fsfun",
           "dBAcorrf", "dBattenf", "BBdirfun", "DTdirfun", "BZdirfun",
           "which_third_octave", "convert_tones_to_third_octave",
           "NoiseBreakdown", "GROUND_REFLECTION_DB", "TONE_FLOOR_DB"]

#: Flat allowance for reflection off the ground, added to every band.
GROUND_REFLECTION_DB = 3.0
#: What an empty third-octave band is set to, rather than silence.
TONE_FLOOR_DB = -40.0

R_AIR = 287.0
GAMMA = 1.4
T_ISA = 288.0
P_ISA = 101325.0


class NoiseBreakdown:
    """The six components and their total, all A-weighted dB."""

    __slots__ = ("jet", "fan_tone_fwd", "fan_broadband_fwd", "fan_buzzsaw",
                 "fan_tone_rear", "fan_broadband_rear", "total")

    def __init__(self, jet, f_t, f_b, f_z, r_t, r_b, total):
        self.jet = jet
        self.fan_tone_fwd = f_t
        self.fan_broadband_fwd = f_b
        self.fan_buzzsaw = f_z
        self.fan_tone_rear = r_t
        self.fan_broadband_rear = r_b
        self.total = total

    def __repr__(self):
        return (f"NoiseBreakdown(total={self.total:.3f}, "
                f"jet={self.jet:.3f}, fan_tone_fwd={self.fan_tone_fwd:.3f}, "
                f"fan_broadband_fwd={self.fan_broadband_fwd:.3f}, "
                f"fan_buzzsaw={self.fan_buzzsaw:.3f}, "
                f"fan_tone_rear={self.fan_tone_rear:.3f}, "
                f"fan_broadband_rear={self.fan_broadband_rear:.3f})")


# --------------------------------------------------------------------------
# Small correlations
# --------------------------------------------------------------------------

def fsfun(Arat: float, Vrat: float) -> float:
    """Coaxial-jet frequency-shift parameter. AIAA J. 21(3) p.337, fig. 2."""
    Apar = math.log10(1.0 + Arat)
    w = 1.0 - abs(Vrat - 0.64) ** 1.1
    return 1.0 - math.exp(-1.28 * Apar * w)


def fpfun(S: float, t: float) -> float:
    """Jet-mixing spectrum shape, ``SPL - UOL`` in dB. AIAA J. 21(3) table 1.

    ``t`` is the corrected directivity angle **in degrees**, clamped to
    ``[110, 250]``. Its only caller passes radians, so in practice this is
    always evaluated at 110 -- see the module docstring.
    """
    Sl = max(-3.6, min(3.6, math.log10(S)))
    tl = max(110.0, min(250.0, t))

    Slo = -0.3 - 0.012 * (tl - 110.0) + 0.00003 * (tl - 110.0) ** 2
    ap = 12.0 + 0.00015 * (tl - 110) ** 3 / (1 + 0.0000115 * (tl - 110) ** 3)
    am = -35.0 + 0.007 * (tl - 157) ** 2 / (1.0 + 0.00022 * (tl - 157) ** 2)
    fo = -2.5 - 0.0022 * (tl - 120) ** 2 / (1 + 0.000027 * (tl - 120) ** 2)
    w = 0.14 - 0.07 / (1 + 0.00005 * (tl - 110) ** 3)

    return (-math.log(math.exp(w * am * (Sl - Slo))
                      + math.exp(w * ap * (Sl - Slo))) / w) + fo


def dBAcorrf(f: float) -> float:
    """A-weighting in dB at frequency ``f``. A simpler fit is commented out
    beside it in the source."""
    den1 = f ** 2 + 20.6 ** 2
    den2 = math.sqrt((f ** 2 + 107.0 ** 2) * (f ** 2 + 737.9 ** 2))
    den3 = f ** 2 + 12200.0 ** 2
    RA = 12200.0 ** 2 * f ** 4 / (den1 * den2 * den3)
    return 2.0 + 20.0 * math.log10(RA)


def dBattenf(f: float) -> float:
    """Atmospheric attenuation, dB per 100 m (negative)."""
    den2 = 1.0 + 0.086 * (f / 1000.0) ** 2
    den1 = 1.0 + 1.46 * (f / 1000.0) ** 2 / den2
    return -1.53 * (f / 1000.0) ** 2 / den1


def _dirfun(a, theta, nterm):
    return sum(a[k] * math.cos(k * theta) for k in range(nterm))


#: Broadband directivity coefficients: forward, then rearward for methods
#: 1/3 and for method 2.
_BB = ((-21.319, 27.786, -7.272, 1.155, -1.604, 0.371, -0.794),
       (-11.600, -8.581, -9.443, 1.443, -2.134, 0.793, -0.751),
       (-13.761, -12.442, -10.418, 1.761, -2.442, 0.487, -0.943))

#: Discrete-tone directivity: forward 1/3, forward 2, rearward 1/3, rearward 2.
_DT = ((-13.875, 17.112, -5.258, 0.501, -0.748, 0.574, -0.457),
       (-20.917, 27.563, -7.531, 0.935, -1.696, 0.416, -0.698),
       (-9.917, -10.048, -9.458, 0.515, -1.809, 0.519, -1.034),
       (-13.056, -12.236, -9.685, 1.456, -2.258, 0.388, -0.625))

_BZ = (-5.553, 5.396, -2.236, -1.313, -1.032)


def BBdirfun(theta: float, direc: int, method: int) -> float:
    """Broadband directivity. ``direc`` +1 forward, -1 rearward."""
    if direc == 1:
        k = 0
    else:
        k = 1 if method in (1, 3) else 2
    return _dirfun(_BB[k], theta, 7)


def DTdirfun(theta: float, direc: int, method: int) -> float:
    """Discrete-tone directivity."""
    if direc == 1:
        k = 0 if method in (1, 3) else 1
    else:
        k = 2 if method in (1, 3) else 3
    return _dirfun(_DT[k], theta, 7)


def BZdirfun(theta: float) -> float:
    """Buzzsaw directivity -- five terms, not seven."""
    return _dirfun(_BZ, theta, 5)


def which_third_octave(f: float) -> int:
    """1-based band index for frequency ``f``, or 0 if outside the range."""
    for i in range(1, len(FMID) + 1):
        if f < FMID[i - 1]:
            return i - 1
    return 0


def convert_tones_to_third_octave(freqs, spl) -> list:
    """Sum discrete tones into the third-octave bands, energetically.

    Bands with no tone stay at :data:`TONE_FLOOR_DB`; the first tone in a band
    *replaces* the floor rather than adding to it, which is why the floor does
    not accumulate.
    """
    out = [TONE_FLOOR_DB] * NFREQ
    for f, s in zip(freqs, spl):
        i = which_third_octave(f)
        if i != 0:
            if out[i - 1] == TONE_FLOOR_DB:
                out[i - 1] = s
            else:
                out[i - 1] = 10.0 * math.log10(10.0 ** (out[i - 1] / 10.0)
                                               + 10.0 ** (s / 10.0))
    return out


# --------------------------------------------------------------------------
# Jet noise
# --------------------------------------------------------------------------

def jet_noise(Dist, theta, alpha, psi, rho0, p0, T0, mu0, c0,
              A6, A8, u6, u8, T6, T8, M0, type_) -> float:
    """Jet mixing noise, A-weighted dB. ``type_`` 0 unmixed, 1 mixed."""
    Dh = math.sqrt(4.0 * A6 / math.pi)      # round nozzle, so D_h = D

    rhoisa = P_ISA / (R_AIR * T_ISA)
    cisa = math.sqrt(GAMMA * R_AIR * T_ISA)

    u0 = M0 * math.sqrt(GAMMA * R_AIR * T0)

    udnorm = max(1.0 - (u0 / u6) * math.cos(alpha), 0.0001)
    ue = u6 * udnorm ** (2.0 / 3.0)

    M6 = u6 / math.sqrt(GAMMA * R_AIR * T6)
    Tt6 = T6 * (1.0 + 0.5 * (GAMMA - 1.0) * M6 ** 2)
    rho6 = p0 / (R_AIR * T6)

    Tt8 = 0.0
    if type_ == 0:
        M8 = u8 / math.sqrt(GAMMA * R_AIR * T8)
        Tt8 = T8 * (1.0 + 0.5 * (GAMMA - 1.0) * M8 ** 2)

    ggmi = GAMMA / (GAMMA - 1.0)
    w = (3.0 * (ue / c0) ** ggmi) / (0.6 + (ue / c0) ** ggmi) - 1.0

    # The fan-jet form of this is commented out above with "BUG???" against
    # it; the core-jet form is what ships.
    Mc = 0.62 * (u6 - u0 * math.cos(alpha)) / c0

    thetap = theta * (u6 / c0) ** 0.1
    cost, costp, cosp = math.cos(theta), math.cos(thetap), math.cos(psi)
    Mcosp = min(M0 * cosp, 0.99)

    UOL1 = (141.0
            + 10.0 * math.log10((rho0 / rhoisa) ** 2 * (c0 / cisa) ** 4)
            + 10.0 * math.log10(A6 / Dist ** 2)
            + 10.0 * math.log10((rho6 / rho0) ** w)
            + 10.0 * math.log10((ue / c0) ** 7.5)
            - 15.0 * math.log10((1.0 + Mc * cost) ** 2 + 0.04 * Mc ** 2)
            - 10.0 * math.log10(1.0 - Mcosp)
            + 3.0 * math.log10(2.0 * A6 / (math.pi * Dh ** 2) + 0.5))

    tS1 = (math.sqrt(4.0 * A6 / math.pi) / ue
           * (Dh / math.sqrt(4.0 * A6 / math.pi)) ** 0.4
           * (Tt6 / T0) ** (0.4 * (1.0 + costp))
           * (1.0 - M0 * cosp)
           * ((1.0 + 0.62 * ((u6 - u0) / c0) * cost) ** 2
              + 0.01538 * ((u6 - u0) / c0) ** 2) ** 0.5
           / ((1.0 + 0.62 * (u6 / c0) * cost) ** 2
              + 0.01538 * (u6 / c0) ** 2) ** 0.5)

    if type_ == 0:
        Arat, urat, Trat = A8 / A6, u8 / u6, Tt8 / Tt6
        m = 1.1 * math.sqrt(Arat) if Arat < 29.7 else 6.0
        arg = (max(1.0 - urat, 0.001) ** m
               + 1.2 * ((1.0 + Arat * urat ** 2) ** 4 / (1.0 + Arat) ** 3))
        UOL = UOL1 - 5.0 * math.log10(Trat) + 10.0 * math.log10(arg)
        Fs = fsfun(Arat, urat)
        TuArat = Trat * urat * Arat
        tmp = 0.5 + TuArat / (1.0 + TuArat)
        tS = tS1 / (1.0 - Fs * tmp ** 2)
    else:
        UOL, tS = UOL1, tS1

    tS = max(tS, 1.0e-6)

    SPLsum = 0.0
    for i in range(NFREQ):
        S = FREQ[i] * tS
        # thetap is in radians and fpfun wants degrees; see the module
        # docstring. Passed through as the source does.
        SPLdBA = (UOL + GROUND_REFLECTION_DB + fpfun(S, thetap)
                  + DBACORR[i] + DBATTEN[i] * Dist / 100.0)
        SPLsum += 10.0 ** (SPLdBA / 10.0)
    return 10.0 * math.log10(SPLsum)


# --------------------------------------------------------------------------
# Fan noise
# --------------------------------------------------------------------------

def esdu98008broadband_total(direc, method, Dist, theta, dTt, mfan, rss,
                             Mtr, Mtrd, bpf):
    """ESDU 98008 fan broadband. Returns ``(freqs, spl)``.

    Single-stage fan without inlet guide vanes only.
    """
    if method in (1, 3):
        L, M_cutover = (55.5, 0.9) if direc == +1 else (58.0, 1.0)
    else:
        L, M_cutover = (58.5, 0.9) if direc == +1 else (60.0, 1.0)

    if Mtrd >= 1.0:
        if Mtr > M_cutover:
            F1 = L + 20.0 * math.log10(Mtrd) - 20.0 * math.log10(Mtr)
        else:
            F1 = L + 20.0 * math.log10(Mtrd)
    elif Mtr <= M_cutover:
        F1 = L
    else:
        F1 = 0.0            # outside the correlation; the source returns zero

    F2 = -5.0 * math.log10(rss / 300.0)
    F3 = BBdirfun(theta, direc, method)

    spl0 = (20.0 * math.log10(dTt / 0.555)
            + 10.0 * math.log10(mfan / 0.454)
            + F1 + F2 + F3 - 20.0 * math.log10(Dist))

    freqs, spl = list(FREQ), []
    for f in freqs:
        if method == 1:
            arg = math.exp(-0.35 * (math.log(f / (2.0 * bpf))
                                    / math.log(2.2)) ** 2)
        elif method == 2:
            arg = math.exp(-0.50 * (math.log(f / (2.5 * bpf))
                                    / math.log(2.2)) ** 2)
        elif bpf > 0.5 * f:
            arg = math.exp(-0.35 * (math.log(f / (2.0 * bpf))
                                    / math.log(2.2)) ** 2)
        else:
            arg = math.exp(-2.00 * (math.log(f / (2.0 * bpf))
                                    / math.log(2.2)) ** 2)
        spl.append(max(spl0 + 10.0 * math.log10(arg), 0.0))
    return freqs, spl


class ToneCorrelationError(Exception):
    """Raised where ``esdu98008discretetone_total`` prints and stops."""


def esdu98008discretetone_total(direc, method, Dist, theta, dTt, mfan,
                                Mtr, Mtrd, Mt, RPM, rss, B, V, bpf,
                                nfmax=NFREQ):
    """ESDU 98008 fan discrete tones. Returns ``(freqs, spl)``.

    Harmonics of the blade-passing frequency. The series runs one harmonic
    past 10 kHz -- the next frequency is appended before the limit is tested
    -- and that one then falls outside every third-octave band.

    If the blade-passing frequency is low enough that 23 harmonics fit under
    10 kHz the source's loop falls out with ``nf`` one larger than the number
    of levels it computed, and its caller then reads an uninitialised level.
    This returns only the pairs it computed. No engine of interest gets near
    that -- a 737 breaks out at the seventh harmonic.
    """
    if method in (1, 2):
        L, M_cutover = (54.5, 0.72) if direc == +1 else (59.0, 1.0)
    else:
        L, M_cutover = (60.5, 0.72) if direc == +1 else (63.0, 1.0)

    if Mtrd >= 1.0:
        if Mtr > M_cutover:
            if direc == +1:
                L1 = (L + 20.0 * math.log10(Mtrd)
                      + 50.0 * math.log10(Mtr / 0.72))
                L2 = (L + 20.0 * math.log10(Mtrd)
                      + 80.0 * math.log10(Mtrd / Mtr) - 1.0)
                F1 = min(L1, L2)
            else:
                F1 = L + 20.0 * math.log10(Mtrd) - 20.0 * math.log10(Mtr)
        else:
            F1 = L + 20.0 * math.log10(Mtrd)
    elif Mtr <= M_cutover:
        F1 = L
    else:
        raise ToneCorrelationError(
            f"esdu98008discretetone_total: outside the correlation, "
            f"Mtrd={Mtrd} Mtr={Mtr} M_cutover={M_cutover}")

    F2 = -10.0 * math.log10(rss / 300.0)
    F3 = DTdirfun(theta, direc, method)

    Lp = max(20.0 * math.log10(dTt / 0.555)
             + 10.0 * math.log10(mfan / 0.454)
             + F1 + F2 + F3 - 20.0 * math.log10(Dist), 0.0)

    cut_off_factor = abs(Mt / (1.0 - V / B))

    freqs = [bpf]
    spl = []
    for i in range(1, nfmax):
        if method in (1, 3):
            if i == 1:
                F4 = Lp - 8.0 if cut_off_factor < 1.05 else Lp
            elif i == 2:
                F4 = Lp - 9.2
            else:
                F4 = Lp - 3.0 * float(i) - 1.8
        elif i == 1 and cut_off_factor < 1.05:
            F4 = Lp - 8.0
        else:
            F4 = Lp + 3.0 - 3.0 * float(i)

        F5 = 0.0
        if direc == +1:
            spl.append(10.0 * math.log10(10.0 ** (F4 / 10.0)
                                         + 10.0 ** (F5 / 10.0)))
        else:
            spl.append(F4)

        if freqs[i - 1] >= 10000.0:
            break
        freqs.append(freqs[i - 1] + bpf)
    return freqs[:len(spl)], spl


def esdu98008combinationtone_total(Dist, theta, shocklocation, rss, dTt,
                                   mfan, Mtr, bpf):
    """ESDU 98008 combination tones -- "buzzsaw". Returns ``(freqs, spl)``.

    Only meaningful above ``Mtr = 1``; three sub-harmonics of blade passing.
    """
    if Mtr < 1.135:
        F1a = 50.0 + 6.1 * (Mtr - 1.0) / 0.135
    else:
        F1a = 56.1 - 3.1 * (Mtr - 1.135) / 0.865
    F1b = F1a
    if Mtr < 1.413:
        F1c = 40.0 + 7.3 * (Mtr - 1.0) / 0.413
    else:
        F1c = 47.3 - 3.1 * (Mtr - 1.413) / 0.587

    F2 = BZdirfun(theta)
    if shocklocation == -1:
        F3 = -6.0           # shock ingested
    elif shocklocation == +1:
        F3 = 0.0            # shock expelled
    else:
        raise ValueError(f"shock location not recognised: {shocklocation}")

    base = (20.0 * math.log10(dTt / 0.555)
            + 10.0 * math.log10(mfan / 0.454) + F2 + F3
            - 20.0 * math.log10(Dist))
    spl = [max(0.0, base + F) for F in (F1a, F1b, F1c)]
    freqs = [bpf / 2.0, bpf / 4.0, bpf / 8.0]
    return freqs, spl


def esdu_fan_noise_total(method, Dist, theta, rho0, p0, T0, mu0, c0, M0,
                         etaf, FPR, rss, B, V, mdot, BPR, Mtrd, Mtr, Mt,
                         RPM):
    """The five fan components, A-weighted dB.

    Returns ``(tone_fwd, broadband_fwd, buzzsaw, tone_rear, broadband_rear)``.
    """
    Tt2 = T0 * (1.0 + 0.5 * (GAMMA - 1.0) * M0 ** 2)
    gmig = (GAMMA - 1.0) / GAMMA
    dTt = Tt2 * (FPR ** gmig - 1.0) / etaf
    mfan = mdot * BPR
    bpf = B * RPM / 60.0
    shocklocation = +1          # expelled; -1 would take 6 dB off the buzzsaw

    _, spl_bb_fore = esdu98008broadband_total(
        +1, method, Dist, theta, dTt, mfan, rss, Mtr, Mtrd, bpf)
    _, spl_bb_rear = esdu98008broadband_total(
        -1, method, Dist, theta, dTt, mfan, rss, Mtr, Mtrd, bpf)

    f_fore, s_fore = esdu98008discretetone_total(
        +1, method, Dist, theta, dTt, mfan, Mtr, Mtrd, Mt, RPM, rss, B, V,
        bpf)
    f_rear, s_rear = esdu98008discretetone_total(
        -1, method, Dist, theta, dTt, mfan, Mtr, Mtrd, Mt, RPM, rss, B, V,
        bpf)

    spl_tt_fore = convert_tones_to_third_octave(f_fore, s_fore)
    spl_tt_rear = convert_tones_to_third_octave(f_rear, s_rear)

    if Mtr > 1.0:
        f_bz, s_bz = esdu98008combinationtone_total(
            Dist, theta, shocklocation, rss, dTt, mfan, Mtr, bpf)
        spl_tt_buzz = convert_tones_to_third_octave(f_bz, s_bz)
    else:
        spl_tt_buzz = [TONE_FLOOR_DB] * NFREQ

    def weighted(spl):
        """A-weight, attenuate, add the ground reflection, floor at zero."""
        return [max(spl[i] + GROUND_REFLECTION_DB + DBACORR[i]
                    + DBATTEN[i] * Dist / 100.0, 0.0)
                for i in range(NFREQ)]

    dBA_tt_fore = weighted(spl_tt_fore)
    dBA_bb_fore = weighted(spl_bb_fore)
    dBA_tt_rear = weighted(spl_tt_rear)
    dBA_bb_rear = weighted(spl_bb_rear)
    dBA_tt_buzz = (weighted(spl_tt_buzz) if Mtr > 1.0
                   else [TONE_FLOOR_DB] * NFREQ)

    def integrate(band):
        return 10.0 * math.log10(sum(10.0 ** (v / 10.0) for v in band))

    return (integrate(dBA_tt_fore), integrate(dBA_bb_fore),
            integrate(dBA_tt_buzz), integrate(dBA_tt_rear),
            integrate(dBA_bb_rear))


# --------------------------------------------------------------------------

def tfnoise(x, y, z, climb, alpha, vector, rho0, p0, T0, mu0, c0,
            A6, A8, u6, u8, T6, T8, M0, etaf, FPR, mdot, BPR,
            Mtrd, Mtr, Mt, RPM, rss, B, V, htr, neng, type_, method):
    """Total A-weighted SPL at an observer, and its six components.

    Signature and argument order follow ``tfnoise.f``, so this drops straight
    into :func:`tasopt_py.sizing.noise.noise` as its ``tfnoise`` argument.
    Axes are standard stability-and-control: ``+x`` forward, ``+y`` right,
    ``+z`` down, all relative to the aircraft. ``htr`` is accepted and unused,
    as in the original.

    Returns a :class:`NoiseBreakdown`; the ``total`` is what the report wants.
    """
    Dist = math.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Observer in engine axes, for the directivity angle.
    te = climb + alpha + vector
    xe = x * math.cos(te) - z * math.sin(te)
    ye = y
    ze = z * math.cos(te) + x * math.sin(te)
    theta = math.atan2(math.sqrt(ye ** 2 + ze ** 2), xe)

    # ...and in flight-path axes, for the modified one.
    tf = climb
    xf = x * math.cos(tf) - z * math.sin(tf)
    yf = y
    zf = z * math.cos(tf) + x * math.sin(tf)
    psi = math.atan2(math.sqrt(yf ** 2 + zf ** 2), xf)

    neng_dB = 10.0 * math.log10(neng)

    spl_jet = jet_noise(Dist, theta, alpha, psi, rho0, p0, T0, mu0, c0,
                        A6, A8, u6, u8, T6, T8, M0, type_) + neng_dB

    f_t, f_b, f_z, r_t, r_b = esdu_fan_noise_total(
        method, Dist, theta, rho0, p0, T0, mu0, c0, M0, etaf, FPR,
        rss, B, V, mdot, BPR, Mtrd, Mtr, Mt, RPM)
    f_t += neng_dB
    f_b += neng_dB
    f_z += neng_dB
    r_t += neng_dB
    r_b += neng_dB

    total = 10.0 * math.log10(
        sum(10.0 ** (v / 10.0)
            for v in (spl_jet, f_t, f_b, f_z, r_t, r_b)))
    return NoiseBreakdown(spl_jet, f_t, f_b, f_z, r_t, r_b, total)
