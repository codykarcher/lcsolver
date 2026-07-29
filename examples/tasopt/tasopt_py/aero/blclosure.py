"""Boundary-layer closure relations -- the correlations inside ``blsys.f``.

The functions an integral BL method needs to close its two (or three)
equations: kinematic shape parameter, energy shape parameter, skin friction
and dissipation, laminar and turbulent. These are XFOIL's correlations, and
TASOPT carries them unchanged.

``hkin``
    Whitfield's kinematic shape parameter, ``Hk`` from ``H`` and ``M^2``.
``hsl`` / ``hst``
    Energy shape parameter ``H*``, laminar (Falkner-Skan) and turbulent.
``cfl`` / ``cft``
    Skin friction, Falkner-Skan and Coles.
``dil`` / ``dilw`` / ``dit``
    Dissipation ``2 C_D/H*``: laminar, laminar-wake, turbulent.
``hct``
    Density shape parameter.

Branches worth knowing about
----------------------------
``hst`` has an attached branch and a separated one, meeting at
``Ho = 3 + 400/Rt`` (floored at ``Ho = 4`` below ``Rt = 400``), and it clamps
the Reynolds number at ``Rtz >= 200`` — a note in the source dates that limit
to 12/4/94. It also carries an older Swafford-profile correlation, commented
out, replaced in November 1991; only the newer arctan-plus-Schlichting form is
live.

``dilw`` is the wake form: it calls ``hsl`` with ``Msq`` forced to **zero**
regardless of the actual Mach number, then builds the dissipation from a
different constant (1.10) than the attached-flow laminar branch.

``cft`` clamps ``log(Rt/Fc)`` at 3.0 and the exponential argument at -20,
which is what keeps it finite at very low Reynolds number and very high shape
parameter.

Values and derivatives
----------------------
Each routine comes in two forms. ``hkin(h, msq)`` returns the value alone;
``hkin_d(h, msq)`` returns the value followed by the partial derivatives, in
the order the Fortran returns them. The derivative forms exist because
:mod:`tasopt_py.aero.blsys` needs the same *analytic* Jacobian the Fortran
builds -- ``blax``'s Newton is driven by it, and matching it is what lets the
whole boundary-layer solve be compared to the reference to machine precision
rather than to solver tolerance.

Only the routines on the live path carry a ``_d`` form: ``hkin``, ``hsl``,
``hst``, ``cfl``, ``cft``, ``dil``. ``dilw``, ``dit`` and ``hct`` are called
from nowhere in the shipped program -- ``blvar``'s ``hct`` call is commented
out and it computes the density shape parameter inline instead, and the
turbulent dissipation is likewise built inline rather than through ``dit``.
Their value forms are kept, and tested, because they are XFOIL's and a reader
will look for them.

Two derivatives deliberately ignore a clamp above them. ``cft``'s ``cf_hk``
uses ``-1.33*cfo`` whether or not the exponential argument hit its ``-20``
floor, and its ``cf_rt`` stays nonzero after ``log(Rt/Fc)`` has been clamped
to 3. ``hkin``'s clamp in ``blvar`` (``hk = max(hk, 1.005)``) is likewise
applied after the derivatives are formed. All three are reproduced as written.

Verified against the compiled Fortran; see ``tests/test_blclosure.py``.
"""
from __future__ import annotations

import math

__all__ = ["hkin", "hsl", "hst", "cfl", "cft", "dil", "dilw", "dit", "hct",
           "hkin_d", "hsl_d", "hst_d", "cfl_d", "cft_d", "dil_d",
           "HSMIN", "DHSINF"]

HSMIN = 1.500
DHSINF = 0.015
GAM = 1.4


def hkin(h: float, msq: float) -> float:
    """Kinematic shape parameter, from Whitfield. Assumes air."""
    return (h - 0.29 * msq) / (1.0 + 0.113 * msq)


def dil(hk: float, rt: float) -> float:
    """Laminar dissipation ``2 C_D/H*``, from Falkner-Skan."""
    if hk < 4.0:
        return (0.00205 * (4.0 - hk) ** 5.5 + 0.207) / rt
    hkb = hk - 4.0
    den = 1.0 + 0.02 * hkb ** 2
    return (-0.0016 * hkb ** 2 / den + 0.207) / rt


def dilw(hk: float, rt: float) -> float:
    """Laminar *wake* dissipation.

    Note it evaluates ``hsl`` at ``Msq = 0`` whatever the real Mach number --
    the source sets ``MSQ = 0.`` explicitly before the call.
    """
    hs = hsl(hk, rt, 0.0)
    rcd = 1.10 * (1.0 - 1.0 / hk) ** 2 / hk
    return 2.0 * rcd / (hs * rt)


def hsl(hk: float, rt: float, msq: float) -> float:
    """Laminar energy shape parameter ``H*``."""
    if hk < 4.35:
        tmp = hk - 4.35
        return (0.0111 * tmp ** 2 / (hk + 1.0)
                - 0.0278 * tmp ** 3 / (hk + 1.0) + 1.528
                - 0.0002 * (tmp * hk) ** 2)
    return 0.015 * (hk - 4.35) ** 2 / hk + 1.528


def cfl(hk: float, rt: float, msq: float) -> float:
    """Laminar skin friction, from Falkner-Skan."""
    if hk < 5.5:
        tmp = (5.5 - hk) ** 3 / (hk + 1.0)
        return (0.0727 * tmp - 0.07) / rt
    tmp = 1.0 - 1.0 / (hk - 4.5)
    return (0.015 * tmp ** 2 - 0.07) / rt


def dit(hs: float, us: float, cf: float, st: float) -> float:
    """Turbulent dissipation ``2 C_D/H*``."""
    return (0.5 * cf * us + st * st * (1.0 - us)) * 2.0 / hs


def hst(hk: float, rt: float, msq: float) -> float:
    """Turbulent energy shape parameter ``H*``.

    Two branches meeting at ``Ho``, with the Reynolds number floored at 200.
    """
    ho = 3.0 + 400.0 / rt if rt > 400.0 else 4.0
    # Rtheta dependence limited below 200 -- source note dated 12/4/94.
    rtz = rt if rt > 200.0 else 200.0

    if hk < ho:
        # Attached: arctan(y+) plus Schlichting profiles, Nov 1991. The older
        # Swafford correlation is still in the source, commented out.
        hr = (ho - hk) / (ho - 1.0)
        hs = ((2.0 - HSMIN - 4.0 / rtz) * hr ** 2 * 1.5 / (hk + 0.5)
              + HSMIN + 4.0 / rtz)
    else:
        grt = math.log(rtz)
        hdif = hk - ho
        rtmp = hk - ho + 4.0 / grt
        htmp = 0.007 * grt / rtmp ** 2 + DHSINF / hk
        hs = hdif ** 2 * htmp + HSMIN + 4.0 / rtz

    fm = 1.0 + 0.014 * msq
    return (hs + 0.028 * msq) / fm


def cft(hk: float, rt: float, msq: float, cffac: float = 1.0) -> float:
    """Turbulent skin friction, from Coles."""
    gmi = GAM - 1.0
    fc = math.sqrt(1.0 + 0.5 * gmi * msq)
    grt = max(math.log(rt / fc), 3.0)
    gex = -1.74 - 0.31 * hk
    arg = max(-20.0, -1.33 * hk)
    thk = math.tanh(4.0 - hk / 0.875)
    cfo = cffac * 0.3 * math.exp(arg) * (grt / 2.3026) ** gex
    return (cfo + 1.1e-4 * (thk - 1.0)) / fc


def hct(hk: float, msq: float) -> float:
    """Density shape parameter."""
    return msq * (0.064 / (hk - 0.8) + 0.251)


# --------------------------------------------------------------------------
# Derivative forms. Each returns the value first, then the partials in the
# order the Fortran's argument list returns them.
# --------------------------------------------------------------------------

def hkin_d(h: float, msq: float):
    """``hk, hk_h, hk_msq``."""
    hk = (h - 0.29 * msq) / (1.0 + 0.113 * msq)
    hk_h = 1.0 / (1.0 + 0.113 * msq)
    # Note this uses the *new* hk, which is what makes it the right derivative.
    hk_msq = (-0.29 - 0.113 * hk) / (1.0 + 0.113 * msq)
    return hk, hk_h, hk_msq


def dil_d(hk: float, rt: float):
    """``di, di_hk, di_rt``."""
    if hk < 4.0:
        di = (0.00205 * (4.0 - hk) ** 5.5 + 0.207) / rt
        di_hk = (-0.00205 * 5.5 * (4.0 - hk) ** 4.5) / rt
    else:
        hkb = hk - 4.0
        den = 1.0 + 0.02 * hkb ** 2
        di = (-0.0016 * hkb ** 2 / den + 0.207) / rt
        di_hk = (-0.0016 * 2.0 * hkb
                 * (1.0 / den - 0.02 * hkb ** 2 / den ** 2)) / rt
    di_rt = -di / rt
    return di, di_hk, di_rt


def hsl_d(hk: float, rt: float, msq: float):
    """``hs, hs_hk, hs_rt, hs_msq``. The laminar ``H*`` has no Rt or Mach
    dependence at all, so the last two are identically zero."""
    if hk < 4.35:
        tmp = hk - 4.35
        hs = (0.0111 * tmp ** 2 / (hk + 1.0)
              - 0.0278 * tmp ** 3 / (hk + 1.0) + 1.528
              - 0.0002 * (tmp * hk) ** 2)
        hs_hk = (0.0111 * (2.0 * tmp - tmp ** 2 / (hk + 1.0)) / (hk + 1.0)
                 - 0.0278 * (3.0 * tmp ** 2 - tmp ** 3 / (hk + 1.0))
                 / (hk + 1.0)
                 - 0.0002 * 2.0 * tmp * hk * (tmp + hk))
    else:
        hs = 0.015 * (hk - 4.35) ** 2 / hk + 1.528
        hs_hk = (0.015 * 2.0 * (hk - 4.35) / hk
                 - 0.015 * (hk - 4.35) ** 2 / hk ** 2)
    return hs, hs_hk, 0.0, 0.0


def cfl_d(hk: float, rt: float, msq: float):
    """``cf, cf_hk, cf_rt, cf_msq``."""
    if hk < 5.5:
        tmp = (5.5 - hk) ** 3 / (hk + 1.0)
        cf = (0.0727 * tmp - 0.07) / rt
        cf_hk = (-0.0727 * tmp * 3.0 / (5.5 - hk)
                 - 0.0727 * tmp / (hk + 1.0)) / rt
    else:
        tmp = 1.0 - 1.0 / (hk - 4.5)
        cf = (0.015 * tmp ** 2 - 0.07) / rt
        cf_hk = (0.015 * tmp * 2.0 / (hk - 4.5) ** 2) / rt
    return cf, cf_hk, -cf / rt, 0.0


def hst_d(hk: float, rt: float, msq: float):
    """``hs, hs_hk, hs_rt, hs_msq``."""
    if rt > 400.0:
        ho = 3.0 + 400.0 / rt
        ho_rt = -400.0 / rt ** 2
    else:
        ho = 4.0
        ho_rt = 0.0

    if rt > 200.0:
        rtz = rt
        rtz_rt = 1.0
    else:
        rtz = 200.0
        rtz_rt = 0.0

    if hk < ho:
        hr = (ho - hk) / (ho - 1.0)
        hr_hk = -1.0 / (ho - 1.0)
        hr_rt = (1.0 - hr) / (ho - 1.0) * ho_rt
        hs = ((2.0 - HSMIN - 4.0 / rtz) * hr ** 2 * 1.5 / (hk + 0.5)
              + HSMIN + 4.0 / rtz)
        hs_hk = (-(2.0 - HSMIN - 4.0 / rtz) * hr ** 2 * 1.5 / (hk + 0.5) ** 2
                 + (2.0 - HSMIN - 4.0 / rtz) * hr * 2.0 * 1.5 / (hk + 0.5)
                 * hr_hk)
        hs_rt = ((2.0 - HSMIN - 4.0 / rtz) * hr * 2.0 * 1.5 / (hk + 0.5)
                 * hr_rt
                 + (hr ** 2 * 1.5 / (hk + 0.5) - 1.0) * 4.0 / rtz ** 2
                 * rtz_rt)
    else:
        grt = math.log(rtz)
        hdif = hk - ho
        rtmp = hk - ho + 4.0 / grt
        htmp = 0.007 * grt / rtmp ** 2 + DHSINF / hk
        htmp_hk = -0.014 * grt / rtmp ** 3 - DHSINF / hk ** 2
        htmp_rt = (-0.014 * grt / rtmp ** 3
                   * (-ho_rt - 4.0 / grt ** 2 / rtz * rtz_rt)
                   + 0.007 / rtmp ** 2 / rtz * rtz_rt)
        hs = hdif ** 2 * htmp + HSMIN + 4.0 / rtz
        hs_hk = hdif * 2.0 * htmp + hdif ** 2 * htmp_hk
        hs_rt = (hdif ** 2 * htmp_rt - 4.0 / rtz ** 2 * rtz_rt
                 + hdif * 2.0 * htmp * (-ho_rt))

    # Whitfield's compressibility correction. hs is overwritten before hs_msq
    # is formed, and hs_msq is written to use the *corrected* hs -- that is
    # correct, not a bug: d/dM^2 (hs0 + .028 M^2)/fm = .028/fm - .014 hs/fm.
    fm = 1.0 + 0.014 * msq
    hs = (hs + 0.028 * msq) / fm
    hs_hk = hs_hk / fm
    hs_rt = hs_rt / fm
    hs_msq = 0.028 / fm - 0.014 * hs / fm
    return hs, hs_hk, hs_rt, hs_msq


def cft_d(hk: float, rt: float, msq: float, cffac: float = 1.0):
    """``cf, cf_hk, cf_rt, cf_msq``.

    The two ``max`` clamps are not differentiated through: ``cf_hk`` keeps the
    ``-1.33 cfo`` term after ``arg`` has been floored at -20, and ``cf_rt``
    stays nonzero after ``grt`` has been floored at 3. Reproduced as written.
    """
    gmi = GAM - 1.0
    fc = math.sqrt(1.0 + 0.5 * gmi * msq)
    grt = max(math.log(rt / fc), 3.0)
    gex = -1.74 - 0.31 * hk
    arg = max(-20.0, -1.33 * hk)
    thk = math.tanh(4.0 - hk / 0.875)
    cfo = cffac * 0.3 * math.exp(arg) * (grt / 2.3026) ** gex
    cf = (cfo + 1.1e-4 * (thk - 1.0)) / fc
    cf_hk = (-1.33 * cfo - 0.31 * math.log(grt / 2.3026) * cfo
             - 1.1e-4 * (1.0 - thk ** 2) / 0.875) / fc
    cf_rt = gex * cfo / (fc * grt) / rt
    cf_msq = (gex * cfo / (fc * grt) * (-0.25 * gmi / fc ** 2)
              - 0.25 * gmi * cf / fc ** 2)
    return cf, cf_hk, cf_rt, cf_msq
