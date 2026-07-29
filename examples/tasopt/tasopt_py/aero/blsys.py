"""The two-point boundary-layer station system -- ``blsys.f``.

Two routines sit on top of the closure relations in
:mod:`tasopt_py.aero.blclosure`:

``blvar``
    everything that depends on a *single* station: shape parameters ``H``,
    ``Hk``, ``H*``, ``Hc``, skin friction ``cf`` and dissipation ``2 C_D/H*``,
    from that station's ``th``, ``ds``, ``ue``.
``blsys``
    the 3x3 system relating station ``i`` to station ``i-1``: the von Karman
    momentum equation, the kinetic-energy shape equation, and a closing
    equation that is either ``ue = uinv`` (direct mode) or ``Hk = Hksep``
    (inverse mode, used where the direct march would separate).

Both carry the Fortran's analytic derivatives rather than differentiating
numerically. That is a departure from how :mod:`tasopt_py.engine.tfoper` was
done, and it is deliberate: ``blax`` runs a *limited* Newton -- twenty passes,
with a step limiter, stopping on step size -- so its answer depends on the
Jacobian it was given, not only on the residual. Reproducing the Jacobian is
what lets the whole solve be checked against the reference to machine
precision instead of to solver tolerance.

Equations, in the form the residuals are written
------------------------------------------------
With ``()l`` denoting ``log`` of the ratio across the interval and ``()a`` the
midpoint average, and ``bd = b + 2 pi ds rn`` the effective perimeter:

``rr(1)``   ``tl - cfxa xl + (Ha + 2) ul + bl + rl``           (momentum)
``rr(2)``   ``hl - dcxa xl + (2 Hca/Hsa + 1 - Ha) ul``         (kinetic energy)
``rr(3)``   ``ue - uinv``  or  ``Hk - Hksep``                  (closure)

``xl = log(x/xm)`` and the equations are differenced logarithmically, so a
station at ``x = 0`` cannot be used -- hence the ``simi`` branch, which
replaces the interval with a self-similar station (``xl = bl = ul = 1``,
``rl = tl = hl = 0``).

Things in the source worth knowing
----------------------------------
* ``blvar``'s wake branch zeroes ``cf``, ``cf_th``, ``cf_ds`` and then writes
  ``cd_ue = 0.`` -- a typo for ``cf_ue``. Under ``implicit real`` that
  silently creates a new variable and leaves ``cf_ue`` alone, so the caller's
  ``cf_ue`` survives from the last station before the wake and is reused for
  every wake station. It reaches only the Jacobian (``cf`` itself is zero, so
  the residual is unaffected), which is why the solve still converges to the
  right answer. This port reproduces it: pass the previous ``cf_ue`` in as
  ``cf_ue``, exactly as the Fortran's argument aliasing does.
* ``blvar``'s ``hct`` call is commented out; the density shape parameter is
  built inline as ``Hc = (gam-1)/2 M_e^2 H``, which is not the same function.
* ``hk`` is clamped to 1.005 *after* its derivatives are formed, so the
  derivatives do not see the clamp.
* ``blsys`` has ``hca`` and all its derivatives zeroed out in a commented-out
  block -- i.e. the density-shape term in the energy equation was at some
  point switched off. It is live as shipped.
* ``blsys`` takes ``lami`` and ``wake``, and ``blvar`` takes ``simi`` and
  ``x``, and none of the four is used. Kept in the signatures so call sites
  match the originals.

Verified against the compiled Fortran; see ``tests/test_blsys.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from tasopt_py.aero.blclosure import cfl_d, cft_d, dil_d, hkin_d, hsl_d, hst_d

__all__ = ["BLState", "blvar", "blsys", "ACON", "BCON", "GAM"]

GAM = 1.4
#: Dissipation-closure constants. An earlier pair, 6.70/0.75, is commented out.
ACON, BCON = 6.0, 0.72


@dataclass
class BLState:
    """One station's shape parameters, friction and dissipation, with the
    derivatives with respect to that station's ``th``, ``ds``, ``ue``."""
    h: float = 0.0
    h_th: float = 0.0
    h_ds: float = 0.0
    hk: float = 0.0
    hk_th: float = 0.0
    hk_ds: float = 0.0
    hk_ue: float = 0.0
    hc: float = 0.0
    hc_th: float = 0.0
    hc_ds: float = 0.0
    hc_ue: float = 0.0
    hs: float = 0.0
    hs_th: float = 0.0
    hs_ds: float = 0.0
    hs_ue: float = 0.0
    cf: float = 0.0
    cf_th: float = 0.0
    cf_ds: float = 0.0
    cf_ue: float = 0.0
    di: float = 0.0
    di_th: float = 0.0
    di_ds: float = 0.0
    di_ue: float = 0.0

    def without_cf(self) -> "BLState":
        """A copy with the friction terms zeroed.

        ``blax`` does this to the *upstream* station once it is at or past the
        trailing edge, so the wake carries no wall shear from behind it.
        """
        return replace(self, cf=0.0, cf_th=0.0, cf_ds=0.0, cf_ue=0.0)


def blvar(simi: bool, lami: bool, wake: bool, Reyn: float, Mach: float,
          fexcr: float, x: float, th: float, ds: float, ue: float,
          cf_ue: float = 0.0) -> BLState:
    """Station values and their derivatives.

    ``Reyn`` and ``Mach`` are referred to freestream, and ``ue`` to freestream
    speed, so the local edge Mach number and Reynolds number are recovered
    from the isentropic temperature ratio here rather than passed in.

    ``fexcr`` is the excrescence multiplier on wall ``cf``. ``x`` and ``simi``
    are unused, as in the original.

    ``cf_ue`` is the incoming value the wake branch leaves untouched; see the
    module docstring.
    """
    gmi = GAM - 1.0

    trat = 1.0 + 0.5 * gmi * Mach ** 2 * (1.0 - ue ** 2)
    trat_ue = -gmi * Mach ** 2 * ue

    msq = (ue * Mach) ** 2 / trat
    msq_ue = 2.0 * ue * Mach ** 2 - (msq / trat) * trat_ue

    h = ds / th
    h_th = -h / th
    h_ds = 1.0 / th

    hk, hk_h, hk_msq = hkin_d(h, msq)
    hk_th = hk_h * h_th
    hk_ds = hk_h * h_ds
    hk_ue = hk_msq * msq_ue

    # Clamped after the derivatives are taken, so they do not see the clamp.
    hk = max(hk, 1.005)

    rh = trat ** (1.0 / gmi)
    rh_ue = rh / (gmi * trat) * trat_ue

    # mu ~ T assumed here; the source notes Sutherland's law could replace it.
    mu = trat / Reyn
    mu_ue = trat_ue / Reyn

    rt = rh * ue * th / mu
    rt_ue = rh * th / mu + rh_ue * ue * th / mu - (rt / mu) * mu_ue
    rt_th = rh * ue / mu

    hs, hs_hk, hs_rt, hs_msq = (hsl_d if lami else hst_d)(hk, rt, msq)
    hs_th = hs_hk * hk_th + hs_rt * rt_th
    hs_ds = hs_hk * hk_ds
    hs_ue = hs_hk * hk_ue + hs_rt * rt_ue + hs_msq * msq_ue

    # The hct call is commented out in the source; this inline form is live.
    hc = 0.5 * gmi * msq * h
    hc_ue = 0.5 * gmi * msq_ue * h
    hc_th = 0.5 * gmi * msq * h_th
    hc_ds = 0.5 * gmi * msq * h_ds

    if wake:
        cf = 0.0
        cf_th = 0.0
        cf_ds = 0.0
        # The source writes `cd_ue = 0.` here -- a typo. cf_ue is left as the
        # caller had it. Reproduced; see the module docstring.
    else:
        cf, cf_hk, cf_rt, cf_msq = (cfl_d if lami else cft_d)(hk, rt, msq)
        cf = fexcr * cf
        cf_hk = fexcr * cf_hk
        cf_rt = fexcr * cf_rt
        cf_msq = fexcr * cf_msq
        cf_th = cf_hk * hk_th + cf_rt * rt_th
        cf_ds = cf_hk * hk_ds
        cf_ue = cf_hk * hk_ue + cf_rt * rt_ue + cf_msq * msq_ue

    if lami:
        di, di_hk, di_rt = dil_d(hk, rt)
        di_th = di_hk * hk_th + di_rt * rt_th
        di_ds = di_hk * hk_ds
        di_ue = di_hk * hk_ue + di_rt * rt_ue
    else:
        # Turbulent dissipation built inline from an equilibrium shear-stress
        # estimate rather than through `dit`.
        hrat = (hk - 1.0) / (ACON * hk)
        hrat_hk = 1.0 / (ACON * hk ** 2)

        fc = math.sqrt(1.0 + 0.5 * gmi * msq)
        fc_ue = (0.25 * gmi / fc) * msq_ue

        uq = (0.5 * cf - hrat ** 2 / fc) / (BCON * hk)
        uq_cf = 0.5 / (BCON * hk)
        uq_hrat = -2.0 * hrat / fc / (BCON * hk)
        uq_fc = hrat ** 2 / fc ** 2 / (BCON * hk)

        uq_hk = uq_hrat * hrat_hk - uq / hk

        uq_th = uq_cf * cf_th + uq_hk * hk_th
        uq_ds = uq_cf * cf_ds + uq_hk * hk_ds
        uq_ue = uq_cf * cf_ue + uq_hk * hk_ue + uq_fc * fc_ue

        di = 0.5 * cf - (hk - 1.0) * uq
        di_cf = 0.5
        di_hk = -uq
        di_uq = -(hk - 1.0)

        di_th = di_cf * cf_th + di_hk * hk_th + di_uq * uq_th
        di_ds = di_cf * cf_ds + di_hk * hk_ds + di_uq * uq_ds
        di_ue = di_cf * cf_ue + di_hk * hk_ue + di_uq * uq_ue

    if wake:
        wfac = 2.0
        di = wfac * di
        di_th = wfac * di_th
        di_ds = wfac * di_ds
        di_ue = wfac * di_ue

    return BLState(h=h, h_th=h_th, h_ds=h_ds,
                   hk=hk, hk_th=hk_th, hk_ds=hk_ds, hk_ue=hk_ue,
                   hc=hc, hc_th=hc_th, hc_ds=hc_ds, hc_ue=hc_ue,
                   hs=hs, hs_th=hs_th, hs_ds=hs_ds, hs_ue=hs_ue,
                   cf=cf, cf_th=cf_th, cf_ds=cf_ds, cf_ue=cf_ue,
                   di=di, di_th=di_th, di_ds=di_ds, di_ue=di_ue)


def blsys(simi: bool, lami: bool, wake: bool, direct: bool, Mach: float,
          uinv: float, hksep: float,
          x: float, b: float, rn: float, th: float, ds: float, ue: float,
          v: BLState,
          xm: float, bm: float, rnm: float, thm: float, dsm: float,
          uem: float, vm: BLState):
    """The 3x3 system for station ``i`` against station ``i-1``.

    ``v`` and ``vm`` carry the station values and derivatives from
    :func:`blvar`; note that ``blax`` does not necessarily re-evaluate
    ``blvar`` at the upstream station -- it carries the previous sweep's
    values forward, with ``cf`` zeroed past the trailing edge.

    Returns ``(aa, bb, rr)``: the 3x3 derivative of the residuals with respect
    to ``(th, ds, ue)`` and to ``(thm, dsm, uem)``, and the residuals. Both
    matrices are 0-indexed here, unlike the parameter arrays elsewhere in this
    port -- they are plain local scratch, not indexed by name.

    ``lami`` and ``wake`` are unused, as in the original.
    """
    gmi = GAM - 1.0

    trat = 1.0 + 0.5 * gmi * Mach ** 2 * (1.0 - ue ** 2)
    trat_ue = -gmi * Mach ** 2 * ue

    amsq = 1.0 + 0.5 * gmi * Mach ** 2 * (1.0 - uem ** 2)
    amsq_uem = -gmi * Mach ** 2 * uem

    rh = trat ** (1.0 / gmi)
    rh_ue = rh / (gmi * trat) * trat_ue

    rhm = amsq ** (1.0 / gmi)
    rhm_uem = rhm / (gmi * amsq) * amsq_uem

    if simi:
        # Self-similar station: the upstream station is at x = 0, so the
        # logarithmic differences are replaced by the similarity values and
        # everything upstream drops out of the Jacobian.
        xl = bl = ul = 1.0
        rl = tl = hl = 0.0
        bl_ds = rl_ue = ul_ue = tl_th = hl_hs = 0.0
        bl_dsm = rl_uem = ul_uem = tl_thm = hl_hsm = 0.0

        hl_th = hl_hs * v.hs_th
        hl_ds = hl_hs * v.hs_ds
        hl_ue = hl_hs * v.hs_ue
        hl_thm = hl_dsm = hl_uem = 0.0

        cfxa = 0.5 * v.cf * x / th
        cfxa_th = 0.5 * v.cf_th * x / th - cfxa / th
        cfxa_ds = 0.5 * v.cf_ds * x / th
        cfxa_ue = 0.5 * v.cf_ue * x / th
        cfxa_thm = cfxa_dsm = cfxa_uem = 0.0

        dcxa = v.di * x / th - 0.5 * v.cf * x / th
        dcxa_th = v.di_th * x / th - 0.5 * v.cf_th * x / th - dcxa / th
        dcxa_ds = v.di_ds * x / th - 0.5 * v.cf_ds * x / th
        dcxa_ue = v.di_ue * x / th - 0.5 * v.cf_ue * x / th
        dcxa_thm = dcxa_dsm = dcxa_uem = 0.0

        ha, ha_th, ha_ds = v.h, v.h_th, v.h_ds
        ha_thm = ha_dsm = 0.0

        hsa, hsa_th, hsa_ds, hsa_ue = v.hs, v.hs_th, v.hs_ds, v.hs_ue
        hsa_thm = hsa_dsm = hsa_uem = 0.0

        hca, hca_th, hca_ds, hca_ue = v.hc, v.hc_th, v.hc_ds, v.hc_ue
        hca_thm = hca_dsm = hca_uem = 0.0

    else:
        bd = b + 2.0 * math.pi * ds * rn          # effective perimeter
        bdm = bm + 2.0 * math.pi * dsm * rnm

        xl = math.log(x / xm)
        bl = math.log(bd / bdm)
        rl = math.log(rh / rhm)
        ul = math.log(ue / uem)
        tl = math.log(th / thm)
        hl = math.log(v.hs / vm.hs)
        bl_ds = 1.0 / bd * 2.0 * math.pi * rn
        rl_ue = 1.0 / rh * rh_ue
        ul_ue = 1.0 / ue
        tl_th = 1.0 / th
        hl_hs = 1.0 / v.hs
        bl_dsm = -1.0 / bdm * 2.0 * math.pi * rnm
        rl_uem = -1.0 / rhm * rhm_uem
        ul_uem = -1.0 / uem
        tl_thm = -1.0 / thm
        hl_hsm = -1.0 / vm.hs

        hl_th = hl_hs * v.hs_th
        hl_ds = hl_hs * v.hs_ds
        hl_ue = hl_hs * v.hs_ue
        hl_thm = hl_hsm * vm.hs_th
        hl_dsm = hl_hsm * vm.hs_ds
        hl_uem = hl_hsm * vm.hs_ue

        cfx = 0.5 * v.cf * x / th
        cfx_th = 0.5 * v.cf_th * x / th - cfx / th
        cfx_ds = 0.5 * v.cf_ds * x / th
        cfx_ue = 0.5 * v.cf_ue * x / th
        cfxm = 0.5 * vm.cf * xm / thm
        cfxm_thm = 0.5 * vm.cf_th * xm / thm - cfxm / thm
        cfxm_dsm = 0.5 * vm.cf_ds * xm / thm
        cfxm_uem = 0.5 * vm.cf_ue * xm / thm

        cfxa = 0.5 * (cfx + cfxm)
        cfxa_th = 0.5 * cfx_th
        cfxa_ds = 0.5 * cfx_ds
        cfxa_ue = 0.5 * cfx_ue
        cfxa_thm = 0.5 * cfxm_thm
        cfxa_dsm = 0.5 * cfxm_dsm
        cfxa_uem = 0.5 * cfxm_uem

        dcx = v.di * x / th - 0.5 * v.cf * x / th
        dcx_th = v.di_th * x / th - 0.5 * v.cf_th * x / th - dcx / th
        dcx_ds = v.di_ds * x / th - 0.5 * v.cf_ds * x / th
        dcx_ue = v.di_ue * x / th - 0.5 * v.cf_ue * x / th
        dcxm = vm.di * xm / thm - 0.5 * vm.cf * xm / thm
        dcxm_thm = (vm.di_th * xm / thm - 0.5 * vm.cf_th * xm / thm
                    - dcxm / thm)
        dcxm_dsm = vm.di_ds * xm / thm - 0.5 * vm.cf_ds * xm / thm
        dcxm_uem = vm.di_ue * xm / thm - 0.5 * vm.cf_ue * xm / thm

        dcxa = 0.5 * (dcx + dcxm)
        dcxa_th = 0.5 * dcx_th
        dcxa_ds = 0.5 * dcx_ds
        dcxa_ue = 0.5 * dcx_ue
        dcxa_thm = 0.5 * dcxm_thm
        dcxa_dsm = 0.5 * dcxm_dsm
        dcxa_uem = 0.5 * dcxm_uem

        ha = 0.5 * (v.h + vm.h)
        ha_th = 0.5 * v.h_th
        ha_ds = 0.5 * v.h_ds
        ha_thm = 0.5 * vm.h_th
        ha_dsm = 0.5 * vm.h_ds

        hsa = 0.5 * (v.hs + vm.hs)
        hsa_th = 0.5 * v.hs_th
        hsa_ds = 0.5 * v.hs_ds
        hsa_ue = 0.5 * v.hs_ue
        hsa_thm = 0.5 * vm.hs_th
        hsa_dsm = 0.5 * vm.hs_ds
        hsa_uem = 0.5 * vm.hs_ue

        # A block zeroing hca and its derivatives follows this in the source,
        # commented out -- the density-shape term in the energy equation is
        # live as shipped.
        hca = 0.5 * (v.hc + vm.hc)
        hca_th = 0.5 * v.hc_th
        hca_ds = 0.5 * v.hc_ds
        hca_ue = 0.5 * v.hc_ue
        hca_thm = 0.5 * vm.hc_th
        hca_dsm = 0.5 * vm.hc_ds
        hca_uem = 0.5 * vm.hc_ue

    aa = [[0.0] * 3 for _ in range(3)]
    bb = [[0.0] * 3 for _ in range(3)]
    rr = [0.0] * 3

    # --- momentum ---------------------------------------------------------
    rr[0] = tl - cfxa * xl + (ha + 2.0) * ul + bl + rl
    aa[0][0] = tl_th - cfxa_th * xl + ha_th * ul
    aa[0][1] = -cfxa_ds * xl + ha_ds * ul + bl_ds
    aa[0][2] = -cfxa_ue * xl + (ha + 2.0) * ul_ue + rl_ue
    bb[0][0] = tl_thm - cfxa_thm * xl + ha_thm * ul
    bb[0][1] = -cfxa_dsm * xl + ha_dsm * ul + bl_dsm
    bb[0][2] = -cfxa_uem * xl + (ha + 2.0) * ul_uem + rl_uem

    # --- kinetic energy ---------------------------------------------------
    btmp = 2.0 * hca / hsa + 1.0 - ha
    btmp_hca = 2.0 / hsa
    btmp_hsa = -2.0 * hca / hsa ** 2
    btmp_ha = -1.0

    btmp_th = btmp_hca * hca_th + btmp_hsa * hsa_th + btmp_ha * ha_th
    btmp_ds = btmp_hca * hca_ds + btmp_hsa * hsa_ds + btmp_ha * ha_ds
    btmp_ue = btmp_hca * hca_ue + btmp_hsa * hsa_ue

    btmp_thm = btmp_hca * hca_thm + btmp_hsa * hsa_thm + btmp_ha * ha_thm
    btmp_dsm = btmp_hca * hca_dsm + btmp_hsa * hsa_dsm + btmp_ha * ha_dsm
    btmp_uem = btmp_hca * hca_uem + btmp_hsa * hsa_uem

    rr[1] = hl - dcxa * xl + btmp * ul
    aa[1][0] = hl_th - dcxa_th * xl + btmp_th * ul
    aa[1][1] = hl_ds - dcxa_ds * xl + btmp_ds * ul
    aa[1][2] = hl_ue - dcxa_ue * xl + btmp_ue * ul + btmp * ul_ue
    bb[1][0] = hl_thm - dcxa_thm * xl + btmp_thm * ul
    bb[1][1] = hl_dsm - dcxa_dsm * xl + btmp_dsm * ul
    bb[1][2] = hl_uem - dcxa_uem * xl + btmp_uem * ul + btmp * ul_uem

    # --- closing equation -------------------------------------------------
    if direct:
        rr[2] = ue - uinv
        aa[2][0] = 0.0
        aa[2][1] = 0.0
        aa[2][2] = 1.0
    else:
        # Inverse mode: hold Hk at the separation value and let ue float.
        rr[2] = v.hk - hksep
        aa[2][0] = v.hk_th
        aa[2][1] = v.hk_ds
        aa[2][2] = v.hk_ue

    return aa, bb, rr
