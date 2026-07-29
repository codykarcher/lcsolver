"""Axisymmetric boundary layer and wake -- ``blax.f``.

Drives an integral BL with lateral divergence along a body and into its wake,
coupled to the inviscid solution through a source model of the viscous
displacement. Two stages:

1. **A direct march.** Sweep downstream station by station, solving the 3x3
   system of :mod:`tasopt_py.aero.blsys` for ``th``, ``ds``, ``ue`` at each
   one with the upstream station frozen. Where the shape parameter exceeds
   ``Hksep = 2.9`` the station switches to inverse mode -- ``Hk`` is held at
   ``Hksep`` and ``ue`` floats -- which is what lets the march walk through
   separated flow instead of failing. This stage only produces a starting
   guess.

2. **A global Newton** over ``3n`` unknowns (``th``, ``md``, ``ue`` at every
   station), where the mass defect ``md = ue ds (b + 2 pi ds rn)`` replaces
   ``ds`` as the unknown, and the edge velocity is required to equal the
   inviscid velocity *plus* the displacement source field,
   ``ue_i = uinv_i + sum_j (duvis/dmd)_ij md_j``. Every station is direct
   here; the inverse-mode fudge is only in stage 1. The coupling is dense in
   ``md`` through that last equation, so the system is solved whole.

The influence coefficients are the 1-D source-line kernel ``1/(4 pi dx |dx|)``
differenced across each interval -- a line of point sources on the axis, not
the compressible source *panels* :mod:`tasopt_py.aero.axisol` uses for the
inviscid problem.

Units: ``xi`` and ``bi`` in whatever length unit the caller likes, ``uinv``
and the returned ``ue`` normalised by freestream speed, ``Reyn`` and ``Mach``
referred to freestream. The docstring of ``blax.f`` lists how to recover
dimensional defects from the returned thicknesses; :mod:`tasopt_py.aero.fusebl`
does exactly that.

Why the Jacobian is analytic here
---------------------------------
Both Newtons stop on *step size* (``dmax < 1e-6``) and both are capped -- 20
inner iterations, 20 outer passes -- with a step limiter that can cut a step
to a fraction of the current value. An iteration count that is allowed to run
out is not a fixed point, so the answer depends on the exact iterate path, and
the iterate path depends on the Jacobian. That is why this module and
:mod:`tasopt_py.aero.blsys` carry the Fortran's analytic derivatives, and why
:func:`tasopt_py.linalg.gaussn` is a literal port rather than a library solve.

Things in the source worth knowing
----------------------------------
* ``cdi(1)`` is never assigned by ``blax``, and the wake-integral seed
  ``phi(2) = phi(1) + 0.5*(dib + dibm)*(x - xm)`` reads it. In the shipped
  program the array is a ``COMMON`` block, so it is zero; this port
  initialises it to zero and says so rather than leaving it to chance.
* ``phi(i) = 0.`` in the initialisation block uses ``i`` left over from the
  loop above it, so it zeroes ``phi(n+1)``, not ``phi(1)``. Harmless --
  ``phi(1)`` is set to zero before the integral that actually matters -- and
  reproduced.
* The mass defect is formed **two different ways**. The march ends each
  station with ``mdi = ue ds (b + 2 pi ds)``, without ``rn``; the Newton
  update ends with ``mdi = ue ds (b + 2 pi ds rn)``, with it. Only the second
  is consistent with the inversion at the top of the Newton sweep, so the
  march's mass defects are slightly off -- they are a starting guess, and the
  Newton corrects them.
* The Newton's step limiter inverts ``md -> ds`` **without** the ``rn``
  division the sweep uses, so ``ddsi`` is not the change in ``dsi`` implied by
  ``dmdi``. It cancels at the fixed point (``dsi`` only feeds ``mdi``, which
  the next sweep re-inverts consistently), so it costs iterations, not
  accuracy.
* ``hkprev`` is set and never read -- the separation test it belonged to,
  ``.and. hk .gt. hkprev``, is commented out.

Verified against the compiled Fortran; see ``tests/test_blax.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from tasopt_py.aero.blsys import BLState, blsys, blvar
from tasopt_py.linalg import gaussn

__all__ = ["BLAXResult", "blax", "HKSEP", "EPS", "NPASS", "IDIM"]

QOPI = 0.07957747154594766788444188168625718     # 1/(4 pi)
#: Shape parameter at which the direct march gives up and goes inverse.
HKSEP = 2.9
#: Newton stopping tolerance, on relative step size rather than on residual.
EPS = 1.0e-6
#: Outer Newton passes, and inner iterations per station in the direct march.
NPASS = 20
NITER = 20
#: blax.f's local array dimension.
IDIM = 60


@dataclass
class BLAXResult:
    """Per-station BL quantities, 0-indexed, one entry per input station."""
    ue: list = field(default_factory=list)    # edge velocity
    ds: list = field(default_factory=list)    # displacement thickness
    th: list = field(default_factory=list)    # momentum thickness
    ts: list = field(default_factory=list)    # kinetic energy thickness
    dc: list = field(default_factory=list)    # density flux thickness
    cf: list = field(default_factory=list)    # skin friction coefficient
    cd: list = field(default_factory=list)    # dissipation coefficient
    ct: list = field(default_factory=list)    # max shear stress coefficient
    hk: list = field(default_factory=list)    # kinematic shape parameter
    ph: list = field(default_factory=list)    # running dissipation integral


def blax(n: int, ite: int, xi, bi, rni, uinv, Reyn: float, Mach: float,
         fexcr: float) -> BLAXResult:
    """Solve the BL and wake along a body.

    ``n`` stations, the trailing edge at 1-based index ``ite``; ``xi`` is arc
    length, ``bi`` the lateral BL width (body perimeter; 1 for 2-D), ``rni``
    is ``dr/dn``, the cosine of the contour angle from the axis (0 for 2-D),
    and ``uinv`` the inviscid edge velocity. ``fexcr`` multiplies wall ``cf``
    for excrescences.

    The input sequences are 0-indexed, as :mod:`tasopt_py.aero.axisol` returns
    them; internally everything is 1-based so the body reads like the Fortran.
    """
    if n > IDIM:
        raise ValueError(f"blax: local array overflow, increase idim to {n}")

    def pad(v):
        return [0.0] + [float(t) for t in v[:n]]

    xi_, bi_, rni_, uinv_ = pad(xi), pad(bi), pad(rni), pad(uinv)

    # The Fortran's output arrays live in a COMMON block and so arrive zeroed;
    # several entries at station 1 are never assigned and are read anyway.
    # Sized n+2 because the initialisation below writes phi(n+1).
    z = lambda: [0.0] * (n + 2)
    uei, rhi, dsi, thi, tsi, dci = z(), z(), z(), z(), z(), z()
    cfi, cdi, cti, hki, phi, mdi = z(), z(), z(), z(), z(), z()

    gmi = 1.4 - 1.0

    for i in range(1, n + 1):
        uei[i] = uinv_[i]
        trat = 1.0 + 0.5 * gmi * Mach ** 2 * (1.0 - uei[i] ** 2)
        rhi[i] = trat ** (1.0 / gmi)

    # First point is not calculated if xi = 0 there.
    if xi_[1] == 0.0:
        thi[1] = 0.0
        dsi[1] = 0.0
        mdi[1] = 0.0
        # `phi(i) = 0.` -- i is left over from the loop above, so this zeroes
        # phi(n+1). Reproduced; see the module docstring.
        phi[n + 1] = 0.0

    # blvar's wake branch never writes cf_ue, so the caller's value carries
    # from the last pre-wake station through the whole wake.
    cf_ue = 0.0
    vm = BLState()

    # =====================================================================
    # Stage 1: direct march downstream, fudging Hk where it would separate.
    # =====================================================================
    direct = True
    for i in range(2, n + 1):
        simi = xi_[i - 1] == 0.0
        lami = False              # `lami = simi` is commented out above it
        wake = i > ite

        x, b, rn = xi_[i], bi_[i], rni_[i]
        xm, bm, rnm = xi_[i - 1], bi_[i - 1], rni_[i - 1]

        if simi:
            uem = thm = dsm = 0.0
            ue = uei[i]
            rex = ue * x * Reyn
            th = 0.4 * x / math.sqrt(rex)
            ds = th * 2.0
        else:
            uem, thm, dsm = uei[i - 1], thi[i - 1], dsi[i - 1]
            th, ds = thi[i - 1], dsi[i - 1]
            # A direct previous station means uinv is still a good guess for
            # ue here; an inverse one means it is not, so carry ue across.
            ue = uei[i] if direct else uei[i - 1]

        direct = True             # always try direct mode first
        hkprev = 0.0              # set and never read; see the docstring
        v = BLState()
        for it in range(1, NITER + 1):
            v = blvar(simi, lami, wake, Reyn, Mach, fexcr, x, th, ds, ue,
                      cf_ue)
            cf_ue = v.cf_ue
            if it == 1:
                hkprev = v.hk
            elif v.hk > HKSEP:
                # Hk limit exceeded -- switch this point to inverse mode.
                direct = False

            aa, bb, rr = blsys(simi, lami, wake, direct, Mach, uinv_[i],
                               HKSEP, x, b, rn, th, ds, ue, v,
                               xm, bm, rnm, thm, dsm, uem, vm)

            # gaussn wants 1-based storage; blsys returns plain 3x3 scratch.
            sol = [0.0] + rr
            gaussn(3, [[0.0] * 4] + [[0.0] + row for row in aa], sol)
            dth, dds, due = -sol[1], -sol[2], -sol[3]

            rlx = 1.0
            if rlx * dth > 1.6 * th:
                rlx = 1.6 * th / dth
            if rlx * dth < -0.6 * th:
                rlx = -0.6 * th / dth
            if rlx * dds > 2.5 * ds:
                rlx = 2.5 * ds / dds
            if rlx * dds < -0.4 * ds:
                rlx = -0.4 * ds / dds
            if rlx * due > 0.2 * ue:
                rlx = 0.2 * ue / due
            if rlx * due < -0.1 * ue:
                rlx = -0.1 * ue / due

            dmax = max(abs(dth) / th, abs(dds) / ds, abs(due) / ue)

            th += rlx * dth
            ds += rlx * dds
            ue += rlx * due

            if dmax < EPS:
                break

        # Note the closure values stored are from the last blvar call, i.e.
        # evaluated one Newton step behind th/ds/ue.
        uei[i], dsi[i], thi[i] = ue, ds, th
        tsi[i] = v.hs * th
        dci[i] = v.hc * th
        cfi[i] = v.cf
        cdi[i] = v.di * v.hs / 2.0
        cti[i] = 0.03 * 0.5 * v.hs * ((v.hk - 1.0) / v.hk) ** 2
        hki[i] = v.hk

        dib = cdi[i] * rhi[i] * uei[i] ** 3 * (b + 2.0 * math.pi * ds * rn)
        dibm = (cdi[i - 1] * rhi[i - 1] * uei[i - 1] ** 3
                * (bm + 2.0 * math.pi * dsm * rnm))
        phi[i] = phi[i - 1] + 0.5 * (dib + dibm) * (x - xm)
        # No rn here, unlike the Newton update below. See the docstring.
        mdi[i] = ue * ds * (b + 2.0 * math.pi * ds)

        vm = v.without_cf() if i >= ite else v

    # =====================================================================
    # Stage 2: global viscous/inviscid Newton with a displacement source
    # model.
    # =====================================================================
    nsys = 3 * n

    uvis_mdi = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n):
            dx = xi_[i] - 0.5 * (xi_[j + 1] + xi_[j])
            uvis_mdi[i][j + 1] += QOPI / (dx * abs(dx))
            uvis_mdi[i][j] -= QOPI / (dx * abs(dx))

    uvis = [0.0] * (n + 1)
    dthi = [0.0] * (n + 1)
    dmdi = [0.0] * (n + 1)
    duei = [0.0] * (n + 1)
    ddsi = [0.0] * (n + 1)

    for _ipass in range(1, NPASS + 1):
        asys = [[0.0] * (nsys + 1) for _ in range(nsys + 1)]
        rsys = [0.0] * (nsys + 1)

        # First point's variables are frozen -- 1s on the diagonal.
        asys[1][1] = 1.0
        asys[2][2] = 1.0
        asys[3][3] = 1.0

        dsi[1] = 0.0
        for i in range(2, n + 1):
            uvis[i] = uinv_[i]
            for j in range(1, n + 1):
                uvis[i] += uvis_mdi[i][j] * mdi[j]

        for i in range(2, n + 1):
            simi = xi_[i - 1] == 0.0
            lami = False
            wake = i > ite

            x, b, rn = xi_[i], bi_[i], rni_[i]
            th, md, ue = thi[i], mdi[i], uei[i]
            xm, bm, rnm = xi_[i - 1], bi_[i - 1], rni_[i - 1]
            thm, mdm, uem = thi[i - 1], mdi[i - 1], uei[i - 1]

            # Invert md = ue ds (b + 2 pi ds rn) for ds:
            #   0.5 rn ds^2 + (b/4pi) ds - md/(4 pi ue) = 0
            bp = b * 0.25 / math.pi
            mpu = 0.5 * md / (math.pi * ue)
            if rn <= 1.0e-6:
                ds = md / (ue * b)
                ds_md = 1.0 / (ue * b)
                ds_ue = -ds / ue
            else:
                ds = (math.sqrt(bp ** 2 + mpu) - bp) / rn
                ds_md = 0.5 / math.sqrt(bp ** 2 + mpu) * (mpu / md) / rn
                ds_ue = 0.5 / math.sqrt(bp ** 2 + mpu) * (-mpu / ue) / rn

            if simi:
                dsm = dsm_mdm = dsm_uem = 0.0
            elif rnm <= 1.0e-6:
                dsm = mdm / (uem * bm)
                dsm_mdm = 1.0 / (uem * bm)
                dsm_uem = -dsm / uem
            else:
                bpm = bm * 0.25 / math.pi
                mpum = 0.5 * mdm / (math.pi * uem)
                rootm = math.sqrt(bpm ** 2 + mpum)
                dsm = (rootm - bpm) / rnm
                dsm_mdm = 0.5 / rootm * (mpum / mdm) / rnm
                dsm_uem = 0.5 / rootm * (-mpum / uem) / rnm

            # The upstream station is not re-evaluated -- the call is
            # commented out in the source -- so vm carries forward from the
            # previous sweep, which is the same state.
            v = blvar(simi, lami, wake, Reyn, Mach, fexcr, x, th, ds, ue,
                      cf_ue)
            cf_ue = v.cf_ue
            aa, bb, rr = blsys(simi, lami, wake, True, Mach, uinv_[i], HKSEP,
                               x, b, rn, th, ds, ue, v,
                               xm, bm, rnm, thm, dsm, uem, vm)

            # Put the BL equations of the small 3x3 system into the big one,
            # changing variable from ds to md as they go in.
            for k in (1, 2):
                ksys = 3 * (i - 1) + k
                rsys[ksys] = rr[k - 1]

                r_th, r_ds, r_ue = aa[k - 1]
                r_thm, r_dsm, r_uem = bb[k - 1]

                lsys = 3 * (i - 1) + 1
                asys[ksys][lsys] = r_th
                asys[ksys][lsys - 3] = r_thm

                lsys = 3 * (i - 1) + 2
                asys[ksys][lsys] = r_ds * ds_md
                asys[ksys][lsys - 3] = r_dsm * dsm_mdm

                lsys = 3 * (i - 1) + 3
                asys[ksys][lsys] = r_ds * ds_ue + r_ue
                asys[ksys][lsys - 3] = r_dsm * dsm_uem + r_uem

            # Third equation: ue = uvis, the viscous/inviscid coupling.
            ksys = 3 * (i - 1) + 3
            rsys[ksys] = ue - uvis[i]
            for j in range(1, n + 1):
                asys[ksys][3 * (j - 1) + 2] = -uvis_mdi[i][j]
            asys[ksys][3 * (i - 1) + 3] += 1.0

            tsi[i] = v.hs * th
            dci[i] = v.hc * th
            cfi[i] = v.cf
            cdi[i] = v.di * v.hs / 2.0
            cti[i] = 0.03 * 0.5 * v.hs * ((v.hk - 1.0) / v.hk) ** 2
            hki[i] = v.hk

            vm = v.without_cf() if i >= ite else v

        gaussn(nsys, asys, rsys)

        dmax = 0.0
        rlx = 1.0
        for i in range(2, n + 1):
            b, th, md, ue = bi_[i], thi[i], mdi[i], uei[i]

            dthi[i] = -rsys[3 * (i - 1) + 1]
            dmdi[i] = -rsys[3 * (i - 1) + 2]
            duei[i] = -rsys[3 * (i - 1) + 3]

            # Note: no rn division here, unlike the sweep above.
            bp = b * 0.25 / math.pi
            mpu = 0.5 * md / (math.pi * ue)
            ds = math.sqrt(bp ** 2 + mpu) - bp
            ds_md = 0.5 / math.sqrt(bp ** 2 + mpu) * (mpu / md)
            ds_ue = 0.5 / math.sqrt(bp ** 2 + mpu) * (-mpu / ue)

            ddsi[i] = ds_md * dmdi[i] + ds_ue * duei[i]

            if rlx * dthi[i] > 1.6 * th:
                rlx = 1.6 * th / dthi[i]
            if rlx * dthi[i] < -0.6 * th:
                rlx = -0.6 * th / dthi[i]
            if rlx * ddsi[i] > 2.5 * ds:
                rlx = 2.5 * ds / ddsi[i]
            if rlx * ddsi[i] < -0.4 * ds:
                rlx = -0.4 * ds / ddsi[i]
            if rlx * duei[i] > 0.2 * ue:
                rlx = 0.2 * ue / duei[i]
            if rlx * duei[i] < -0.1 * ue:
                rlx = -0.1 * ue / duei[i]

            dmax = max(dmax, abs(dthi[i]) / th, abs(ddsi[i]) / ds,
                       abs(duei[i]) / ue)

        for i in range(2, n + 1):
            thi[i] += rlx * dthi[i]
            dsi[i] += rlx * ddsi[i]
            uei[i] += rlx * duei[i]
            mdi[i] = uei[i] * dsi[i] * (bi_[i]
                                        + 2.0 * math.pi * dsi[i] * rni_[i])

        if dmax < EPS:
            break

    # ---- running dissipation integral ----------------------------------
    phi[1] = 0.0
    for i in range(2, n + 1):
        x, xm = xi_[i], xi_[i - 1]
        dib = (cdi[i] * rhi[i] * uei[i] ** 3
               * (bi_[i] + 2.0 * math.pi * dsi[i] * rni_[i]))
        dibm = (cdi[i - 1] * rhi[i - 1] * uei[i - 1] ** 3
                * (bi_[i - 1] + 2.0 * math.pi * dsi[i - 1] * rni_[i - 1]))
        phi[i] = phi[i - 1] + 0.5 * (dib + dibm) * (x - xm)

    cut = slice(1, n + 1)
    return BLAXResult(ue=uei[cut], ds=dsi[cut], th=thi[cut], ts=tsi[cut],
                      dc=dci[cut], cf=cfi[cut], cd=cdi[cut], ct=cti[cut],
                      hk=hki[cut], ph=phi[cut])
