"""Walking a tank along a mission -- ``tanktools.jl``.

The tank models elsewhere in :mod:`tasopt_py.cryo` answer point questions:
what does this vessel weigh, what leaks into it *here*. A mission is not a
point. Fuel drains, the altitude and Mach change, the heat leak changes with
them, and the tank pressure follows all of it.

This module supplies the two things a time integration needs: the fuel demand
and the heat rate, each as a function of **time** rather than of mission
point.

Two different interpolations, and the difference is deliberate
--------------------------------------------------------------
* **Fuel flow is interpolated exponentially**, ``m0 exp(k (t - t0))``.
  Fuel flow falls roughly geometrically through a cruise as the aircraft
  lightens, so a straight line between mission points would understate it in
  the middle of every segment.
* **Heat rate is interpolated linearly.** It tracks altitude and speed, which
  the mission already discretises finely enough.

The exponential form cannot handle a zero, and the two ends fail differently.
A segment that *starts* at zero is guarded explicitly and reads zero
throughout. A segment that *ends* at zero is **not** guarded, and survives
only because ``log(0)`` is ``-Inf`` in Julia, making ``exp(-Inf dt)`` zero for
any ``dt > 0``; Python raises, so that case is written out here to give the
same answer. Either way, touching zero anywhere in a segment zeroes the whole
segment -- which is the descent-idle case, and a real understatement of fuel
burn there.

Nothing here exists in TASOPT 2.16.

Verified against TASOPT.jl; see ``tests/test_mission_tank.py``.
"""
from __future__ import annotations

import math

__all__ = ["fuel_flow_at", "heat_rate_at", "mission_heat_rates"]


def _bracket(times, t: float):
    """``(i, t0, tf)`` for the interval containing ``t``, or the exact index.

    Returns ``(i, None, None)`` when ``t`` lands exactly on a station, which
    both interpolations short-circuit on.
    """
    for i, ti in enumerate(times):
        if ti == t:
            return i, None, None
    for i in range(len(times) - 1):
        if times[i] <= t < times[i + 1]:
            return i, times[i], times[i + 1]
    raise ValueError(
        f"time {t} is outside the mission, which runs {times[0]} to "
        f"{times[-1]}. The reference's loop simply falls through and returns "
        "`nothing`, which then fails as a type error somewhere else.")


def fuel_flow_at(t: float, times, mdots) -> float:
    """Fuel mass flow into the engines at time ``t``, kg/s per tank.

    Exponentially interpolated between mission points, because fuel flow
    falls roughly geometrically as the aircraft lightens.
    """
    i, t0, tf = _bracket(times, t)
    if t0 is None:
        return mdots[i]
    m0, mf = mdots[i], mdots[i + 1]
    if m0 <= 0.0:
        # The reference's explicit guard: a segment that *starts* at zero
        # contributes nothing over its whole length, even if it ends at full
        # flow.
        return 0.0
    if mf <= 0.0:
        # Not guarded in the reference, and it survives only by IEEE luck:
        # `log(0)` is -Inf in Julia, so `k` is -Inf and `exp(-Inf dt)` is
        # zero for any dt > 0. Python raises on log(0), so the same answer is
        # written out here. A segment that *ends* at zero therefore reads
        # zero everywhere except its first instant.
        return 0.0
    k = math.log(mf / m0) / (tf - t0)
    return m0 * math.exp(k * (t - t0))


def heat_rate_at(t: float, times, Qs) -> float:
    """Heat rate into the tank at time ``t``, W. Linearly interpolated."""
    i, t0, tf = _bracket(times, t)
    if t0 is None:
        return Qs[i]
    return Qs[i] + (Qs[i + 1] - Qs[i]) / (tf - t0) * (t - t0)


def mission_heat_rates(params, altitudes, machs, qfac: float = 1.0) -> list:
    """The heat rate at every mission point, W.

    Precomputed once and interpolated thereafter, because each one is a
    Newton solve of the whole series heat path -- doing it per integration
    step would dominate the cost.
    """
    from .thermal import tank_heat_leak

    out = []
    for z, M in zip(altitudes, machs):
        p = _at(params, z=z, Mair=M)
        out.append(tank_heat_leak(p, qfac))
    return out


def _at(params, **kw):
    """A copy of a ThermalParams with some fields replaced."""
    from dataclasses import replace
    return replace(params, **kw)
