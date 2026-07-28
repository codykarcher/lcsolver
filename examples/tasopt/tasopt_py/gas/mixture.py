"""Gas mixture thermodynamics — port of TASOPT ``src/gascalc.f``.

These are the workhorses the turbofan cycle is built from: given a mixture of
constituents by mass fraction, find the state after a specified pressure
ratio, enthalpy change, or combustion.

Everything here is *thermally perfect but not calorically perfect* — cp
varies with temperature — so the familiar closed-form isentropic relations do
not apply. Each routine instead solves a scalar Newton iteration on
temperature against the tabulated properties. Each docstring below quotes the
constant-cp equivalent from the Fortran comments, which is the quickest way
to see what the routine is doing.

Constituent ordering is fixed by the Fortran and assumed throughout:

    1 = N2, 2 = O2, 3 = CO2, 4 = H2O, 5 = Ar

``alpha[i]`` is the mass fraction of constituent ``i+1`` in that ordering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from tasopt_py.gas.properties import gasfun, gaschem

ITMAX = 10
TTOL = 1.0e-6

# molar weights of C, H, O, N atoms (gasfuel in gascalc.f)
W_CHON = (12.01078, 1.00795, 15.99943, 14.00672)

# constituent indices within alpha/beta/gamma (1-based, as the Fortran)
I_N2, I_O2, I_CO2, I_H2O = 1, 2, 3, 4


@dataclass(frozen=True)
class MixState:
    s: float
    s_t: float
    h: float
    h_t: float
    cp: float
    r: float


class ConvergenceError(RuntimeError):
    """Newton iteration failed to converge.

    The Fortran prints a message and carries on with an unconverged value.
    Raising instead means a silently wrong cycle calculation cannot propagate
    unnoticed; callers that want the old behaviour can catch this.
    """


def gassum(alpha, n: int, t: float) -> MixState:
    """Mixture properties at temperature *t*, mass fractions ``alpha[0:n]``."""
    s = h = cp = r = s_t = h_t = 0.0
    for i in range(1, n + 1):
        g = gasfun(i, t)
        a = alpha[i - 1]
        s += g.s * a
        h += g.h * a
        cp += g.cp * a
        r += g.r * a
        s_t += g.s_t * a
        h_t += g.h_t * a
    return MixState(s=s, s_t=s_t, h=h, h_t=h_t, cp=cp, r=r)


def gassumd(alpha, n: int, t: float) -> tuple:
    """``gassum`` plus dcp/dT, which the Fortran takes by central difference.

    The step is hard-coded at dt = 0.01 K in the original and kept here: it is
    not a tunable, and changing it would change results in the last digits.
    """
    st = gassum(alpha, n, t)
    dt = 0.01
    cp_t = 0.0
    for i in range(1, n + 1):
        cp1 = gasfun(i, t - dt).cp
        cp2 = gasfun(i, t + dt).cp
        cp_t += (cp2 - cp1) * 0.5 / dt * alpha[i - 1]
    return st, cp_t


def gasfuel(ifuel: int, n: int) -> list:
    """Mass-fraction changes of air constituents from burning fuel *ifuel*.

    Balances the reaction ``fuel + O2 -> CO2 + H2O (+ N2)`` by atom count and
    returns the change per unit fuel mass. The left side (O2 consumed) is
    negative, the right side (products) positive.
    """
    nchon = gaschem(ifuel)
    wc, wh, wo, wn = W_CHON

    wfuel = wc * nchon[0] + wh * nchon[1] + wo * nchon[2] + wn * nchon[3]

    wn2 = (wn * 2.0) * (nchon[3] * 0.5)
    wco2 = (wc + wo * 2.0) * nchon[0]
    wh2o = (wo + wh * 2.0) * (nchon[1] * 0.5)
    wo2 = (wo * 2.0) * (nchon[0] + nchon[1] * 0.25 - nchon[2] * 0.5)

    gamma = [0.0] * n
    gamma[I_N2 - 1] = wn2 / wfuel
    gamma[I_O2 - 1] = -wo2 / wfuel
    gamma[I_CO2 - 1] = wco2 / wfuel
    gamma[I_H2O - 1] = wh2o / wfuel
    return gamma


def gas_tset(alpha, n: int, hspec: float, tguess: float) -> float:
    """Temperature at a specified enthalpy.  Constant-cp equivalent::

        t = (hspec - hf)/cp
    """
    t = tguess
    for _ in range(ITMAX):
        st = gassum(alpha, n, t)
        dt = -(st.h - hspec) / st.h_t
        if abs(dt) < TTOL:
            return t
        t += dt
    raise ConvergenceError(f"gas_tset: convergence failed, dT = {dt}")


def gas_prat(alpha, n: int, po, to, ho, so, cpo, ro,
             pi: float, epol: float) -> tuple:
    """State change across a specified pressure ratio *pi*.

    ``epol`` is the polytropic efficiency for a compression, or its reciprocal
    for an expansion. Constant-cp equivalent::

        g = cp/(cp-r);  gexp = (g-1)/(g*epol);  tau = pi**gexp
        p = po*pi;  t = to*tau;  (h-hf) = (ho-hf)*tau

    Solves ``(s - so)/r = log(pi)/epol`` for t by Newton, which is the
    variable-cp statement of the same polytropic relation.
    """
    gexp = ro / (cpo * epol)
    t = to * pi**gexp
    pile = math.log(pi) / epol

    for _ in range(ITMAX):
        st = gassum(alpha, n, t)
        res = (st.s - so) / st.r - pile
        dt = -res / (st.s_t / st.r)
        if abs(dt) < TTOL:
            p = po * pi
            return p, t, st.h, st.s, st.cp, st.r
        t += dt
    raise ConvergenceError(
        f"gas_prat: convergence failed, dT = {dt} (to={to}, pi={pi})")


def gas_mach(alpha, n: int, po, to, ho, so, cpo, ro,
             mo: float, m: float, epol: float) -> tuple:
    """State change across a specified *Mach number* change.

    Constant-cp equivalent::

        g = cp/(cp-r);  gexp = (g-1)/(g*epol)
        tau = (1 + 0.5*(g-1)*mo**2) / (1 + 0.5*(g-1)*m**2)
        pi = tau**(1/gexp);  p = po*pi;  t = to*tau

    The variable-cp statement is conservation of stagnation enthalpy,
    ``h + u^2/2 = ho + uo^2/2`` with ``u^2 = m^2 * cp*r/(cp-r) * t``, solved
    for ``t`` by Newton. Note the residual's derivative carries a ``cp_t``
    term that the source sets to zero with the comment "could evaluate this
    from the cp(T) splines (later)" -- so the Newton step uses an approximate
    Jacobian. That costs iterations, not accuracy: the residual itself is
    exact, so the converged root is the true one. Reproduced as written.
    """
    uosq = mo ** 2 * cpo * ro / (cpo - ro) * to

    # Constant-gamma initial guess.
    t = (to * (1.0 + 0.5 * ro / (cpo - ro) * mo ** 2)
         / (1.0 + 0.5 * ro / (cpo - ro) * m ** 2))

    for _ in range(ITMAX):
        st = gassum(alpha, n, t)
        cp_t = 0.0                      # see the note above

        usq = m ** 2 * st.cp * st.r / (st.cp - st.r) * t
        usq_t = m ** 2 * st.cp * st.r / (st.cp - st.r)
        usq_cp = m ** 2 * st.r / (st.cp - st.r) * t - usq / (st.cp - st.r)

        res = st.h + 0.5 * usq - ho - 0.5 * uosq
        res_t = st.h_t + 0.5 * (usq_t + usq_cp * cp_t)
        dt = -res / res_t

        if abs(dt) < TTOL:
            p = po * math.exp(epol * (st.s - so) / st.r)
            return p, t, st.h, st.s, st.cp, st.r
        t += dt
    raise ConvergenceError(
        f"gas_mach: convergence failed, dT = {dt} (mo={mo}, m={m})")


def gas_delh(alpha, n: int, po, to, ho, so, cpo, ro,
             delh: float, epol: float) -> tuple:
    """State change across a specified enthalpy change *delh*.

    Constant-cp equivalent::

        t - to = delh/cp;  tau = t/to;  pi = tau**(1/gexp);  p = po*pi
    """
    t = to + delh / cpo
    for _ in range(ITMAX):
        st = gassum(alpha, n, t)
        dt = -(st.h - ho - delh) / st.h_t
        if abs(dt) < TTOL:
            p = po * math.exp(epol * (st.s - so) / st.r)
            return p, t, st.h, st.s, st.cp, st.r
        t += dt
    raise ConvergenceError(f"gas_delh: convergence failed, dT = {dt}")


def gas_burn(alpha, beta, gamma, n: int, ifuel: int,
             to: float, tf: float, t: float) -> tuple:
    """Fuel/air mass fraction and product composition for combustion.

    Air enters at ``to``, fuel at ``tf``, products leave at ``t``. Returns
    ``(f, lambda)`` where f is the fuel/air mass ratio and lambda the product
    mass fractions.

    The energy balance is ``f = (ha - ho)/(hf - hc)``: enthalpy needed to heat
    the air from to to t, over the enthalpy released per unit fuel.
    """
    nm = n - 1
    ho = gassum(alpha, nm, to).h
    ha = gassum(alpha, nm, t).h
    hc = gassum(gamma, nm, t).h
    hf = gassum(beta, nm, tf).h

    # The fuel itself is constituent n, which gassum cannot reach by index --
    # it only sums 1..nm -- so its contribution is added explicitly.
    hf += gasfun(ifuel, tf).h * beta[n - 1]

    f = (ha - ho) / (hf - hc)
    lam = [(alpha[i] + f * gamma[i]) / (1.0 + f) for i in range(n)]
    return f, lam


__all__ = [
    "gassum", "gassumd", "gasfuel", "gas_tset", "gas_prat", "gas_delh",
    "gas_burn", "gas_mach", "MixState", "ConvergenceError",
    "I_N2", "I_O2", "I_CO2", "I_H2O", "W_CHON",
]
