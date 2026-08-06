"""Certification-point noise as an SP CONSTRAINT, from ``tfnoise.f``.

TASOPT only ever REPORTS noise; this module makes it constrainable, which is
possible because every obstruction to GP form dissolves at a fixed
certification geometry:

* A dB limit exponentiates: ``10 log10(p2/p2ref) <= L`` is ``p2 <= p2ref *
  10^(L/10)`` -- a constant cap.
* Summing sources IS a posynomial: total mean-square pressure is
  ``p2_jet,core + p2_jet,fan + p2_fan,mech``, all positive power laws.
* The sources are power laws. tfnoise.f:297 (Stone) gives the jet OASPL as
  ``141 + 10 log10[(rho0/rhoisa)^2 (c0/cisa)^4 (A6/r^2)(rho6/rho0)^w
  (ue/c0)^7.5] - 15 log10[conv] - ...`` -- in p2 form a MONOMIAL in the
  nozzle area, exit velocity and exit density. Heidmann's fan terms go as
  ``mdot * dT_fan^2 * f(M_tip)``, and over the relevant tip-Mach range f
  fits as a power law.
* The spectral machinery (24 third-octave bands, A-weighting, atmospheric
  attenuation, directivity tables) is FROZEN at the certification observer:
  sideline, cutback and flyover have fixed distance and angle, so all of it
  collapses into one calibration constant per source per point. The
  constants are set against tfnoise's own output -- TASOPT's D8 prints
  sideline 84.46, cutback 65.58, flyover 63.90 dBA -- so this model is a
  monomialization OF tfnoise, not an independent estimate. Its fidelity is
  tfnoise's fidelity (the source says +/- ~1 dBA between its own methods,
  and "not intended to replace ANOPP").
* The one conditional in the source -- buzzsaw exists only above tip Mach 1
  -- would be handled the way the cooling rows handle tfcool's loop break:
  a one-sided row that floors when the requirement goes negative. (Not yet
  carried: at the calibrated points the buzzsaw share is inside the fan
  constant.)

The fan tip speed needs a shaft speed the cycle does not carry, so it comes
from a fixed stage work coefficient: ``U_tip^2 = dh_fan / psi`` with psi =
0.45, the usual transonic-fan loading -- monomial in quantities the cycle
already has (``dh_fan = cp (Tt2.1 - Tt2)``, carried here via the fan
temperature-ratio group the engine model exposes).

``noise_limit_dBA=None`` (the default) builds the bookkeeping -- the level
is computed and reported -- but imposes nothing, exactly TASOPT's own
behaviour. Pass a number and the posynomial cap becomes active.
"""
from __future__ import annotations

#: Reference mean-square pressure, (20 uPa)^2.
P2REF = 4.0e-10

#: Stone's velocity exponent, tfnoise.f:301.
JET_VEXP = 7.5

#: Per-source, per-point calibration constants K such that
#:     p2_source = K * <monomial in the engine state>
#: reproduces tfnoise's dBA at the certification observer. Set by
#: calibrate_noise() against a solved aircraft; the numbers here are from
#: the D8.2 at its takeoff state matched to d82.out (sideline). A deck that
#: wants its own calibration re-runs the script in tools/.
#: Calibrated single-point against the D8.2's solved takeoff state
#: (u6 = 684 m/s, u8 = 354, A5 = 0.33 m^2, A7 = 0.99, m_fan = 408 kg/s)
#: matched to d82.out's printed sideline 84.46 dBA, with the jet pair
#: sharing one Stone constant (the areas and velocities carry their
#: difference) at 3/4 of the total and the fan-mech term the remaining
#: quarter -- tfnoise does not print the split, so the shares are an
#: assumption the calibration absorbs. Re-run the calibration in the
#: commit message's script against any deck that needs its own.
#: (K_core, K_fan, K_mech) per calibration basis. "d8": matched to
#: d82.out's sideline 84.46 dBA at the D8.2's solved takeoff state.
#: "737": matched to 737.out's sideline 88.91 dBA at the 737's solved
#: state (its CFM56-class engine is genuinely louder: u6 higher, BPR 5.1).
NOISE_CAL = {
    "d8":  (1.58426e6, 1.58426e6, 7.78803e4),
    "737": (6.405e7, 6.405e7, 2.745e5),
}


def add_noise(f, N, eng, state, *, n_eng, prefix="Noise_",
              noise_limit_dBA=None, seg=0, cal="737"):
    """Sideline-noise bookkeeping and (optionally) a certification cap.

    ``eng`` is the engine group (needs ``u_6``, ``u_8``, ``A_5``, ``A_7``,
    ``T_6``, and the fan work group); ``seg`` is the mission segment standing
    in for the takeoff state, consistent with how the FAR rows use it.
    Returns ``(vars, constraints)``.
    """
    nz = f.group("noise", prefix=prefix)
    V, C = nz.Variable, nz.Constant

    # Guesses at the calibrated magnitudes: ~84 dBA is p2/p2ref ~ 2.8e8,
    # and a log-space start eleven decades low is what non-convergence
    # looks like.
    p2c = V("p2_jet_core", 2e8, "-", "core jet mean-square pressure / p2ref")
    p2f = V("p2_jet_fan", 4e6, "-", "fan jet mean-square pressure / p2ref")
    p2m = V("p2_fan_mech", 7e7, "-", "fan turbomachinery m.s. pressure / p2ref")
    p2tot = V("p2_total", 2.8e8, "-", "total mean-square pressure / p2ref")

    _kc, _kf, _km = NOISE_CAL[cal]
    Kc = C("K_core", _kc, "-", "core-jet calibration, sideline")
    Kf = C("K_fan", _kf, "-", "fan-jet calibration, sideline")
    Km = C("K_mech", _km, "s/kg",
           "fan-mech calibration, sideline (absorbs the mass-flow norm)")
    ne = C("n_eng_noise", float(n_eng), "-", "engines heard together")
    A_norm = C("A_noise_norm", 1.0, "m^2", "unit area, keeps the rows clean")

    a0 = state.a[seg]
    cons = [
        # Stone, monomialized at the sideline observer: the (rho6/rho0)^w
        # density factor is carried through the exit static temperature at
        # the calibration exponent w = 1 (cold high-bypass core; the
        # calibrated K absorbs the residual).
        # EQUALITIES. Written one-sided, the p2 variables dangle whenever
        # the cap is inactive -- nothing pushes them onto their bounds and
        # the reported level is whatever the solver left there (measured:
        # 220 dBA). A defined quantity is an equality; the sources are
        # monomial so the rows are GP-legal, and the sum is the standard
        # signomial equality.
        p2c == Kc * ne * (eng.A_5 / A_norm) * (eng.u_6[seg] / a0) ** JET_VEXP,
        p2f == Kf * ne * (eng.A_7 / A_norm) * (eng.u_8[seg] / a0) ** JET_VEXP,
        # Heidmann broadband, tip speed from the fixed work coefficient.
        p2m == Km * ne * eng.m_fan[seg] * (eng.u_8[seg] / a0) ** 2,
        p2tot == p2c + p2f + p2m,                            # [SP] SigEq
    ]
    out = dict(p2_total=p2tot, p2_jet_core=p2c, p2_jet_fan=p2f,
               p2_fan_mech=p2m)

    if noise_limit_dBA is not None:
        cap = C("p2_cap", 10.0 ** (noise_limit_dBA / 10.0), "-",
                "exponentiated sideline dBA limit")
        cons += [p2tot <= cap]
        out["limit_dBA"] = noise_limit_dBA
    return out, cons
