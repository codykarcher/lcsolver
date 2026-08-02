"""Generate SP-compatible thermo fits from pycycle's janaf tables.

Writes ``components/turbofan/sp_thermo.py``: a pure-data module of fitted
constants the SP cycle rows consume. Everything here is derived from the SAME
janaf coefficient tables pycycle's CEA solver uses, so the SP cycle and the
truth harness share one thermodynamic source and disagreements between them
are formulation errors, not data errors.

What is fitted, and why in these shapes
---------------------------------------
* ``h_j(T)`` per species, molar, J/mol: cubic in T over the hot-path range.
  The NASA-9 form has ln(T)/T and 1/T^2 terms a posynomial cannot carry;
  a plain cubic refit stays within ~0.1% over the range we use and its mixed
  signs are exactly what signomial equalities exist for.
* ``h_air(T)`` per kg over the compressor range, same shape.
* ``psi(T) = exp(sum_j x_j S0_j(T)/R)``, the entropy exponential, fitted as a
  POSYNOMIAL in T. This is the key trick: along a frozen-composition
  isentrope  sum x_j dS0 = R dln(P), so  P2/P1 == psi(T2)/psi(T1) EXACTLY --
  a pure monomial row connecting pressure ratio to temperatures, with all of
  the variable-cp physics buried in the 1-D psi fit. ln(psi) is convex in
  ln(T) (cp rises with T), which is the condition for a posynomial fit to
  work to arbitrary accuracy. Fitted by NNLS over a fixed exponent basis.
* ``psi_vit(T; far)`` and ``h_vit(T; far)`` for the turbine side, generated
  from EQUILIBRIUM compositions on a (T, far) grid along a representative
  turbine expansion pressure line, so dissociation/recombination enthalpy is
  in the surface rather than ignored by a frozen assumption.
* ``Kp_r(T)`` per reaction for the burner's mass-action rows, fitted as
  monomial-or-posynomial in T over the burner temperature band, whichever
  side of log-log convexity the reaction falls on.

The module also carries an independent Gibbs-minimization equilibrium solver
(Lagrange-Newton on ln n_j) used to make the grids and to cross-check the
mass-action formulation against pycycle's burner products.

Run:  python -m components.turbofan.truth.fits   (from examples/spcomparisons)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from pycycle.thermo.cea import species_data
from pycycle.constants import CEA_AIR_COMPOSITION

R_UNIV = 8.314462618  # J/(mol K)
P_REF_BAR = 1.01325   # CEA reference pressure, atm expressed in bar

JD = species_data.janaf

#: The SP burner's species. Selection: everything that reaches ~1e-5 mole
#: fraction in jet-A/air products at 1900 K / 30 bar. N is out (~1e-9);
#: CH4/NH3/etc. are out (equilibrium at these temperatures leaves no fuel
#: species).
SPECIES = ['N2', 'O2', 'Ar', 'CO2', 'H2O', 'CO', 'H2', 'OH', 'NO', 'O', 'H']
ELEMENTS = ['C', 'H', 'N', 'O', 'Ar']

#: Mass-action reactions for the burner rows, written as
#: product <-> sum(nu_b * base). Kp is for the FORMATION direction shown,
#: with pressures in bar.
REACTIONS = {
    'CO':  {'from': {'CO2': 1.0, 'O2': -0.5}},   # CO2 -> CO + 1/2 O2
    'H2':  {'from': {'H2O': 1.0, 'O2': -0.5}},   # H2O -> H2 + 1/2 O2
    'OH':  {'from': {'H2O': 0.5, 'O2': 0.25}},   # 1/2 H2O + 1/4 O2 -> OH
    'NO':  {'from': {'N2': 0.5, 'O2': 0.5}},     # 1/2 N2 + 1/2 O2 -> NO
    'O':   {'from': {'O2': 0.5}},                # 1/2 O2 -> O
    'H':   {'from': {'H2': 0.5}},                # 1/2 H2 -> H
}


# ---------------------------------------------------------------------------
# Exact janaf evaluation
# ---------------------------------------------------------------------------

def _coeffs(sp, T):
    d = JD.products[sp]
    ranges = d['ranges']
    for i in range(len(ranges) - 1):
        if T <= ranges[i + 1] or i == len(ranges) - 2:
            return np.asarray(d['coeffs'][i], dtype=float)
    raise ValueError(f"{sp}: T={T} outside janaf ranges {ranges}")


def h_molar(sp, T):
    """Standard-state molar enthalpy, J/mol (includes formation)."""
    a = _coeffs(sp, T)
    HRT = (-a[0] / T**2 + a[1] / T * np.log(T) + a[2] + a[3] * T / 2.
           + a[4] * T**2 / 3. + a[5] * T**3 / 4. + a[6] * T**4 / 5.
           + a[7] / T)
    return HRT * R_UNIV * T


def s0_molar(sp, T):
    """Standard-state molar entropy at 1 atm, J/(mol K)."""
    a = _coeffs(sp, T)
    SR = (-a[0] / (2 * T**2) - a[1] / T + a[2] * np.log(T) + a[3] * T
          + a[4] * T**2 / 2. + a[5] * T**3 / 3. + a[6] * T**4 / 4. + a[8])
    return SR * R_UNIV


def g0_molar(sp, T):
    return h_molar(sp, T) - T * s0_molar(sp, T)


WT = {sp: JD.products[sp]['wt'] for sp in SPECIES}  # g/mol
ELEM = {sp: dict(JD.products[sp]['elements']) for sp in SPECIES}


# ---------------------------------------------------------------------------
# Compositions
# ---------------------------------------------------------------------------

def air_moles_per_kg():
    """kmol of each air species per kg of air, from pycycle's element b0."""
    b = CEA_AIR_COMPOSITION            # kmol element / kg air
    nCO2 = b['C']
    nO2 = (b['O'] - 2.0 * nCO2) / 2.0
    nN2 = b['N'] / 2.0
    nAr = b['Ar']
    return {'N2': nN2, 'O2': nO2, 'Ar': nAr, 'CO2': nCO2}


# Jet-A(g) as pycycle burns it: the janaf reactant entry is just the element
# ratio C12 H23. The fuel stream's enthalpy is an INPUT to pycycle's
# ThermoAdd defaulting to 0 Btu/lbm on the CEA absolute scale, and the HBTF
# example leaves the default -- verified against truth/data/cfm56_class.json,
# where h4*W4 - h3*W3 = 0 to machine precision (the heat release is entirely
# the products' lower formation enthalpy). The SP burner must copy this
# convention or its T4/FAR relation will be off by the difference between 0
# and Jet-A's real formation enthalpy.
def fuel_data():
    r = JD.reactants['Jet-A(g)']
    elem = dict(r)  # {'C': 12.0, 'H': 23.0}
    wt = sum(cnt * JD.element_wts[e] for e, cnt in elem.items())
    return {'elements': elem, 'wt': float(wt), 'h_J_kg': 0.0}


def mix_elements_per_kg(far):
    """kmol of each element per kg of (air + fuel) mixture at fuel-air
    ratio ``far``."""
    fd = fuel_data()
    air = CEA_AIR_COMPOSITION
    out = {}
    w_air = 1.0 / (1.0 + far)
    w_fuel = far / (1.0 + far)
    for e in ELEMENTS:
        b_air = air.get(e, 0.0)
        b_fuel = fd['elements'].get(e, 0.0) / fd['wt']  # kmol/kg fuel
        out[e] = w_air * b_air + w_fuel * b_fuel
    return out


# ---------------------------------------------------------------------------
# Equilibrium (Gibbs minimization, Lagrange-Newton on ln n)
# ---------------------------------------------------------------------------

def equilibrium(T, P_bar, far, tol=1e-12, maxiter=200):
    """Equilibrium composition of jet-A/air products.

    Returns kmol of each SPECIES per kg mixture. Minimizes
    G = sum n_j (g0_j + RT ln(n_j P / n_tot Pref)) subject to element
    conservation, by Newton on the standard mass-action fixed point: express
    every minor species through the reaction table and iterate the majors.
    Simple, and adequate for grid generation -- checked against pycycle's
    burner products in __main__.
    """
    b = mix_elements_per_kg(far)
    RT = R_UNIV * T
    mu0 = {sp: g0_molar(sp, T) for sp in SPECIES}

    # complete combustion start
    nCO2 = b['C']
    nH2O = b['H'] / 2.0
    nO2 = max((b['O'] - 2 * nCO2 - nH2O) / 2.0, 1e-8)
    n = {sp: 1e-12 for sp in SPECIES}
    n.update({'N2': b['N'] / 2.0, 'O2': nO2, 'Ar': b['Ar'],
              'CO2': nCO2, 'H2O': nH2O})

    majors = ['N2', 'O2', 'CO2', 'H2O']
    for it in range(maxiter):
        ntot = sum(n.values())
        # minor species from mass action at current majors
        lnK = {}
        for sp, r in REACTIONS.items():
            dG = mu0[sp] - sum(c * mu0[base] for base, c in r['from'].items())
            lnK[sp] = -dG / RT
        for sp, r in REACTIONS.items():
            ln_n = (lnK[sp]
                    + sum(c * np.log(n[base] * P_bar / (ntot * P_REF_BAR))
                          for base, c in r['from'].items())
                    - np.log(P_bar / (ntot * P_REF_BAR)))
            n[sp] = np.exp(ln_n)
        # re-solve element balances for the majors, holding minors
        # C: CO2 + CO ; H: 2 H2O + 2 H2 + OH + H ; N: 2 N2 + NO ;
        # O: 2 O2 + 2 CO2 + H2O + CO + OH + NO + O
        nCO2 = b['C'] - n['CO']
        nH2O = (b['H'] - 2 * n['H2'] - n['OH'] - n['H']) / 2.0
        nN2 = (b['N'] - n['NO']) / 2.0
        nO2 = (b['O'] - 2 * nCO2 - nH2O
               - n['CO'] - n['OH'] - n['NO'] - n['O']) / 2.0
        new = {'CO2': max(nCO2, 1e-14), 'H2O': max(nH2O, 1e-14),
               'N2': max(nN2, 1e-14), 'O2': max(nO2, 1e-14)}
        delta = max(abs(new[k] - n[k]) / max(new[k], 1e-14) for k in new)
        n.update(new)
        if delta < tol:
            break
    n['Ar'] = b['Ar']
    return n


def h_mix_per_kg(n, T):
    """J/kg for a composition dict (kmol/kg) at T."""
    return sum(nj * h_molar(sp, T) for sp, nj in n.items()) * 1000.0


def psi_ln(n, T):
    """ln(psi) = sum_j x_j S0_j(T)/R for composition ``n`` (frozen)."""
    ntot = sum(n.values())
    return sum((nj / ntot) * s0_molar(sp, T) for sp, nj in n.items()) / R_UNIV


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_cubic(Ts, ys):
    """y ~= c0 + c1 T + c2 T^2 + c3 T^3, max |rel residual| reported."""
    A = np.vstack([np.ones_like(Ts), Ts, Ts**2, Ts**3]).T
    c, *_ = np.linalg.lstsq(A, ys, rcond=None)
    res = (A @ c - ys) / np.maximum(np.abs(ys), 1e-30)
    return c, float(np.max(np.abs(res)))


def fit_posy_1d(Ts, ys, exps):
    """y ~= sum_k c_k T^a_k with c_k >= 0 (NNLS, relative weighting).

    Fitted on y scaled by its geometric mean -- K(T) spans decades and raw
    NNLS on that dynamic range hits its iteration cap.
    """
    scale = float(np.exp(np.mean(np.log(ys))))
    yn = ys / scale
    A = np.vstack([Ts**a for a in exps]).T
    W = 1.0 / yn
    try:
        c, _ = nnls(A * W[:, None], yn * W, maxiter=100 * len(exps))
    except RuntimeError:
        return [], float('inf')
    res = (A @ c - yn) / yn
    keep = c > 0
    return ([(float(ck * scale), float(a))
             for ck, a in zip(c[keep], np.array(exps)[keep])],
            float(np.max(np.abs(res))))


def fit_signomial_1d(Ts, ys, exps):
    """y ~= sum_k c_k T^a_k with FREE-sign c_k (plain LSQ, relative
    weighting). Signomial equalities carry mixed signs anyway (that is what
    they are for), so psi fits need not be posynomial -- and free signs buy
    two orders of magnitude of residual over NNLS on these curves."""
    A = np.vstack([Ts**a for a in exps]).T
    W = 1.0 / ys
    c, *_ = np.linalg.lstsq(A * W[:, None], np.ones_like(ys), rcond=None)
    res = (A @ c - ys) / ys
    return ([(float(ck), float(a)) for ck, a in zip(c, exps)
             if abs(ck) > 0],
            float(np.max(np.abs(res))))


def fit_signomial_2d(X1, X2, ys, exps1, exps2):
    """y ~= sum c_k X1^a_k X2^b_k, free-sign LSQ, relative weighting."""
    terms = [(a, b) for a in exps1 for b in exps2]
    A = np.vstack([X1**a * X2**b for a, b in terms]).T
    W = 1.0 / ys
    c, *_ = np.linalg.lstsq(A * W[:, None], np.ones_like(ys), rcond=None)
    res = (A @ c - ys) / ys
    out = [(float(ck), float(a), float(b))
           for ck, (a, b) in zip(c, terms) if abs(ck) > 0]
    return out, float(np.max(np.abs(res)))


def fit_posy_2d(X1, X2, ys, exps1, exps2):
    """y ~= sum c_k X1^a_k X2^b_k over the (a, b) grid, NNLS."""
    terms = [(a, b) for a in exps1 for b in exps2]
    A = np.vstack([X1**a * X2**b for a, b in terms]).T
    W = 1.0 / ys
    c, _ = nnls(A * W[:, None], ys * W)
    res = (A @ c - ys) / ys
    keep = c > 0
    out = [(float(ck), float(a), float(b))
           for ck, (a, b) in zip(c, terms) if ck > 0]
    return out, float(np.max(np.abs(res)))


def fit_monomial(Ts, ys):
    """y ~= C T^a (log-log linear least squares)."""
    A = np.vstack([np.ones_like(Ts), np.log(Ts)]).T
    c, *_ = np.linalg.lstsq(A, np.log(ys), rcond=None)
    fit = np.exp(A @ c)
    return (float(np.exp(c[0])), float(c[1])), float(np.max(np.abs(fit / ys - 1)))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(path=None):
    import io, pprint, pathlib
    out = {}
    report = []

    # -- species molar enthalpies over the hot path ------------------------
    Ts_hot = np.linspace(800.0, 2400.0, 161)
    out['SPECIES_H'] = {}
    for sp in SPECIES:
        ys = np.array([h_molar(sp, T) for T in Ts_hot])
        c, r = fit_cubic(Ts_hot, ys)
        # relative residual is meaningless when h crosses zero; report vs the
        # enthalpy SPAN instead
        span = ys.max() - ys.min()
        res = float(np.max(np.abs(np.polyval(c[::-1], Ts_hot) - ys))) / span
        out['SPECIES_H'][sp] = {'c': [float(x) for x in c],
                                'range_K': [800.0, 2400.0]}
        report.append(f"h {sp:4s} cubic 800-2400K  max|res|/span {res:.2e}")

    out['SPECIES_WT'] = {sp: float(WT[sp]) for sp in SPECIES}
    out['SPECIES_ELEM'] = {sp: {k: float(v) for k, v in ELEM[sp].items()}
                           for sp in SPECIES}

    # -- air: enthalpy + psi over the compressor range ---------------------
    air = air_moles_per_kg()
    Ts_cold = np.linspace(200.0, 1150.0, 191)
    h_air = np.array([h_mix_per_kg(air, T) for T in Ts_cold])
    # shift so the fitted quantity is positive (posynomial-friendly h is not
    # needed on the air side -- the rows use dh -- but a stable zero helps
    # the cubic's conditioning): reference to 200 K.
    h0 = h_air[0]
    c, _ = fit_cubic(Ts_cold, h_air - h0)
    res = float(np.max(np.abs(np.polyval(c[::-1], Ts_cold) - (h_air - h0)))
                / (h_air.max() - h_air.min()))
    out['AIR_H'] = {'c': [float(x) for x in c], 'h_ref_J_kg': float(h0),
                    'T_ref_K': 200.0, 'range_K': [200.0, 1150.0]}
    report.append(f"h air  cubic 200-1150K  max|res|/span {res:.2e}")

    lnpsi = np.array([psi_ln(air, T) for T in Ts_cold])
    lnpsi0 = psi_ln(air, 288.15)
    psi = np.exp(lnpsi - lnpsi0)
    terms, res = fit_signomial_1d(Ts_cold / 288.15, psi,
                                  exps=[3.0, 3.4, 3.8, 4.2, 4.6, 5.0])
    out['AIR_PSI'] = {'terms': terms, 'T_ref_K': 288.15,
                      'range_K': [200.0, 1150.0]}
    report.append(f"psi air signomial({len(terms)} terms) 200-1150K  "
                  f"max|res| {res:.2e}")

    # -- vitiated gas: h and psi over (T, far), equilibrium composition ----
    # Pressure along the grid follows a representative expansion line
    # anchored at 30 bar / 1900 K falling as T^ (gamma/(gamma-1) ~ 4): the
    # equilibrium composition's pressure sensitivity is second order, but
    # evaluating it ON the turbine line rather than at 1 bar keeps the
    # dissociation content honest.
    fars = np.linspace(0.012, 0.042, 7)
    Ts_vit = np.linspace(900.0, 2100.0, 41)
    Hgrid, PSIgrid, X1, X2 = [], [], [], []
    for far in fars:
        for T in Ts_vit:
            P = 30.0 * (T / 1900.0) ** 4.0
            n = equilibrium(T, P, far)
            Hgrid.append(h_mix_per_kg(n, T))
            PSIgrid.append(psi_ln(n, T))
            X1.append(T)
            X2.append(1.0 + far)
    Hgrid = np.array(Hgrid); PSIgrid = np.array(PSIgrid)
    X1 = np.array(X1); X2 = np.array(X2)

    lnpsi0 = psi_ln(equilibrium(1200.0, 10.0, 0.027), 1200.0)
    psi_v = np.exp(PSIgrid - lnpsi0)
    terms, res = fit_signomial_2d(X1 / 1200.0, X2, psi_v,
                                  exps1=[3.4, 3.8, 4.2, 4.6, 5.0],
                                  exps2=[0.0, 4.0, 8.0])
    out['VIT_PSI'] = {'terms': terms, 'T_ref_K': 1200.0,
                      'range_K': [900.0, 2100.0], 'far_range': [0.012, 0.042]}
    report.append(f"psi vit signomial({len(terms)} terms) 900-2100K  "
                  f"max|res| {res:.2e}")

    # h_vit: h(T, far) = (A(t) + far*B(t) + far^2*C(t))/(1+far), A/B/C
    # quartics in t = T/1200 (normalized -- at T^4 ~ 2e13 raw-T lstsq loses
    # the fit to rcond trimming). The far^2 block carries what dissociation
    # curvature the equilibrium grid has.
    Tb = lambda T: np.array([1.0, T / 1200.0, (T / 1200.0)**2,
                             (T / 1200.0)**3, (T / 1200.0)**4])
    Amat = []
    for T, w in zip(X1, X2):
        far = w - 1.0
        Amat.append(np.concatenate([Tb(T) / w, far * Tb(T) / w,
                                    far**2 * Tb(T) / w]))
    Amat = np.array(Amat)
    cAB, *_ = np.linalg.lstsq(Amat, Hgrid, rcond=None)
    span = Hgrid.max() - Hgrid.min()
    res = float(np.max(np.abs(Amat @ cAB - Hgrid))) / span
    out['VIT_H'] = {'cA': [float(x) for x in cAB[:5]],
                    'cB': [float(x) for x in cAB[5:10]],
                    'cC': [float(x) for x in cAB[10:]],
                    'T_ref_K': 1200.0,
                    'range_K': [900.0, 2100.0], 'far_range': [0.012, 0.042]}
    report.append(f"h vit  (A+far*B+far^2*C)/(1+far) quartics  "
                  f"max|res|/span {res:.2e}")

    # -- burner equilibrium constants --------------------------------------
    Ts_burn = np.linspace(1400.0, 2200.0, 33)
    out['KP'] = {}
    for sp, r in REACTIONS.items():
        Ks = []
        for T in Ts_burn:
            dG = g0_molar(sp, T) - sum(c * g0_molar(base, T)
                                       for base, c in r['from'].items())
            Ks.append(np.exp(-dG / (R_UNIV * T)))
        Ks = np.array(Ks)
        (C, a), res_m = fit_monomial(Ts_burn, Ks)
        # try posynomial in both directions; keep the best under 0.5%
        t_ref = 1800.0
        terms_p, res_p = fit_posy_1d(Ts_burn / t_ref, Ks,
                                     exps=np.linspace(0.0, 40.0, 41))
        terms_i, res_i = fit_posy_1d(Ts_burn / t_ref, 1.0 / Ks,
                                     exps=np.linspace(-40.0, 0.0, 41))
        best = min([('monomial', ((C, a),), res_m),
                    ('posy', terms_p, res_p),
                    ('posy_inv', terms_i, res_i)], key=lambda t: t[2])
        kind, terms, res = best
        out['KP'][sp] = {'kind': kind,
                         'terms': [list(map(float, t)) for t in terms],
                         'T_ref_K': (t_ref if kind != 'monomial' else 1.0),
                         'stoich': r['from'], 'range_K': [1400.0, 2200.0]}
        report.append(f"Kp {sp:3s} {kind:9s}({len(terms)} terms) "
                      f"max|res| {res:.2e} "
                      f"(mono {res_m:.1e} posy {res_p:.1e} inv {res_i:.1e})")

    out['FUEL'] = fuel_data()
    out['AIR_COMPOSITION_KMOL_KG'] = {k: float(v)
                                      for k, v in air_moles_per_kg().items()}
    out['R_UNIV'] = R_UNIV
    out['P_REF_BAR'] = P_REF_BAR

    # -- emit --------------------------------------------------------------
    if path is None:
        path = (pathlib.Path(__file__).resolve().parents[1] / 'sp_thermo.py')
    buf = io.StringIO()
    buf.write('"""SP-side thermo constants, GENERATED by '
              'components/turbofan/truth/fits.py.\n\n'
              'Do not edit by hand; rerun the generator. Derived from the '
              'same janaf tables\npycycle CEA uses. See fits.py for the '
              'shapes and why they are SP-compatible.\n"""\n\n')
    for key in ['SPECIES_H', 'SPECIES_WT', 'SPECIES_ELEM', 'AIR_H',
                'AIR_PSI', 'VIT_PSI', 'VIT_H', 'KP', 'FUEL',
                'AIR_COMPOSITION_KMOL_KG', 'R_UNIV', 'P_REF_BAR']:
        buf.write(f"{key} = ")
        buf.write(pprint.pformat(out[key], width=78))
        buf.write("\n\n")
    pathlib.Path(path).write_text(buf.getvalue())
    report.append(f"wrote {path}")
    return out, report


if __name__ == "__main__":
    import json, pathlib
    out, report = generate()
    print("\n".join(report))

    # Cross-check the equilibrium solver against pycycle's burner products
    # at the CFM56-class design point.
    data = pathlib.Path(__file__).resolve().parent / 'data/cfm56_class.json'
    if data.exists():
        d = json.loads(data.read_text())
        pt = d['points']['DESIGN']
        prods = pt.get('burner_products_kmol_per_kg')
        if prods:
            T4 = pt['performance']['T4_R'] / 1.8            # K
            P4 = pt['stations']['burner.Fl_O']['Pt_psi'] * 0.0689476  # bar
            far = pt['performance']['FAR']
            mine = equilibrium(T4, P4, far)
            print(f"\nequilibrium check vs pycycle burner "
                  f"(T={T4:.1f} K, P={P4:.2f} bar, FAR={far:.5f}):")
            print(f"  {'species':6s} {'mine':>12s} {'pycycle':>12s} ratio")
            for sp in SPECIES:
                pv = prods.get(sp)
                if pv is None or pv < 1e-12:
                    continue
                mv = mine[sp]
                print(f"  {sp:6s} {mv:12.5e} {pv:12.5e} {mv/pv:8.4f}")
