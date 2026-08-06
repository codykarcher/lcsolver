"""Airfoil drag polars, as swappable data.

Every profile-drag model in this study is a GP-compatible softmax-affine (SMA)
surrogate::

    cd**alpha  >=  sum_k  c_k * prod_i  x_i ** e_ki

with ``x = (Re, tau, M_perp, c_l)``. Dividing through gives a posynomial <= 1,
which is what makes it GP-compatible; it is the same form Hoburg's gpfit emits
and the same form Karcher's ``sfit`` returns as ``(A, b, alpha)`` via
``c_k = exp(alpha*b_k)``, ``e_k = alpha*A_k``.

Why this file exists
--------------------
The polar turned out to be the single most consequential modelling choice in
the whole aircraft, and it was buried as four hard-coded lines inside the wing.
Freeing wing sweep walked the model straight out of the fit's valid region,
where it under-predicts drag by 3-6x, and the optimizer promptly spent the
difference. Nothing in the model said so.

So the polar is data now, with its validity range attached, and swapping one is
a keyword rather than an edit. A refit done elsewhere drops in as a new entry.

The validity range is not decoration
------------------------------------
``m_perp_max`` is enforced as a constraint by the caller. Measured against the
C-series table the fit came from (``Tasopt2.16/air/C.air``, cl 0.6, t/c 0.12,
at the table's own Re_ref = 2e7):

    M_perp   0.700  0.740  0.760  0.780  0.785  0.810
    fit/tab   0.79   0.76   0.57   0.32   0.28   0.16

The table climbs from cd = 0.0078 at M 0.70 to 0.0245 at M 0.785 -- the 3.1x
transonic rise that makes a transport sweep its wing -- and the fit returns
0.0069 there. Past M_perp ~ 0.74 the surrogate is not a model of anything.
"""
from __future__ import annotations

import os as _os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Polar:
    """An SMA profile-drag surrogate, plus the range it may be used over.

    ``terms`` are ``(c, e_Re, e_tau, e_M, e_cl)``. ``re_mode`` selects how the
    Reynolds dependence is carried:

    ``'per_term'``
        Re appears inside each term with its own exponent, scaled by
        ``re_scale`` (York divides by 1000). This is what SPaircraft's fits do.
    ``'factor'``
        ``cd = fit(tau, M, cl) * (Re/re_ref)**re_exp``, which is what TASOPT
        does -- ``surfcd.f`` applies ``Refac = (Reco/Reref)**aRexp`` outside
        the table lookup, with ``aRexp = -0.15`` in every run deck.

    ``perp_cl`` says the fit's lift argument is the PERPENDICULAR section lift
    coefficient ``c_l/cos^2(Lambda)``, as TASOPT feeds ``airfun``
    (``surfcd.f:109``, ``wingpo.f:59``). With it False the caller passes the
    streamwise C_L, which takes sweep's Mach relief while skipping its cost.
    """
    name: str
    alpha: float
    terms: tuple
    re_mode: str = 'per_term'
    re_scale: float = 1000.0
    re_ref: float = 2.0e7
    re_exp: float = -0.15
    perp_cl: bool = True
    m_perp_max: float | None = None
    tau_range: tuple = (0.09, 0.145)
    source: str = ''
    #: Reference point (Re, tau, M_perp, cl) the terms are normalised about,
    #: and the cd there. See `from_yaml`: with these set the constraint is
    #: (cd/cd_ref)**alpha >= sum_k c_k * prod (x_i/x_ref_i)**e_ki, which keeps
    #: every coefficient in (0,1) summing to 1 instead of spanning ~90 orders
    #: of magnitude. Purely a change of variables -- exponents are untouched
    #: and the fit is bit-for-bit the same function.
    x_ref: tuple | None = None
    cd_ref: float = 1.0
    #: Validity box from the dataset the fit was built on. Recorded whether or
    #: not it is enforced, because a surrogate outside its box is not a model
    #: of anything and the caller should at least be able to check.
    m_range: tuple | None = None
    re_range: tuple | None = None
    cl_range: tuple | None = None

    def cd(self, Re, tau, M_perp, cl):
        """The posynomial that ``(cd/cd_ref)**alpha`` must dominate.

        With ``x_ref`` set the arguments are divided by it, which is what
        keeps the coefficients representable; ``cd_ref`` is 1.0 for the
        un-normalised fits so the caller's expression is unchanged.
        """
        if self.x_ref is not None:
            rr, tr, mr, cr = self.x_ref
            Re, tau, M_perp, cl = (Re / rr, tau / tr, M_perp / mr, cl / cr)
            return sum(c * Re ** eR * tau ** et * M_perp ** eM * cl ** ec
                       for c, eR, et, eM, ec in self.terms)
        if self.re_mode == 'per_term':
            return sum(c * (Re / self.re_scale) ** eR * tau ** et
                       * M_perp ** eM * cl ** ec
                       for c, eR, et, eM, ec in self.terms)
        return sum(c * tau ** et * M_perp ** eM * cl ** ec
                   for c, _eR, et, eM, ec in self.terms)

    def re_factor(self, Re):
        """The separate Reynolds monomial, 1 unless ``re_mode='factor'``."""
        if self.re_mode == 'factor':
            return (Re / self.re_ref) ** self.re_exp
        return 1.0


#: Martin York's fit to the TASOPT C-series transonic airfoils, as it appears
#: in SPaircraft. Valid to M_perp ~ 0.74; see the module docstring.
YORK_C = Polar(
    name='york_c',
    alpha=1.6515,
    terms=(
        (1.61418,     -0.550434,  1.29151,   3.03609,   1.77743),
        (0.0466407,   -0.389048,  0.784123, -0.340157,  0.950763),
        (190.811,     -0.218621,  3.94654,  19.2524,    1.15233),
        (2.82283e-12,  1.18147,  -1.75664,   0.10563,  -1.44114),
    ),
    re_mode='per_term',
    perp_cl=True,
    m_perp_max=0.74,
    source='SPaircraft wing.py; fit to Tasopt2.16/air/C.air',
)

#: The same fit with the perpendicular-c_l correction switched off, i.e.
#: exactly as stock SPaircraft has it. Kept so the difference is measurable
#: rather than asserted: without it the fit's optimum sweep is 45 deg at every
#: condition tested, because the (cosL*M) Mach relief is monotone and nothing
#: opposes it.
YORK_C_STREAMWISE = Polar(
    name='york_c_streamwise',
    alpha=YORK_C.alpha,
    terms=YORK_C.terms,
    re_mode='per_term',
    perp_cl=False,
    m_perp_max=None,
    source='stock SPaircraft, for comparison only',
)

# ---------------------------------------------------------------------------
# MSES refits, 2026-07-31
# ---------------------------------------------------------------------------
#: Loaded from components/mses_fits/*.yaml. These replace York's fit with a
#: surrogate built on a purpose-run MSES sweep (Mach 0.10-0.86, Re 5e6-4e7,
#: forced transition 3%/5%, Ncrit 9), cross-checked against TASOPT's own polar
#: tables at median CD ratio 1.000, 96% within 5%.
#:
#: The reason this matters is not accuracy for its own sake. York's fit is
#: valid only to M_perp ~0.74 and under-predicts drag by 3-6x beyond it, so
#: the model carried a hard `M_perp <= 0.74` fence -- and that fence, not
#: aerodynamics, was setting wing sweep. These fits run to M 0.86 and capture
#: the transonic rise, which lets sweep be decided by a real trade against the
#: structural penalty in surfw.f rather than by a surrogate's edge.
_FIT_DIR = _os.path.join(_os.path.dirname(__file__), "mses_fits")


def from_yaml(path, name=None, x_ref=(2.0e7, 0.12, 0.72, 0.60), **kw):
    """Build a normalised :class:`Polar` from an sfit YAML.

    The YAML gives ``alpha``, ``A`` (K x n) and ``b`` for
    ``log cd = (1/alpha) logsumexp_k(alpha*(b_k + A_k . log x))`` with
    ``x = (Re/1000, tau, Mach[, CL])``.

    ``x_ref`` normalises the fit about a representative operating point. This
    is a pure change of variables -- shift b, leave A alone -- and it is what
    makes the coefficients usable: the T-series tail fit has a term with
    ``alpha*b = -774.7``, whose coefficient ``exp(-774.7)`` UNDERFLOWS to zero
    in float64, silently deleting the high-Mach drag-rise barrier. Normalised,
    every coefficient lands in (0, 1) and they sum to exactly 1 at the
    reference point.

    A 3-column ``A`` (no CL) is padded with a zero CL exponent, so a
    CL-independent tail fit is the same object as a wing fit and needs no
    special case downstream.
    """
    import math
    import yaml
    with open(path) as fh:
        d = yaml.safe_load(fh)
    alpha = float(d["alpha"])
    A = [[float(v) for v in row] for row in d["A"]]
    b = [float(v) for v in d["b"]]
    n_in = len(A[0])
    if n_in == 3:                      # (Re/1000, tau, M); no CL dependence
        A = [row + [0.0] for row in A]
    rr, tr, mr, cr = x_ref
    lxr = [math.log(rr / 1000.0), math.log(tr), math.log(mr), math.log(cr)]

    def _logsum(bs):
        m = max(bs)
        return m + math.log(sum(math.exp(v - m) for v in bs))

    tl = [alpha * (b[k] + sum(A[k][i] * lxr[i] for i in range(4)))
          for k in range(len(b))]
    log_cd_ref = _logsum(tl) / alpha
    terms = []
    for k in range(len(b)):
        bp = b[k] + sum(A[k][i] * lxr[i] for i in range(4)) - log_cd_ref
        terms.append((math.exp(alpha * bp),
                      *(alpha * A[k][i] for i in range(4))))
    return Polar(name=name or d.get("family", "mses"), alpha=alpha,
                 terms=tuple(terms), x_ref=x_ref, cd_ref=math.exp(log_cd_ref),
                 source=f"{_os.path.basename(path)}; rms_log "
                        f"{d.get('rms_log')}, p95 {d.get('p95_pct')}%", **kw)


#: C-series wing polar. The like-for-like replacement for YORK_C, which was a
#: fit to the same family's TASOPT tables.
MSES_C = from_yaml(_os.path.join(_FIT_DIR, "NC_SMA_final.yaml"), name="mses_c",
                   perp_cl=True, m_perp_max=None,
                   m_range=(0.10, 0.86), re_range=(5.0e6, 4.0e7),
                   cl_range=(0.1, 1.2), tau_range=(0.090, 0.145))

#: E-series wing polar, same sweep, different airfoil family.
MSES_E = from_yaml(_os.path.join(_FIT_DIR, "NE_SMA_final.yaml"), name="mses_e",
                   perp_cl=True, m_perp_max=None,
                   m_range=(0.10, 0.86), re_range=(5.0e6, 4.0e7),
                   cl_range=(0.1, 1.2), tau_range=(0.090, 0.145))

#: T-series TAIL polar, CL-independent (fit at CL=0.1, which is where a tail
#: lives). This is the accurate one of the T fits -- rms_log 0.043, p95 7.4%
#: against 0.278 for the CL-dependent version -- and it replaces TASOPT's two
#: Mach-independent constants with a surrogate that has a real drag rise.
MSES_T_TAIL = from_yaml(_os.path.join(_FIT_DIR, "N3-T_SMA_CL0p1_final.yaml"),
                        name="mses_t_tail", x_ref=(1.0e7, 0.10, 0.70, 1.0),
                        perp_cl=True, m_perp_max=None,
                        m_range=(0.10, 0.86), re_range=(5.0e6, 4.0e7),
                        tau_range=(0.100, 0.140))

#: Registry. A refit produced elsewhere is added here and selected by name;
#: nothing else in the model needs to change.
POLARS = {p.name: p for p in (YORK_C, YORK_C_STREAMWISE,
                              MSES_C, MSES_E, MSES_T_TAIL)}


def from_sfit(name, A, b, alpha, **kw):
    """Build a :class:`Polar` from what ``sfit`` returns.

    ``sfit``'s ``SoftMaxAffineFitObject`` gives ``(A, b, alpha)`` for
    ``log cd = (1/alpha) log sum_k exp(alpha*(b_k + A_k . log x))``, so
    ``c_k = exp(alpha*b_k)`` and ``e_k = alpha*A_k``. Column order of ``A``
    must be ``(Re, tau, M_perp, c_l)``, or ``(tau, M_perp, c_l)`` with
    ``re_mode='factor'``, which is the natural shape when fitting a ``.air``
    table at its own Re_ref.
    """
    import math
    terms = []
    for k, bk in enumerate(b):
        e = list(A[k])
        if len(e) == 3:                       # (tau, M, cl); Re rides outside
            e = [0.0] + e
        terms.append((math.exp(alpha * float(bk)),
                      *(alpha * float(v) for v in e)))
    return Polar(name=name, alpha=float(alpha), terms=tuple(terms), **kw)


# ---------------------------------------------------------------------------
# Tail profile drag
# ---------------------------------------------------------------------------
#: TASOPT does not use an airfoil table for tails at all. ``cdsum.f:234-250``
#: calls ``surfcd`` for the horizontal and vertical with two CONSTANTS from the
#: run deck, scaled only by Reynolds number and sweep:
#:
#:     0.0060  !  cdft    tail profile cd
#:     0.0030  !  cdpt
#:     10.0e6  !  Rereft
#:
#: No Mach dependence whatsoever. SPaircraft instead applies a wing-airfoil fit
#: to the tails, which is where ``tau**133.796 * M**1022.7`` comes from -- terms
#: that are meant to fence a fit's valid region and, at subsonic Mach, do not:
#: M**1022.7 underflows to ~1e-158 at M 0.7 and annihilates the tau barrier,
#: which is how the fin reached t/c = 17.55 when the thickness band was lifted.
TASOPT_TAIL_CDF = 0.0060
TASOPT_TAIL_CDP = 0.0030
TASOPT_TAIL_REREF = 10.0e6
TASOPT_ARE_XP = -0.15
