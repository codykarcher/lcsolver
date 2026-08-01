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

    def cd(self, Re, tau, M_perp, cl):
        """The posynomial that ``cd**alpha`` must dominate."""
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

#: Registry. A refit produced elsewhere is added here and selected by name;
#: nothing else in the model needs to change.
POLARS = {p.name: p for p in (YORK_C, YORK_C_STREAMWISE)}


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
