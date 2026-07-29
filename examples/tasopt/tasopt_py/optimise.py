"""Design optimisation -- ``fobj.f``, ``simpop.f``, ``hsort.f``.

Wraps the sizing loop in a Nelder-Mead simplex search over a handful of design
variables. Each objective evaluation is a *complete aircraft*: size it for the
design mission with ``wsize``, fly every weighted off-design mission with
``woper``, and return the fleet fuel burn.

The variables
-------------
Whichever of ``cparo`` the ``.tas`` file gives a nonzero perturbation to --
cruise CL, aspect ratio, sweep, the two box heights, the two taper ratios, the
two spanload scales, FPR, BPR, cruise altitude, cruise and takeoff ``Tt4``, and
OPR. The perturbation doubles as the initial simplex edge and as the scale the
convergence tolerance is measured against.

The objective
-------------
``fun`` is the weighted fleet fuel weight. Three *ad hoc* penalties are added
to it before any constraint is considered, and they are worth knowing about
because they are shaping the design space, not describing the aircraft:

* **negative sweep**, because the model only ever uses ``cos(sweep)`` and so
  cannot tell a swept-forward wing from a swept-back one;
* **inner-panel reverse taper**, which the optimiser would otherwise find
  attractive on strut-braced wings, the model not knowing about the download
  requirement that rules it out;
* **excessive tip taper**, to keep the search away from a negative tip chord.

Each is ``Wpay * max(violation, 0)**2`` with the violation scaled to a
tenth-ish of the variable's range, so they are soft walls rather than bounds.

Then up to four real constraints -- balanced field length, fuel volume, span,
top-of-climb climb angle -- each switched on by its own flag in the ``.tas``
file, each entering as ``penfac * max(g, 0)**2`` with ``penfac`` a multiple of
the payload weight. So this is a pure penalty method: nothing is enforced, and
a converged design can sit slightly outside a constraint.

Things in the source worth knowing
----------------------------------
* ``gradop.f``, the BFGS alternative, is **an empty shell**: its step loop
  computes a gradient norm and does nothing with it, there is no line search
  and no update, and its call site in ``tasopt.f`` is commented out. It is not
  ported; :func:`gobj` is, since it is what a working ``gradop`` would need,
  and it is what ``fobj``'s finite-difference gradients come from.
* ``voptset`` clamps the tip taper ratio at 0.1 and **writes the clamped value
  back into the optimiser's own variable vector**, so the simplex vertex
  moves. That is unusual -- a clamp that the search can see -- and it is
  reproduced.
* Off-design missions with zero weight are skipped during the search and their
  fuel set to zero, then computed once at the end. So a zero-weighted mission
  does not slow the optimisation down but still gets reported.
* The simplex constants are ``alpha, beta, gamma = 1.0, 0.7, 1.4``; two other
  sets, the textbook ``1.0, 0.5, 2.0`` and ``1.0, 0.6, 1.6``, are commented
  out above them.

Verified against the compiled Fortran; see ``tests/test_optimise.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .model import indices as I
from .sizing.woper import woper
from .sizing.wsize import wsize

__all__ = ["optimise", "fobj", "gobj", "voptset", "simpop", "hsort",
           "OptResult", "ObjectiveResult", "ALPHA", "BETA", "GAMMA",
           "SWEEP_FLOOR", "LAMS_CEILING", "LAMT_FLOOR"]

#: Nelder-Mead reflection, contraction and expansion factors. Two other sets
#: are commented out above these in the source.
ALPHA, BETA, GAMMA = 1.0, 0.7, 1.4

#: The three ad-hoc penalty thresholds, and the scale each violation is
#: measured against.
SWEEP_FLOOR, SWEEP_SCALE = 1.5, 100.0
LAMS_CEILING, LAMS_SCALE = 1.0, 0.25
LAMT_FLOOR, LAMT_SCALE = 0.15, 0.25
#: voptset's hard clamp on the tip taper ratio.
LAMT_CLAMP = 0.1


@dataclass
class ObjectiveResult:
    fun: float = 0.0          # unconstrained objective (fleet fuel weight)
    func: float = 0.0         # with the penalties added
    con: list = field(default_factory=list)     # constraint violations
    PFEI: float = 0.0
    converged: bool = True


@dataclass
class OptResult:
    x: list = field(default_factory=list)       # best variable vector
    f: float = 0.0                              # its objective value
    steps: int = 0
    converged: bool = False
    fdel: float = 0.0                           # simplex spread at exit
    history: list = field(default_factory=list)
    variables: dict = field(default_factory=dict)   # by cparo keyword


# --------------------------------------------------------------------------
# hsort.f
# --------------------------------------------------------------------------

def hsort(a) -> list:
    """Heapsort, returning the permutation that orders ``a`` ascending.

    A literal port of ``hsort.f`` -- itself, its comment says, "stolen from
    numerical recipes" -- rather than ``sorted(range(n), key=a.__getitem__)``,
    because a different tie-break would reorder equal simplex vertices and
    send the search down a different path.
    """
    n = len(a)
    indx = list(range(1, n + 1))         # 1-based, as the Fortran returns
    if n <= 1:
        return indx

    a = [0.0] + list(a)                  # 1-based view of the values
    ll = n // 2 + 1
    ir = n
    while True:
        if ll > 1:
            ll -= 1
            indxt = indx[ll - 1]
            q = a[indxt]
        else:
            indxt = indx[ir - 1]
            q = a[indxt]
            indx[ir - 1] = indx[0]
            ir -= 1
            if ir == 1:
                indx[0] = indxt
                return indx
        i = ll
        j = ll + ll
        while j <= ir:
            if j < ir and a[indx[j - 1]] < a[indx[j]]:
                j += 1
            if q < a[indx[j - 1]]:
                indx[i - 1] = indx[j - 1]
                i = j
                j += j
            else:
                j = ir + 1
        indx[i - 1] = indxt


# --------------------------------------------------------------------------
# fobj.f
# --------------------------------------------------------------------------

def voptset(case, iovar, vopt) -> None:
    """Scatter the optimiser's variable vector into the parameter arrays.

    ``iovar`` holds the ``cparo`` keyword for each active variable, in the
    order they appear in ``vopt``. Modifies ``vopt`` in place where the tip
    taper ratio is clamped -- see the module docstring.
    """
    parg = case.parg
    for iv, name in enumerate(iovar):
        v = vopt[iv]
        if name == "CL":
            for m in case.missions:
                for ip in range(I.IPCLIMB1 + 1, I.IPDESCENTN):
                    m.para[I.IACL, ip] = v
        elif name == "AR":
            parg[I.IGAR] = v
        elif name == "sweep":
            parg[I.IGSWEEP] = v
            parg[I.IGSWEEPH] = v
        elif name == "hboxo":
            parg[I.IGHBOXO] = v
        elif name == "hboxs":
            parg[I.IGHBOXS] = v
        elif name == "lambdas":
            parg[I.IGLAMBDAS] = v
        elif name == "lambdat":
            parg[I.IGLAMBDAT] = v
            if parg[I.IGLAMBDAT] < LAMT_CLAMP:
                # The clamp is written back into the search vector, so the
                # simplex vertex itself moves.
                parg[I.IGLAMBDAT] = LAMT_CLAMP
                vopt[iv] = LAMT_CLAMP
        elif name == "rcls":
            for m in case.missions:
                for ip in range(I.IPCLIMB1 + 1, I.IPDESCENTN):
                    m.para[I.IARCLS, ip] = v
        elif name == "rclt":
            for m in case.missions:
                for ip in range(I.IPCLIMB1 + 1, I.IPDESCENTN):
                    m.para[I.IARCLT, ip] = v
        elif name == "FPR":
            for m in case.missions:
                for ip in range(I.IPCRUISE1, I.IPCRUISEN + 1):
                    m.pare[I.IEPIF, ip] = v
        elif name == "BPR":
            for m in case.missions:
                for ip in range(I.IPCRUISE1, I.IPCRUISEN + 1):
                    m.pare[I.IEBPR, ip] = v
        elif name == "alt":
            for m in case.missions:
                m.para[I.IAALT, I.IPCRUISE1] = v
        elif name == "Tt4CR":
            for m in case.missions:
                for ip in range(I.IPCRUISE1, I.IPCRUISEN + 1):
                    m.pare[I.IETT4, ip] = v
        elif name == "Tt4TO":
            for m in case.missions:
                for ip in (I.IPSTATIC, I.IPROTATE, I.IPTAKEOFF):
                    m.pare[I.IETT4, ip] = v
        elif name == "OPR":
            for m in case.missions:
                for ip in range(I.IPCRUISE1, I.IPCRUISEN + 1):
                    m.pare[I.IEPIHC, ip] = v / m.pare[I.IEPILC, ip]
        else:
            raise ValueError(f"unknown optimisation variable {name!r}")


def voptget(case, iovar) -> list:
    """The current value of each active variable -- the first simplex vertex.

    The mirror of :func:`voptset`; ``tasopt.f`` writes it out inline.
    """
    parg = case.parg
    m = case.missions[0]
    out = []
    for name in iovar:
        if name == "CL":
            out.append(m.para[I.IACL, I.IPCRUISE1])
        elif name == "AR":
            out.append(parg[I.IGAR])
        elif name == "sweep":
            out.append(parg[I.IGSWEEP])
        elif name == "hboxo":
            out.append(parg[I.IGHBOXO])
        elif name == "hboxs":
            out.append(parg[I.IGHBOXS])
        elif name == "lambdas":
            out.append(parg[I.IGLAMBDAS])
        elif name == "lambdat":
            out.append(parg[I.IGLAMBDAT])
        elif name == "rcls":
            out.append(m.para[I.IARCLS, I.IPCRUISE1])
        elif name == "rclt":
            out.append(m.para[I.IARCLT, I.IPCRUISE1])
        elif name == "FPR":
            out.append(m.pare[I.IEPIF, I.IPCRUISE1])
        elif name == "BPR":
            out.append(m.pare[I.IEBPR, I.IPCRUISE1])
        elif name == "alt":
            out.append(m.para[I.IAALT, I.IPCRUISE1])
        elif name == "Tt4CR":
            out.append(m.pare[I.IETT4, I.IPCRUISE1])
        elif name == "Tt4TO":
            out.append(m.pare[I.IETT4, I.IPTAKEOFF])
        elif name == "OPR":
            out.append(m.pare[I.IEPIHC, I.IPCRUISE1]
                       * m.pare[I.IEPILC, I.IPCRUISE1])
        else:
            raise ValueError(f"unknown optimisation variable {name!r}")
    return out


def fobj(case, iovar, vopt, *, table, settings, state) -> ObjectiveResult:
    """One objective evaluation: a whole aircraft.

    ``state`` is a mutable dict carrying ``initwgt``/``initeng`` between
    calls -- the source keeps them in a COMMON block and flips both to 1 after
    the first evaluation, so every later sizing warm-starts from the last one.
    """
    voptset(case, iovar, vopt)

    parg = case.parg
    s = settings
    r = wsize(*case.design, iterwmax=s.iterwmax, wrlx1=s.wrlx1,
              wrlx2=s.wrlx2, wrlx3=s.wrlx3,
              initwgt=state["initwgt"], initeng=state["initeng"],
              table=table)

    design = case.missions[0]
    for m in case.missions[1:]:
        if m.parm[I.IMWOPT] == 0.0:
            # No influence on the objective -- skip it for now and compute it
            # once the search has converged.
            m.parm[I.IMWFUEL] = 0.0
        else:
            woper(case.pari, parg, m.parm, m.para, m.pare,
                  design.para, design.pare, iterfmax=s.iterfmax,
                  initeng=state["initeng"], table=table)

    Wftotal = Hftotal = PRtotal = 0.0
    for m in case.missions:
        wOpt = m.parm[I.IMWOPT]
        if wOpt == 0.0:
            continue
        hfuel = m.pare[I.IEHFUEL, I.IPCRUISE1]
        Wfuel = m.parm[I.IMWFUEL]
        Wburn = Wfuel / (1.0 + parg[I.IGFRESERVE])
        Wftotal += wOpt * Wfuel
        Hftotal += wOpt * Wburn * hfuel
        PRtotal += wOpt * m.parm[I.IMWPAY] * m.parm[I.IMRANGE]

    parg[I.IGPFEI] = Hftotal / PRtotal
    fun = Wftotal

    # --- ad-hoc penalties, before any real constraint ---------------------
    Wpay = parg[I.IGWPAY]
    for value, thresh, scale, above in (
            (parg[I.IGSWEEP], SWEEP_FLOOR, SWEEP_SCALE, False),
            (parg[I.IGLAMBDAS], LAMS_CEILING, LAMS_SCALE, True),
            (parg[I.IGLAMBDAT], LAMT_FLOOR, LAMT_SCALE, False)):
        fcon = ((value - thresh) if above else (thresh - value)) / scale
        fun += Wpay * max(fcon, 0.0) ** 2

    # --- constraints, each behind its own flag ----------------------------
    con, penfac = [], []
    m1 = case.missions[0]
    if s.LlBFcon:
        con.append(m1.parm[I.IMLBF] / parg[I.IGLBFMAX] - 1.0)
        penfac.append(5.0 * Wpay)
    if s.LWfmaxcon:
        con.append(parg[I.IGWFUEL]
                   / (parg[I.IGRWFMAX] * parg[I.IGWFMAX]) - 1.0)
        penfac.append(5.0 * Wpay)
    if s.Lbmaxcon:
        con.append(parg[I.IGB] / parg[I.IGBMAX] - 1.0)
        penfac.append(25.0 * Wpay)
    if s.Lgtoccon:
        con.append(1.0 - m1.para[I.IAGAMV, I.IPCLIMBN] / parg[I.IGGTOCMIN])
        penfac.append(1.0 * Wpay)

    dfunc = sum(p * max(c, 0.0) ** 2 for c, p in zip(con, penfac))
    func = fun + dfunc
    parg[I.IGFOPT] = func

    # From here on, warm-start both the weights and the engine.
    state["initwgt"] = 1
    state["initeng"] = 1

    return ObjectiveResult(fun=fun, func=func, con=con,
                           PFEI=parg[I.IGPFEI], converged=r.converged)


def gobj(case, iovar, vopt, dvopt, *, table, settings, state):
    """Objective, its gradient, and the constraint gradients.

    Central differences with a step of half the nominal perturbation. This is
    what ``gradop`` would consume; ``gradop`` itself is an empty shell in the
    source and is not ported.
    """
    base = fobj(case, iovar, list(vopt), table=table, settings=settings,
                state=state)
    veps = 0.5
    funv, conv = [], []
    for io in range(len(vopt)):
        vdel = veps * dvopt[io]
        v1, v2 = list(vopt), list(vopt)
        v1[io] -= 0.5 * vdel
        v2[io] += 0.5 * vdel
        r1 = fobj(case, iovar, v1, table=table, settings=settings,
                  state=state)
        r2 = fobj(case, iovar, v2, table=table, settings=settings,
                  state=state)
        funv.append((r2.fun - r1.fun) / vdel)
        conv.append([(c2 - c1) / vdel for c1, c2 in zip(r1.con, r2.con)])
    return base, funv, conv


# --------------------------------------------------------------------------
# simpop.f
# --------------------------------------------------------------------------

def simpop(f_of, x, ftol, istep0, istepn, on_step=None) -> OptResult:
    """Nelder-Mead over the simplex ``x`` (a list of ``n+1`` vertices).

    ``f_of(vertex, tag, istep)`` returns the objective. The tag is the
    one-character label the source prints -- ``j`` for an initial vertex,
    ``r`` reflected, ``e`` expanded, ``c`` contracted, ``m`` shrunk, ``!``
    final -- and is carried through because it is how a run is read.

    Stops when the spread between the best and worst vertices falls below
    ``ftol``, or at ``istepn``. Returns the best vertex first.
    """
    n = len(x[0])
    np1 = n + 1
    if len(x) != np1:
        raise ValueError(f"simplex needs {np1} vertices, got {len(x)}")

    x = [list(v) for v in x]
    f = [0.0] * np1
    history = []

    if istep0 == 0:
        for j in range(np1):
            f[j] = f_of(x[j], "j", 1 if j == 0 else 0)

    converged = False
    fdel = 0.0
    istep = istep0
    for istep in range(istep0 + 1, istepn + 1):
        order = hsort(f)
        f = [f[k - 1] for k in order]
        x = [x[k - 1] for k in order]
        jlo, jnxhi, jhi = 0, np1 - 2, np1 - 1

        fdel = f[jhi] - f[jlo]
        dfrel = fdel / ftol
        history.append((istep, f[jlo], fdel))
        if on_step is not None:
            on_step(istep, x[jlo], f[jlo], fdel)
        if dfrel < 1.0:
            converged = True
            break

        # Centroid of the face opposite the worst vertex.
        xcen = [sum(x[j][i] for j in range(np1) if j != jhi) / float(n)
                for i in range(n)]

        xrfl = [(1.0 + ALPHA) * xcen[i] - ALPHA * x[jhi][i] for i in range(n)]
        frfl = f_of(xrfl, "r", istep)

        if frfl <= f[jlo]:
            # Better than the best -- try going further.
            xexp = [GAMMA * xrfl[i] + (1.0 - GAMMA) * xcen[i]
                    for i in range(n)]
            fexp = f_of(xexp, "e", istep)
            if fexp < f[jlo]:
                x[jhi], f[jhi] = xexp, fexp
            else:
                x[jhi], f[jhi] = xrfl, frfl
        elif frfl >= f[jnxhi]:
            # Worse than the second worst.
            if frfl < f[jhi]:
                x[jhi], f[jhi] = list(xrfl), frfl
            xcon = [BETA * x[jhi][i] + (1.0 - BETA) * xcen[i]
                    for i in range(n)]
            fcon = f_of(xcon, "c", istep)
            if fcon < f[jhi]:
                x[jhi], f[jhi] = xcon, fcon
            else:
                # Shrink the whole simplex towards the best vertex.
                for j in range(np1):
                    if j != jlo:
                        x[j] = [BETA * x[j][i] + (1.0 - BETA) * x[jlo][i]
                                for i in range(n)]
                        f[j] = f_of(x[j], "m", istep)
        else:
            x[jhi], f[jhi] = xrfl, frfl

    f[0] = f_of(x[0], "!", istep)
    return OptResult(x=x[0], f=f[0], steps=istep, converged=converged,
                     fdel=fdel, history=history)


def optimise(case, *, table, settings, istepmax=None, on_step=None,
             Loprint: bool = False) -> OptResult:
    """Optimise a case read from a ``.tas`` file.

    The active variables and their perturbations come from the file's
    optimisation block; the perturbation is both the initial simplex edge and
    the scale the search moves on. Returns the best vertex found, with the
    case left holding that design.
    """
    iovar = [k for k in I.CPARO if case.dvarso.get(k, 0.0) != 0.0]
    if not iovar:
        raise ValueError("no optimisation variables have a perturbation; "
                         "the .tas file's optimisation block is all zeros")

    v0 = voptget(case, iovar)
    simplex = [list(v0)]
    for iv, name in enumerate(iovar):
        v = list(v0)
        v[iv] += case.dvarso[name]
        simplex.append(v)

    state = {"initwgt": 0, "initeng": 0}
    steps = settings.istepmax if istepmax is None else istepmax

    def f_of(vertex, tag, istep):
        r = fobj(case, iovar, vertex, table=table, settings=settings,
                 state=state)
        if Loprint and istep >= 0:
            print(f" {tag}{istep:4d} {r.func * 1.0:12.2f}"
                  f"  fuel {case.missions[0].parm[I.IMWFUEL]:12.2f}"
                  f"  PFEI {r.PFEI:8.4f}"
                  + "".join(f"{v:10.5f}" for v in vertex))
        return r.func

    out = simpop(f_of, simplex, settings.Wftol, 0, steps, on_step=on_step)

    # Leave the case holding the winning design, and pick up the off-design
    # missions that were skipped during the search.
    voptset(case, iovar, out.x)
    out.variables = dict(zip(iovar, out.x))
    return out
