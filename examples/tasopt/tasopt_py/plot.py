"""Drawing a sized aircraft -- what ``picwrt``, ``picidr`` and the ``.plt``
files were for, done with matplotlib.

TASOPT does not plot anything. It writes *plotting instructions*: ``picwrt``
emits a gnuplot data file of polylines, ``picidr`` the same picture as idraw
drawing commands, ``pltwrt`` three Matlab files, ``mapwrt`` and ``prfwrt``
tables meant to be fed to something else. Five emitters, five syntaxes, one
picture each, and none of them draws anything on its own.

Porting those syntaxes to Python would be translating a dead format. What is
actually wanted is the picture, so this module takes the same data --
``airpic``'s polylines, ``prfwrt``'s columns, ``blfwrt``'s stations,
``mapwrt``'s map coordinates -- and draws it.

============================  ===========================================
:func:`plan_view`             ``picwrt``, ``picidr``, ``pltwrt``'s geometry
:func:`mission_profile`       ``prfwrt`` / ``prof_MMM.dat``
:func:`fuselage_bl`           ``blfwrt`` / ``blfwrt2``
:func:`compressor_maps`       ``mapwrt`` / ``tfan_MMM.dat``
:func:`engine_deck`           ``eopwrt`` / the ``.oute`` deck
:func:`sweep`                 ``pltwrt``'s parameter file
============================  ===========================================

Everything takes an optional ``ax`` (or ``axes``) and returns what it drew
on, so a caller can compose figures. Nothing sets a style, a colour cycle or
a figure size unless it has to -- the caller's rcParams win.

matplotlib is not a dependency of the port; it is imported inside these
functions, so ``import tasopt_py`` works without it and only plotting fails.

Verified in the sense that matters here: :func:`plan_view` is checked against
the polylines ``picwrt`` writes for the 737 (``tests/test_plot.py``), so the
picture is the reference program's picture and not a redrawing of it.
"""
from __future__ import annotations

import math

from .model import indices as I
from .output import FT_M, HR_S, KFT_M, LB_N, NMI_M, map_point
from .planview import airpic

__all__ = ["plan_view", "mission_profile", "fuselage_bl", "compressor_maps",
           "engine_deck", "sweep", "figure_for"]

#: Metres per unit, by name. The reports are in feet; the model is SI.
_UNITS = {"ft": FT_M, "m": 1.0}


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "plotting needs matplotlib; the rest of tasopt_py does not. "
            "pip install matplotlib") from exc
    return plt


def _ax(ax, **kw):
    if ax is not None:
        return ax
    return _plt().subplots(**kw)[1]


def _scale(units):
    try:
        return _UNITS[units]
    except KeyError:
        raise ValueError(f"units must be one of {sorted(_UNITS)}, "
                         f"not {units!r}") from None


# --------------------------------------------------------------------------
# The aircraft -- picwrt, picidr
# --------------------------------------------------------------------------

def plan_view(pari, parg, ax=None, *, xorg=None, units="ft", marks=True,
              fill=True, **kw):
    """Draw the plan view of a sized aircraft.

    The same picture ``picwrt`` writes, in the same orientation: span across
    the page, **nose up**, origin at ``xorg``. ``picwrt`` passes
    ``parg(igxNP)`` for that, putting x = 0 at the neutral point, and so does
    this by default; pass ``xorg=0`` to keep the model's own nose-at-zero
    coordinates.

    ``marks`` draws what ``picwrt`` draws besides the outline -- a long tick
    at the neutral point, short ticks at the forward and aft CG limits, and
    the CG range as a line between them. Those three numbers are most of what
    a stability argument turns on, and having them on the picture is the
    reason ``picwrt`` exists rather than ``pltwrt`` alone.

    Extra keyword arguments go to the outline ``plot`` call.
    """
    ax = _ax(ax)
    view = airpic(pari, parg)
    lunit = _scale(units)
    if xorg is None:
        xorg = parg[I.IGXNP]

    def draw(poly, **more):
        ys = [y * lunit for _, y in poly]
        xs = [-(x - xorg) * lunit for x, _ in poly]
        return ax.plot(ys, xs, **more)

    style = dict(color="0.25", lw=1.0)
    style.update(kw)
    fillstyle = dict(color="0.85", zorder=0)

    for poly in (view.wing, view.htail, view.fuselage):
        mirrored = [(x, -y) for x, y in poly]
        for half in (poly, mirrored):
            if fill:
                ax.fill([y * lunit for _, y in half],
                        [-(x - xorg) * lunit for x, _ in half], **fillstyle)
            draw(half, **style)

    # picwrt spreads neng engines evenly from +yeng to -yeng.
    xeng, yeng = parg[I.IGXENG], parg[I.IGYENG]
    neng = int(parg[I.IGNENG] + 0.01)
    for k in range(neng):
        frac = 0.0 if neng == 1 else float(k) / float(neng - 1)
        nacelle = view.nacelle_at(xeng, yeng * (1.0 - 2.0 * frac))
        if fill:
            ax.fill([y * lunit for _, y in nacelle],
                    [-(x - xorg) * lunit for x, _ in nacelle], **fillstyle)
        draw(nacelle, **style)

    if marks:
        dy = 0.1 * parg[I.IGBO] * lunit
        def tick(x, half, **more):
            ax.plot([half, -half], [-(x - xorg) * lunit] * 2, **more)
        tick(parg[I.IGXNP], 2.0 * dy, color="C3", lw=1.2)
        tick(parg[I.IGXCGAFT], dy, color="C0", lw=1.2)
        tick(parg[I.IGXCGFWD], dy, color="C0", lw=1.2)
        ax.plot([0.0, 0.0],
                [-(parg[I.IGXCGFWD] - xorg) * lunit,
                 -(parg[I.IGXCGAFT] - xorg) * lunit],
                color="C0", lw=1.2)

    ax.set_aspect("equal")
    ax.set_xlabel(f"y [{units}]")
    ax.set_ylabel(f"nose-up from x_NP [{units}]" if xorg else f"x [{units}]")
    return ax


# --------------------------------------------------------------------------
# The mission -- prfwrt
# --------------------------------------------------------------------------

def _flown(para):
    """The mission points that have a range, in order.

    Static and rotation sit at or behind zero range and would fold the
    profile back on itself, so they are left out -- ``prfwrt`` tabulates them
    but there is nothing to plot them against.
    """
    return [ip for ip in range(I.IPTAKEOFF, I.IPDESCENTN + 1)
            if ip != I.IPROTATE]


def mission_profile(parg, parm, para, pare, axes=None, **kw):
    """The mission, in four panels against range.

    Altitude; weight as a fraction of takeoff weight; lift coefficient and
    L/D; and turbine inlet temperature with TSFC. Those are ``prfwrt``'s
    columns, which is the table the ``prof_MMM.dat`` file holds.

    Returns the four axes.
    """
    plt = _plt()
    if axes is None:
        _, axes = plt.subplots(2, 2, sharex=True, figsize=(9, 6))
    axes = list(getattr(axes, "flat", axes))
    if len(axes) < 4:
        raise ValueError(f"mission_profile needs 4 axes, got {len(axes)}")

    ips = _flown(para)
    R = [para[I.IARANGE, ip] * NMI_M for ip in ips]
    style = dict(marker="o", ms=3)
    style.update(kw)

    axes[0].plot(R, [para[I.IAALT, ip] * KFT_M for ip in ips], **style)
    axes[0].set_ylabel("altitude [kft]")

    axes[1].plot(R, [para[I.IAFRACW, ip] for ip in ips], **style)
    axes[1].set_ylabel("W / W_TO")

    axes[2].plot(R, [para[I.IACL, ip] for ip in ips], label="CL", **style)
    LoD = [para[I.IACL, ip] / para[I.IACD, ip] if para[I.IACD, ip] else
           float("nan") for ip in ips]
    twin = axes[2].twinx()
    twin.plot(R, LoD, color="C1", **style)
    twin.set_ylabel("L/D", color="C1")
    axes[2].set_ylabel("CL", color="C0")
    axes[2].set_xlabel("range [nmi]")

    axes[3].plot(R, [pare[I.IETT4, ip] for ip in ips], **style)
    axes[3].set_ylabel("Tt4 [K]", color="C0")
    twin = axes[3].twinx()
    twin.plot(R, [pare[I.IETSFC, ip] / HR_S for ip in ips], color="C1",
              **style)
    twin.set_ylabel("TSFC [1/hr]", color="C1")
    axes[3].set_xlabel("range [nmi]")

    return axes


# --------------------------------------------------------------------------
# The fuselage boundary layer -- blfwrt
# --------------------------------------------------------------------------

def fuselage_bl(bl, axes=None, *, units="ft", **kw):
    """The fuselage BL and wake development, in four panels against x.

    Edge velocity, kinematic shape parameter, skin friction and the
    displacement and momentum thicknesses -- the quantities ``blfwrt`` puts
    in the ``.out`` report. The trailing edge is marked, because everything
    downstream of it is wake and the closure switches there.

    ``bl`` is a :class:`~tasopt_py.aero.fusebl.FuselageBL`.
    """
    plt = _plt()
    if axes is None:
        _, axes = plt.subplots(2, 2, sharex=True, figsize=(9, 6))
    axes = list(getattr(axes, "flat", axes))
    if len(axes) < 4:
        raise ValueError(f"fuselage_bl needs 4 axes, got {len(axes)}")

    lunit = _scale(units)
    x = [v * lunit for v in bl.x]
    xte = bl.x[bl.iblte - 1] * lunit
    style = dict(lw=1.2)
    style.update(kw)

    axes[0].plot(x, bl.uinv, ls="--", color="0.6", label="inviscid", lw=1.0)
    axes[0].plot(x, bl.ue, label="ue/V", **style)
    axes[0].set_ylabel("edge velocity / V")
    axes[0].legend(fontsize="small")

    axes[1].plot(x, bl.hk, **style)
    axes[1].set_ylabel("Hk")

    axes[2].plot(x, [0.5 * v for v in bl.cf], **style)
    axes[2].set_ylabel("Cf / 2")
    axes[2].set_xlabel(f"x [{units}]")

    axes[3].plot(x, [v * lunit for v in bl.ds], label="delta*", **style)
    axes[3].plot(x, [v * lunit for v in bl.th], label="theta", **style)
    axes[3].set_ylabel(f"thickness [{units}]")
    axes[3].set_xlabel(f"x [{units}]")
    axes[3].legend(fontsize="small")

    for a in axes[:4]:
        a.axvline(xte, color="0.7", lw=0.8, zorder=0)

    return axes


# --------------------------------------------------------------------------
# The engine -- mapwrt, eopwrt
# --------------------------------------------------------------------------

#: The mission segments ``tfwrt`` separates with blank lines, and what to
#: call them.
_SEGMENTS = (("ground", (I.IPSTATIC, I.IPROTATE, I.IPTAKEOFF, I.IPCUTBACK)),
             ("climb", tuple(range(I.IPCLIMB1, I.IPCLIMBN + 1))),
             ("cruise", tuple(range(I.IPCRUISE1, I.IPCRUISEN + 1))),
             ("descent", tuple(range(I.IPDESCENT1, I.IPDESCENTN + 1))))


def compressor_maps(pare, axes=None, **kw):
    """Where the fan, LPC and HPC sit on their maps over the mission.

    Corrected mass flow against pressure ratio, both as fractions of the
    design value, which is the coordinate system a compressor map is drawn
    in and exactly what ``mapwrt`` tabulates. The design point is marked;
    the mission track is drawn segment by segment so it is visible which
    excursion belongs to takeoff and which to descent.

    Returns the three axes.
    """
    plt = _plt()
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    axes = list(getattr(axes, "flat", axes))
    if len(axes) < 3:
        raise ValueError(f"compressor_maps needs 3 axes, got {len(axes)}")

    style = dict(marker="o", ms=4, lw=1.0)
    style.update(kw)

    for a, name in zip(axes, ("fan", "lpc", "hpc")):
        for label, ips in _SEGMENTS:
            pts = [getattr(map_point(pare.column(ip)), name) for ip in ips]
            a.plot([p[0] for p in pts], [p[1] for p in pts], label=label,
                   **style)
        # Design: corrected flow and speed are unity there by definition,
        # and the design pressure ratio comes off the same column.
        design = pare[{"fan": I.IEPIFD, "lpc": I.IEPILCD,
                       "hpc": I.IEPIHCD}[name], I.IPCRUISE1]
        a.plot([1.0], [design], marker="*", ms=12, color="k", ls="none",
               label="design", zorder=5)
        a.set_xlabel("corrected mass flow / design")
        a.set_ylabel("pressure ratio")
        a.set_title(name.upper() if name != "fan" else "Fan")
    axes[0].legend(fontsize="small")
    return axes


def engine_deck(deck, ax=None, *, y="TSFC", **kw):
    """The off-design deck: one throttle curve per flight condition.

    ``y`` is ``"TSFC"`` (1/hr) or ``"mfuel"`` (kg/s per engine), against
    thrust per engine in kN. Each curve is one altitude and Mach number, so
    the family shows how the engine's efficiency moves with both -- which is
    what the deck is computed for.
    """
    ax = _ax(ax)
    style = dict(marker="o", ms=4, lw=1.0)
    style.update(kw)

    conditions = []
    for p in deck.points:
        if (p.alt, p.mach) not in conditions:
            conditions.append((p.alt, p.mach))

    for alt, mach in conditions:
        pts = deck.at(alt, mach)
        ax.plot([p.F / 1000.0 for p in pts], [getattr(p, y) for p in pts],
                label=f"{alt * KFT_M:.0f} kft, M{mach:.2f}", **style)

    ax.set_xlabel("thrust per engine [kN]")
    ax.set_ylabel({"TSFC": "TSFC [1/hr]",
                   "mfuel": "fuel flow [kg/s]"}.get(y, y))
    ax.legend(fontsize="small", ncol=2)
    return ax


# --------------------------------------------------------------------------
# Parameter sweeps -- pltwrt's parameter file
# --------------------------------------------------------------------------

def sweep(results, x, y, ax=None, *, label=None, **kw):
    """Plot one swept quantity against another over a family of runs.

    ``results`` is what :func:`tasopt_py.run.run_sweep` yields;  ``x`` and
    ``y`` are :data:`tasopt_py.planview.PLOT_COLUMNS` names, which are the 28
    columns ``pltwrt`` writes per grid point -- PFEI, WMTO, AR, L/D, span
    loading, the noise levels and so on.

    This is the carpet plot the Matlab files exist to produce, without the
    Matlab.

    One column to be careful with: ``"PFEI"`` is the *fleet* value out of
    ``parg``, and nothing puts it there except
    :func:`tasopt_py.output.report`. Plot it on a run whose report has not
    been written and you get the "unset" fill value, 9e307, exactly as
    ``pltwrt`` would -- see ``DISCREPANCIES.md`` §43. Use the design
    mission's own ``parm(imPFEI)`` if you have not written a report.
    """
    from .planview import PLOT_COLUMNS, plot_row

    ax = _ax(ax)
    for name in (x, y):
        if name not in PLOT_COLUMNS:
            raise ValueError(f"{name!r} is not a swept column; "
                             f"valid names are {list(PLOT_COLUMNS)}")
    kx = PLOT_COLUMNS.index(x) + 1        # +1 for the leading j value
    ky = PLOT_COLUMNS.index(y) + 1

    rows = []
    for r in results:
        m = r.case.missions[0]
        rows.append(plot_row(r.case.parg, m.parm, m.para, m.pare))

    style = dict(marker="o", ms=4)
    style.update(kw)
    ax.plot([row[kx] for row in rows], [row[ky] for row in rows],
            label=label, **style)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax


def figure_for(result, *, units="ft"):
    """A one-page summary of a completed run: the aircraft, the mission, the
    boundary layer and the compressor maps.

    Convenience only -- the individual functions are what to reach for when
    composing a figure. Returns the Figure.
    """
    plt = _plt()
    fig = plt.figure(figsize=(13, 11), layout="constrained")
    gs = fig.add_gridspec(4, 4)

    case = result.case
    m = case.missions[0]

    plan_view(case.pari, case.parg, fig.add_subplot(gs[0:2, 0:2]),
              units=units)
    mission_profile(case.parg, m.parm, m.para, m.pare,
                    [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]),
                     fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])])
    if result.fuselage_bl is not None:
        fuselage_bl(result.fuselage_bl,
                    [fig.add_subplot(gs[2, k]) for k in range(4)],
                    units=units)
    compressor_maps(m.pare, [fig.add_subplot(gs[3, k]) for k in range(3)])

    # The *fleet* PFEI is computed and stored by the report, not by the
    # sizing -- parg(igPFEI) still holds the "unset" fill value here unless
    # output.report has been called (DISCREPANCIES.md §43). Summed rather
    # than read, so a figure does not depend on having written a file.
    PFEI = sum(mm.parm[I.IMWOPT] * mm.parm[I.IMPFEI] for mm in case.missions)
    fig.suptitle(f"{case.configname.strip()} -- "
                 f"WTO {m.parm[I.IMWTO] * LB_N:,.0f} lbf, "
                 f"PFEI {PFEI:.4f} KJ/kg-km", fontsize="medium")
    return fig
