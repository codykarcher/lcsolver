"""Run a ``.tas`` case end to end -- the port's equivalent of ``tasopt.f``.

Reads a case file, sizes the aircraft for the design mission, flies each
remaining mission off-design, and runs the certification-noise points. With
``Lopt = T`` in the file (or ``--optimise``) it instead runs the Nelder-Mead
search of :mod:`tasopt_py.optimise`, sizing an aircraft per objective
evaluation. With ``i``/``j`` sequence values in the file it sweeps over them,
sizing one aircraft per grid point.

Not covered, of what ``tasopt.f`` also does: the Matlab and gnuplot plot
files, the ASWING export, and the ``.sav`` optimiser restart files.

    python -m tasopt_py /path/to/737.tas [--out 737.out] [--optimise]

or, from Python::

    from tasopt_py.run import run_case
    result = run_case("runs/737/737.tas")
    print(result.WTO_lbf)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

from .aero.airfoil import airtable
from .model import indices as I
from .optimise import OptResult, optimise
from .sizing.noise import noise
from .sizing.woper import WOperResult, woper
from .sizing.wsize import WSizeResult, wsize
from .tasfile import TasCase, apply_sweep, read_tas

__all__ = ["run_case", "RunResult", "LB_N"]

LB_N = 1.0 / 4.44822
#: The acoustic model, if one is available. ``tfnoise.f`` is not ported, so
#: the three certification decibel values stay unset; every other quantity
#: noise.f produces -- the takeoff and cutback engine points, and the observer
#: positions -- is computed.
_TFNOISE = None


@dataclass
class RunResult:
    case: TasCase
    sized: WSizeResult
    off_design: list = field(default_factory=list)   # one WOperResult each
    #: Set when the run optimised rather than just sized.
    optimum: OptResult = None
    #: ``(i_value, j_value)`` for this point, when sweeping.
    sweep_point: tuple = None

    @property
    def fuselage_bl(self):
        return self.sized.fuselage_bl

    @property
    def WTO_lbf(self) -> float:
        return self.case.missions[0].parm[I.IMWTO] * LB_N

    @property
    def Wfuel_lbf(self) -> float:
        return self.case.missions[0].parm[I.IMWFUEL] * LB_N

    @property
    def PFEI(self) -> float:
        """Payload-fuel energy intensity of the design mission."""
        return self.sized.mission.PFEI


def run_case(path, *, Litprint: bool = False, off_design: bool = True,
             optimise_it: bool = None) -> RunResult:
    """Read a ``.tas`` file, size the aircraft, and fly the other missions.

    ``optimise_it`` defaults to the file's own ``Lopt`` flag. When it is true
    the design variables are searched rather than taken as given, and the
    returned result carries the optimum.

    The airfoil database named in the file is loaded once and shared, as
    ``getparm.f`` does -- it reads the same file into two identical tables and
    hands out an index; here it is one table handed to everything.
    """
    case = read_tas(path)
    return run_prepared(case, Litprint=Litprint, off_design=off_design,
                        optimise_it=optimise_it)


def run_sweep(path, *, Litprint: bool = False, **kw):
    """Run every point of a case's ``i``/``j`` parameter sweep.

    Yields one :class:`RunResult` per grid point, in the order ``tasopt.f``
    walks them -- ``j`` innermost. A file with no sequence values is one
    point, and this is then just :func:`run_case`.
    """
    case = read_tas(path)
    ivals = case.parsi or [None]
    jvals = case.parsj or [None]
    for iv in ivals:
        for jv in jvals:
            pt = read_tas(path)
            if iv is not None:
                apply_sweep(pt, case.ispars, iv)
            if jv is not None:
                apply_sweep(pt, case.jspars, jv)
            r = run_prepared(pt, Litprint=Litprint, **kw)
            r.sweep_point = (iv, jv)
            yield r


def run_prepared(case, *, Litprint: bool = False, off_design: bool = True,
                 optimise_it: bool = None) -> RunResult:
    """Run one already-read case. See :func:`run_case`."""
    table = airtable(case.airfoil_file)
    s = case.settings
    if optimise_it is None:
        optimise_it = s.Lopt

    optimum = None
    if optimise_it:
        optimum = optimise(case, table=table, settings=s,
                           Loprint=Litprint or s.Loprint)

    sized = wsize(*case.design, iterwmax=s.iterwmax,
                  wrlx1=s.wrlx1, wrlx2=s.wrlx2, wrlx3=s.wrlx3,
                  initwgt=1 if optimise_it else 0,
                  initeng=1 if optimise_it else 0,
                  table=table, Litprint=Litprint)

    results = []
    if off_design:
        design = case.missions[0]
        for m in case.missions[1:]:
            # Off-design missions start from the design engine state, so
            # initeng is 1 -- as tasopt.f sets it before its woper loop.
            results.append(
                woper(case.pari, case.parg, m.parm, m.para, m.pare,
                      design.para, design.pare, iterfmax=s.iterfmax,
                      initeng=1, table=table, Litprint=Litprint))

    # noise.f is the only routine that runs the engine at the takeoff and
    # cutback points, so nothing can report on them until it has. tasopt.f
    # calls it once per mission, after the off-design loop -- so only for
    # missions that have actually been flown.
    flown = case.missions if off_design else case.missions[:1]
    for m in flown:
        noise(case.pari, case.parg, m.parm, m.para, m.pare, initeng=1,
              table=table, tfnoise=_TFNOISE)
    return RunResult(case=case, sized=sized, off_design=results,
                     optimum=optimum)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out_path = None
    if "--out" in argv:
        k = argv.index("--out")
        if k + 1 >= len(argv):
            print("--out needs a filename", file=sys.stderr)
            return 2
        out_path = argv[k + 1]
        del argv[k:k + 2]
    opt = None
    if "--optimise" in argv:
        argv.remove("--optimise")
        opt = True
    if not argv:
        print("usage: python -m tasopt_py <case.tas> [--out <report>] "
              "[--optimise]", file=sys.stderr)
        return 2
    r = run_case(argv[0], Litprint=True, optimise_it=opt)
    if out_path is not None:
        from .output import report
        with open(out_path, "w") as fh:
            fh.write(report(r.case, r))
        print(f"\n Writing output file:  {out_path}")
    print()
    print(f"{r.case.configname}: {' '.join(r.case.casename)}")
    print(f"  WTO   = {r.WTO_lbf:12.4f} lbf"
          f"   ({'converged' if r.sized.converged else 'NOT CONVERGED'}"
          f" in {r.sized.iterations} iterations)")
    print(f"  Wfuel = {r.Wfuel_lbf:12.4f} lbf")
    print(f"  PFEI  = {r.PFEI:12.6f}")
    if r.optimum is not None:
        o = r.optimum
        print(f"  optimised in {o.steps} steps"
              f" ({'converged' if o.converged else 'step limit'})")
        for k, v in o.variables.items():
            print(f"    {k:<8} = {v:12.6f}")
    for k, o in enumerate(r.off_design, start=2):
        print(f"  mission {k}: WTO = "
              f"{r.case.missions[k - 1].parm[I.IMWTO] * LB_N:12.4f} lbf"
              f"   ({'converged' if o.converged else 'NOT CONVERGED'}"
              f" in {o.iterations} iterations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
