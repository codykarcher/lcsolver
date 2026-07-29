"""Run a ``.tas`` case end to end -- the port's equivalent of ``tasopt.f``.

``tasopt.f`` does a good deal besides: parameter sweeps, an optimiser, plot
and save files, five output formats. This is only the path a plain
``tasopt 737`` takes with ``Lopt = F``: read the file, size the aircraft for
the design mission, then fly each remaining mission off-design.

    python -m tasopt_py /path/to/737.tas [--out 737.out]

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
from .sizing.noise import noise
from .sizing.woper import WOperResult, woper
from .sizing.wsize import WSizeResult, wsize
from .tasfile import TasCase, read_tas

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


def run_case(path, *, Litprint: bool = False,
             off_design: bool = True) -> RunResult:
    """Read a ``.tas`` file, size the aircraft, and fly the other missions.

    The airfoil database named in the file is loaded once and shared, as
    ``getparm.f`` does -- it reads the same file into two identical tables and
    hands out an index; here it is one table handed to everything.
    """
    case = read_tas(path)
    table = airtable(case.airfoil_file)
    s = case.settings

    sized = wsize(*case.design, iterwmax=s.iterwmax,
                  wrlx1=s.wrlx1, wrlx2=s.wrlx2, wrlx3=s.wrlx3,
                  initwgt=0, initeng=0, table=table, Litprint=Litprint)

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
    return RunResult(case=case, sized=sized, off_design=results)


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
    if not argv:
        print("usage: python -m tasopt_py <case.tas> [--out <report>]",
              file=sys.stderr)
        return 2
    r = run_case(argv[0], Litprint=True)
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
    for k, o in enumerate(r.off_design, start=2):
        print(f"  mission {k}: WTO = "
              f"{r.case.missions[k - 1].parm[I.IMWTO] * LB_N:12.4f} lbf"
              f"   ({'converged' if o.converged else 'NOT CONVERGED'}"
              f" in {o.iterations} iterations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
