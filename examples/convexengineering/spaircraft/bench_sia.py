"""Regenerate the SPaircraft rows of Table 1 in the SIA paper.

    python -m spaircraft.bench_sia

Reports, for each of the three decks, exactly what the table's caption says it
reports: variables and rows counted **after unit correction and structure
detection, before presolve**, then a solve from the model's own published
starting guess terminating on the KKT certificate rather than an iteration cap.

    structure_detector(unit_corrector(build()))  ->  solve_sia(...)

The three decks are the same airframe in three configurations, built from one
component library and differing only in structural layout, so together they
isolate the effect of problem *structure* at nearly fixed problem size. That is
what makes them worth three rows rather than one:

    D8.2           rear engines, double bubble, pi-tail        140 iterations
    D8, no BLI     the same, clean inlet                        85
    conventional   wing engines, tube, conventional tail       353

The conventional aircraft takes two and a half times the D8.2's iterations at
one fewer variable and the same row count. Its horizontal tail is mounted on
the fuselage, so the root moment is a sum of positive terms; the D8.2's pi-tail
is carried on two fins and the source writes its centreline moment as a
difference, which is signomial and can very nearly cancel.

Both new decks are built with their default ``faithful=False``, so the two
upstream defects in :mod:`~.defects` are corrected. That changes the objective
by about 1% on the no-BLI deck and does not change its structure.
"""
from __future__ import annotations

import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcsolver.presolve import structure_detector, unit_corrector          # noqa: E402
from lcsolver.solvers.ipopt.slcp_bridge import solve_sia                  # noqa: E402

LBF_N = 4.44822161526

#: label -> import path of the deck's ``build``
DECKS = [
    ("SPaircraft, D8.2",  "spaircraft.model"),
    ("  no BLI",          "spaircraft.d8_no_bli"),
    ("  conventional",    "spaircraft.conventional"),
]

HEADER = (f"{'Problem':22} {'Vars':>5} {'Rows':>6} {'Iter':>5} {'Time(s)':>8} "
          f"{'Station.':>10} {'Violation':>10} {'Fuel(lbf)':>11} {'Cert':>5}")


def bench(label: str, module: str) -> dict | None:
    """One row. Returns None and prints the traceback if the deck fails."""
    build = __import__(module, fromlist=["build"]).build
    try:
        st = structure_detector(unit_corrector(build()))
        t0 = time.time()
        res = solve_sia(st)
        dt = time.time() - t0
    except Exception:
        print(f"{label:22} FAILED", flush=True)
        traceback.print_exc()
        return None
    row = dict(label=label, vars=st.n_variables, rows=len(st.operators),
               iters=res.iterations, time=dt, stat=res.stationarity,
               viol=res.max_violation, fuel=res.objective / LBF_N,
               cert=res.converged)
    print(f"{row['label']:22} {row['vars']:5d} {row['rows']:6d} "
          f"{row['iters']:5d} {row['time']:8.2f} {row['stat']:10.1e} "
          f"{row['viol']:10.1e} {row['fuel']:11.1f} {str(row['cert']):>5}",
          flush=True)
    return row


def latex(rows) -> str:
    """The rows as they appear in the paper's tabular, for pasting."""
    out = []
    for r in rows:
        name = r["label"].strip()
        name = f"\\quad {name}" if r["label"].startswith("  ") else name
        out.append(f"{name:24} & {r['vars']} & {r['rows']} & {r['iters']} & "
                   f"{r['time']:.2f} & \\num{{{r['stat']:.1e}}} & "
                   f"\\num{{{r['viol']:.1e}}} \\\\")
    return "\n".join(out)


def main() -> None:
    print(HEADER, flush=True)
    rows = [r for r in (bench(*d) for d in DECKS) if r]
    if rows:
        print("\n% Table 1 rows:")
        print(latex(rows))


if __name__ == "__main__":
    main()
