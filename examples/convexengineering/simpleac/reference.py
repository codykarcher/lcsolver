"""Record the gpkit SimPleAC solution as reference.json.

Requires gpkit and a checkout of https://github.com/convexengineering/gplibrary.
Point GPLIBRARY at it::

    GPLIBRARY=~/src/gplibrary python reference.py

The resulting reference.json is committed, so verifying the LCsolver rebuild does
not require gpkit.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import save_reference  # noqa: E402


def main() -> None:
    gplib = os.environ.get("GPLIBRARY")
    if not gplib:
        raise SystemExit("set GPLIBRARY to a gplibrary checkout")
    sys.path.insert(0, str(Path(gplib) / "gpkitmodels" / "SP" / "SimPleAC"))

    import SimPleAC as S  # noqa: E402
    import gpkit  # noqa: E402

    m = S.SimPleAC()
    m.cost = m["W_f"]
    sol = m.localsolve(verbosity=0)

    values = {}
    for key, val in sol["freevariables"].items():
        name = str(key).split(".")[-1]
        try:
            values[name] = float(val)
        except (TypeError, ValueError):
            continue

    save_reference(
        Path(__file__).with_name("reference.json"),
        values,
        source="gpkit localsolve of gplibrary SP/SimPleAC/SimPleAC.py",
        meta={
            "gpkit_version": gpkit.__version__,
            "objective": "W_f",
            "cost": float(sol["cost"]),
            "units": "values are in each variable's declared units "
                     "(SI except T_flight in hr)",
        },
    )


if __name__ == "__main__":
    main()
