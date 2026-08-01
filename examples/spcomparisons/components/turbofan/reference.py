"""Record the gpkit turbofan solutions as reference.json.

Requires gpkit plus checkouts of
https://github.com/convexengineering/turbofan and .../gplibrary::

    TURBOFAN=~/src/turbofan GPLIBRARY=~/src/gplibrary python reference.py

Solver
------
This model cannot be solved with cvxopt -- the repo's own ``TESTCONFIG`` says
``skipsolvers : cvxopt, mosek_cli``, and the published results came from
MOSEK, which needs a license. So the GP subproblems are solved with IPOPT via
``gpkit_ipopt``, which swaps the numerics while leaving the gpkit model (the
thing the EDI rebuild is verified against) untouched. See that module's
docstring; it reproduces gplibrary's SimPleAC to 1.5e-6 against cvxopt.

Validation data
---------------
``turbofan/test_missions.py:diffs()`` carries measured/published TSFC, engine
weight, and turbine temperatures for the CFM56, the TASOPT 737-800 engine, and
a GE90-like NPSS model. Those are recorded here as ``published``, alongside
the SP model's *own* values from York, Hoburg & Drela, "Turbofan Engine Sizing
and Tradeoff Analysis via Signomial Programming", AIAA J. Aircraft (DOI
10.2514/1.C034463), Tables 9 and 12 -- the numbers this code should reproduce.
The paper is explicit that the model is not expected to match the measured
data exactly; it reports its own errors of -1.91/+9.76/+0.69% for the 737-800.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import save_reference  # noqa: E402

# Engine index -> (name, n_flight_segments). Indices are turbofan's own.
ENGINES = {0: ("CFM56", 2), 1: ("TASOPT_737800", 3), 2: ("GE90", 2),
           3: ("D82", 2)}

# turbofan/test_missions.py diffs(): measured / published engine data.
PUBLISHED = {
    "CFM56": {"TSFC[0]": 0.6793, "TSFC[1]": 0.6941, "W_engine_lbf": 5216.0},
    "TASOPT_737800": {
        "TSFC[0]": 0.48434, "TSFC[1]": 0.65290, "TSFC[2]": 0.64009,
        "W_engine_lbf": 7870.7,
        "T_t4.1[0]": 1658.7, "T_t4.1[1]": 1605.4, "T_t4.1[2]": 1433.8,
        "cooling_dT[0]": 174.3, "cooling_dT[1]": 178.4, "cooling_dT[2]": 153.2,
    },
    "GE90": {"TSFC[0]": 0.5418, "TSFC[1]": 0.5846, "W_engine_lbf": 17400.0},
    "D82": {},
}

# The SP model's own published values -- what this code should reproduce.
# Table 9 (737-800, engine weight capped at 110% of TASOPT) and Table 12
# (GE90 vs NPSS). Note Table 12 lists the NPSS TOC TSFC as 0.5876 while
# diffs() uses 0.5846; one of the two is a transcription slip.
PAPER_SP = {
    "TASOPT_737800": {"TSFC[0]": 0.4751, "TSFC[1]": 0.7166, "TSFC[2]": 0.6445},
    "GE90": {"TSFC[0]": 0.5328, "TSFC[1]": 0.5997},
}


def main() -> None:
    checkouts = ("TURBOFAN", "GPLIBRARY")
    for var in checkouts:
        if not os.environ.get(var):
            raise SystemExit(f"set {var} to a checkout")
    # Order matters: put the owning repo first. turbofan and SPaircraft both
    # ship top-level stand_alone_simple_profile.py and simple_ac_imports.py,
    # and both are imported unqualified, so whichever checkout lands earlier
    # on sys.path silently wins. See spaircraft/reference.py, where getting
    # this backwards moved fuel burn 2.2% with no warning.
    for var in reversed(checkouts):
        sys.path.insert(0, os.environ[var])

    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import gpkit_ipopt
    from gpkit import Model, Vectorize
    from gpkit.small_scripts import mag

    from turbofan.engine_validation import Engine, TestState
    from turbofan.initial_guess import initialize_guess
    from turbofan.test_missions import (TestMissionCFM, TestMissionTASOPT,
                                        TestMissionGE90, TestMissionD82)
    from turbofan.subs import (get_cfm56_subs, get_737800_subs,
                               get_ge90_subs, get_D82_subs)

    missions = {0: TestMissionCFM, 1: TestMissionTASOPT,
                2: TestMissionGE90, 3: TestMissionD82}
    getters = {0: get_cfm56_subs, 1: get_737800_subs,
               2: get_ge90_subs, 3: get_D82_subs}

    out = {}
    for eng, (name, N) in ENGINES.items():
        with Vectorize(N):
            state = TestState()
        engine = Engine(True, N, state, eng)
        subs = getters[eng]()

        # engine_validation.__main__ weights the first segment's TSFC by 10.
        weights = [10.0] + [1.0] * (N - 1)
        cost = engine["W_{engine}"] ** 0.5 * np.dot(
            weights, engine.engineP.thrustP["TSFC"])

        m = Model(cost, [engine, missions[eng](engine)], subs,
                  x0=initialize_guess())
        m.substitutions.update(subs)
        gpkit_ipopt.reset_cache()
        sol = m.localsolve(verbosity=0, mutategp=False, reltol=1e-6,
                           solver=gpkit_ipopt.optimize)

        tsfc = np.atleast_1d(mag(sol("TSFC"))).ravel()
        tt41 = np.atleast_1d(mag(sol("T_{t_{4.1}}"))).ravel()
        tt4 = np.atleast_1d(mag(sol("T_{t_4}"))).ravel()
        checked = {"W_engine_lbf": float(mag(sol("W_{engine}").to("lbf")))}
        checked.update({f"TSFC[{i}]": float(v) for i, v in enumerate(tsfc)})
        checked.update({f"T_t4.1[{i}]": float(v) for i, v in enumerate(tt41)})
        checked.update({f"cooling_dT[{i}]": float(tt4[i] - tt41[i])
                        for i in range(len(tt41))})

        values = {}
        for key, val in sol["variables"].items():
            try:
                values[str(key)] = float(val)
            except (TypeError, ValueError):
                try:
                    values[str(key)] = [float(x)
                                        for x in np.atleast_1d(val).ravel()]
                except Exception:
                    pass
        out[name] = {"cost": float(sol["cost"]), "checked": checked,
                     "published": PUBLISHED[name],
                     "paper_sp": PAPER_SP.get(name, {}),
                     "values": values}
        print(f"{name}: cost={float(sol['cost']):.6g}, "
              f"{len(values)} values")

    save_reference(Path(__file__).with_name("reference.json"), out,
                   source="convexengineering/turbofan @ master, "
                          "solved with gpkit_ipopt (see module docstring)",
                   meta={"reltol": 1e-6, "mutategp": False,
                         "x0": "turbofan.initial_guess.initialize_guess()"})


if __name__ == "__main__":
    main()
