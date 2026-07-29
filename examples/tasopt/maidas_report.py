"""Drive MAIDAS structure detection over the TASOPT port, module by module.

Usage::

    python maidas_report.py              # every registered target
    python maidas_report.py fuselage     # one target

Each target names a TASOPT routine, the discrete *selectors* to bind (gas
index, planform code -- things that choose a configuration rather than vary
continuously), and the SPaircraft module it should be compared against.

Writes nothing; prints a table. ``MAIDAS_REPORT.md`` is the written-up
version, updated by hand as findings accumulate.
"""
from __future__ import annotations

import functools
import sys
import warnings

warnings.filterwarnings("ignore")

from maidas.core import analyze, autobranch, classify_flat  # noqa: E402

#: Tightest-first. A quantity is reported at the first label that applies.
ORDER = ("constant", "monomial", "posynomial", "gp_representable",
         "signomial", "not_gp")


def _flat_labels(x, acc=None):
    """Flatten ``classify_flat`` output.

    It maps element-wise over containers, so a routine returning a tuple gives
    back a tuple of label-lists rather than a flat list.
    """
    acc = set() if acc is None else acc
    if isinstance(x, str):
        acc.add(x)
    elif isinstance(x, dict):
        for v in x.values():
            _flat_labels(v, acc)
    elif isinstance(x, (list, tuple, set)):
        for v in x:
            _flat_labels(v, acc)
    return acc


def _import(path):
    mod, _, name = path.rpartition(".")
    return getattr(__import__(f"tasopt_py.{mod}", fromlist=[name]), name)


def classify_routine(path, selectors=None):
    """``(n_intermediates, {name: verdict})`` for one TASOPT routine.

    ``selectors`` are bound with ``functools.partial`` so they stay concrete:
    a planform code, a web count, a gas index is a discrete choice, and
    tracing it symbolically only yields a Proxy that later dies in a table
    lookup or an ``int()``.

    ``autobranch`` is applied first and the selectors bound over the result,
    so a routine that both branches and takes a selector is handled. Branch
    leaves report their own taps via ``.branch_taps``; verdicts are merged
    across leaves, taking the loosest, since a quantity is only as GP-clean
    as its worst branch.
    """
    fn = _import(path)
    try:
        fn = autobranch(fn)
    except NotImplementedError:
        pass
    if selectors:
        fn = functools.partial(fn, **selectors)
    ir = analyze(fn)

    dicts = []
    if hasattr(ir, "items"):
        dicts.append(ir)
    for t in (getattr(ir, "branch_taps", None) or ()):
        if hasattr(t, "items"):
            dicts.append(t)
    if not dicts:
        # A function that is one expression with no named intermediates is
        # not a failure -- there is simply nothing to tap. Classify the
        # returned value itself.
        try:
            lbls = _flat_labels(classify_flat(ir))
            v = next((o for o in ORDER if o in lbls), "other")
        except Exception as exc:
            v = f"ERR:{type(exc).__name__}"
        return 1, {"<return>": v}

    def rank(v):
        return ORDER.index(v) if v in ORDER else len(ORDER)

    out = {}
    for d in dicts:
        for name, sub in d.items():
            key = name.split("__")[-1]
            try:
                lbls = _flat_labels(classify_flat(sub))
                v = next((o for o in ORDER if o in lbls), "other")
            except Exception as exc:
                v = f"ERR:{type(exc).__name__}"
            prev = out.get(key)
            # Loosest wins: a quantity is only as GP-clean as its worst branch.
            if prev is None or rank(v) > rank(prev):
                out[key] = v
    return len(out), out


#: name -> (routine, selectors, spaircraft counterpart, quantities of interest)
TARGETS = {
    "wing": ("structures.surface.surfw", {"iwplan": 0}, "wingbox.py + wing.py",
             ("tbcap", "tbweb", "Abcap", "Abweb", "EIc", "EIn", "GJ",
              "hrms", "havg", "Ss", "Ms", "So", "Mo", "Ws")),
    "fuselage": ("structures.fuselage.fusew", {"nfweb": 1}, "fuselage.py (83 cons)",
                 ("tskin", "tcone", "tfweb", "tfloor", "Wtail", "Wshell",
                  "Wwindow", "Winsul", "Wfloor", "Wcone", "Ihshell",
                  "Ivshell", "sigx", "sigth", "rMh", "rMv", "xhbend",
                  "xvbend", "Ahbendf", "Avbendf", "Wh", "Wv")),
    "landing_gear": ("structures.landing_gear.size_landing_gear", {},
                     "landing_gear.py (46 cons)",
                     ("W", "l", "d", "n", "t")),
    "tail_load": ("structures.planform.tailpo", {}, "horizontal/vertical_tail",
                  ("po", "pl")),
    "wing_scale": ("structures.planform.wingsc", {}, "wing.py",
                   ("S", "b", "co", "cs", "ct")),
    "chord_int": ("structures.planform.chord_integrals", {}, "wing.py",
                  ("Kc", "Kp", "Ks", "Kco")),
    "surface_drag": ("aero.drag.surfcd", {}, "model.py drag build-up",
                     ("CDsurf", "CDfric", "CDover", "cd")),
    "wing_load": ("aero.loading.wingpo", {}, "wing.py load case",
                  ("po", "pl", "N")),
    "wing_cl": ("aero.loading.wingcl", {}, "model.py lift",
                ("CL", "cl", "S")),
    "planform_int": ("aero.loading.planform_integrals", {}, "wing.py",
                     ("K", "S", "b")),
    "surf_moment": ("aero.moment.surfcm", {}, "model.py pitching moment",
                    ("CM", "cm")),
    "turb_friction": ("aero.cdsum.cfturb", {}, "model.py Cf",
                      ("cf", "Re")),
    "induced_drag": ("aero.cdsum.cditrp", {}, "model.py induced drag",
                     ("CDi", "e")),
    "cool_flow": ("engine.cooling.mcool", {}, "(none)",
                  ("m", "eta", "T")),
    "cool_temp": ("engine.cooling.tmcalc", {}, "(none)",
                  ("Tm", "T")),
    "bl_dissipation": ("aero.blclosure.dil", {}, "(none -- York fits)",
                       ("di",)),
    "bl_shape": ("aero.blclosure.hsl", {}, "(none)", ("hs",)),
    "bl_friction": ("aero.blclosure.cfl", {}, "(none)", ("cf",)),
    "atmos": ("atmosphere.atmos", {}, "flight_state.py",
              ("T", "p", "rho", "a", "mu")),
    "engine_size": ("engine.tfsize.tfsize", {}, "(none -- York fits instead)",
                    ("mcore", "A", "Tt", "pt", "ht", "u", "Fsp", "TSFC")),
}


def report(name):
    path, sel, counterpart, want = TARGETS[name]
    n, verdicts = classify_routine(path, sel)
    tally = {}
    for v in verdicts.values():
        tally[v] = tally.get(v, 0) + 1
    print(f"\n=== {name}: {path} ===")
    print(f"    SPaircraft counterpart: {counterpart}")
    print(f"    {n} distinct intermediates")
    for k in ORDER + ("other",):
        if k in tally:
            print(f"      {tally[k]:4d}  {k}")
    hits = [(q, v) for q, v in verdicts.items()
            if any(q.startswith(w) for w in want)]
    if hits:
        print("    quantities with an SPaircraft counterpart:")
        for q, v in hits:
            print(f"      {q:12s} {v}")
    return tally


def main(argv):
    names = argv[1:] or list(TARGETS)
    for nm in names:
        if nm not in TARGETS:
            print(f"unknown target {nm!r}; known: {sorted(TARGETS)}")
            continue
        report(nm)


if __name__ == "__main__":
    main(sys.argv)
