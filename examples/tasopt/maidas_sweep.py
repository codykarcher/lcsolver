"""Sweep MAIDAS over every scalar-signature function in the TASOPT port.

Buckets the outcome so that MAIDAS *bugs* separate from MAIDAS *limits*:
a TraceError on data-dependent control flow is a limit; anything else is
worth looking at.
"""
import importlib, inspect, pkgutil, sys, traceback, warnings
warnings.filterwarnings("ignore")

from maidas.core import analyze, classify_flat

import tasopt_py

SKIP_MODULES = {"plot", "run", "__main__", "tasfile", "output", "aswing"}


def scalar_signature(fn):
    """True if every parameter is annotated float (or unannotated w/ float default)."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return False
    if not sig.parameters:
        return False
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            return False
        ann = p.annotation
        if ann is float or ann == "float":
            continue
        if ann is inspect.Parameter.empty and isinstance(p.default, float):
            continue
        if ann is int or ann == "int":
            continue
        return False
    return True


def walk(node, out):
    if isinstance(node, dict):
        for v in node.values():
            walk(v, out)
    elif isinstance(node, (list, tuple)):
        for v in node:
            walk(v, out)
    else:
        out.append(node)


def main():
    mods = []
    for m in pkgutil.walk_packages(tasopt_py.__path__, "tasopt_py."):
        short = m.name.rsplit(".", 1)[-1]
        if short in SKIP_MODULES:
            continue
        try:
            mods.append(importlib.import_module(m.name))
        except Exception:
            pass

    seen, rows = set(), []
    for mod in mods:
        for name, fn in vars(mod).items():
            if not inspect.isfunction(fn) or name.startswith("_"):
                continue
            if fn.__module__ != mod.__name__ or fn in seen:
                continue
            if not scalar_signature(fn):
                continue
            seen.add(fn)
            key = f"{mod.__name__.replace('tasopt_py.','')}.{name}"
            try:
                ir = analyze(fn)
                nodes = []
                walk(ir, nodes)
                labels = set()
                for n in nodes:
                    try:
                        labels |= set(classify_flat(n))
                    except Exception as e:
                        labels.add(f"CLASSIFY_ERR:{type(e).__name__}")
                rows.append((key, "ok", sorted(labels)))
            except Exception as e:
                rows.append((key, type(e).__name__, [str(e)[:90]]))

    ok = [r for r in rows if r[1] == "ok"]
    trace = [r for r in rows if r[1] == "TraceError"]
    other = [r for r in rows if r[1] not in ("ok", "TraceError")]
    cerr = [r for r in ok if any(str(l).startswith("CLASSIFY_ERR") for l in r[2])]

    print(f"{len(rows)} scalar-signature functions traced\n")
    print(f"  {len(ok):3d} analysed        ({len(cerr)} with classify errors)")
    print(f"  {len(trace):3d} TraceError     (data-dependent control flow -- a limit, not a bug)")
    print(f"  {len(other):3d} other failures (candidate bugs)\n")

    if other:
        print("=== OTHER FAILURES ===")
        for k, t, msg in other:
            print(f"  {k:46s} {t}: {msg[0]}")
    if cerr:
        print("\n=== CLASSIFY ERRORS ===")
        for k, _, l in cerr:
            print(f"  {k:46s} {[x for x in l if str(x).startswith('CLASSIFY_ERR')]}")

    print("\n=== ANALYSED: label tally ===")
    tally = {}
    for _, _, labels in ok:
        for l in labels:
            tally[l] = tally.get(l, 0) + 1
    for l, c in sorted(tally.items(), key=lambda t: -t[1]):
        print(f"  {c:3d}  {l}")
    return rows


if __name__ == "__main__":
    main()
