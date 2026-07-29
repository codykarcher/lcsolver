"""Extract the hydrogen gas-property table from TASOPT.jl's ``gasdata.jl``.

TASOPT 2.16 has eleven species and **cannot burn hydrogen at all**: there is
no ``gas_H2`` in ``gasfun.f``, and ``gaschem`` stops at ``igas = 24`` and
calls ``stop`` on anything else. TASOPT.jl adds a twelfth species, ``H2`` at
``igas = 40``, in exactly the same tabulated form the Fortran uses -- t, log
t, cp, dcp/dt, h and s at each of 45 temperatures.

This lifts that table out of the Julia source rather than transcribing it,
for the same reason ``fortran_ref/extract_gas_tables.py`` parses the Fortran
DATA blocks: 45 x 6 hand-copied numbers is 270 chances at a typo that would
be invisible in an answer that still looked plausible.

The emitted entry is appended to :mod:`tasopt_py.gas.tables` in the same
shape as the eleven from ``gasfun.f``, so nothing downstream needs to know
where it came from.

Run as::

    python tools/gen_h2_table.py /path/to/TASOPT.jl/src/engine/gasdata.jl
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

#: ``const H2 = simple_gas(t=[...], tl=[...], ...)``. The call is matched by
#: balancing parentheses rather than by regex -- the H2 block is the last
#: thing in the file and ends ``])`` with no trailing newline, so a pattern
#: anchored on a closing delimiter misses it.
START = re.compile(r"const\s+H2\s*=\s*simple_gas\(")
FIELD = re.compile(r"(\w+)\s*=\s*\[(.*?)\]", re.S)


def _call_body(text, m):
    """The text between ``simple_gas(`` and its matching ``)``."""
    depth, i = 1, m.end()
    while i < len(text) and depth:
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        i += 1
    if depth:
        raise SystemExit("unbalanced parentheses in the H2 block")
    return text[m.end():i - 1]

#: ``r`` and ``hform`` are arguments to ``get_thermo`` in ``gas_H2``, not part
#: of the data block.
GAS_H2 = re.compile(r"function\s+gas_H2\(.*?\n\s*r\s*=\s*([\d.eE+-]+)"
                    r".*?hform\s*=\s*([\d.eE+-]+)", re.S)


def parse(gasdata_path, gasfun_path):
    text = Path(gasdata_path).read_text()
    m = START.search(text)
    if not m:
        raise SystemExit(f"no `const H2 = simple_gas(...)` in {gasdata_path}")

    cols = {}
    for name, body in FIELD.findall(_call_body(text, m)):
        cols[name] = [float(v) for v in body.replace("\n", " ").split(",")
                      if v.strip()]

    missing = {"t", "tl", "cp", "cpt", "h", "s"} - set(cols)
    if missing:
        raise SystemExit(f"H2 table is missing {sorted(missing)}")
    n = len(cols["t"])
    for name, v in cols.items():
        if len(v) != n:
            raise SystemExit(f"H2 column {name!r} has {len(v)} entries, "
                             f"but t has {n}")

    fn = GAS_H2.search(Path(gasfun_path).read_text())
    if not fn:
        raise SystemExit(f"no `function gas_H2` with r/hform in {gasfun_path}")
    r, hform = float(fn.group(1)), float(fn.group(2))
    return n, r, hform, cols


def emit(n, r, hform, cols) -> str:
    out = ['    "H2": {',
           f'        "ndim": {n}, "r": {r}, "hform": {hform},']
    for name in ("t", "tl", "cp", "cpt", "h", "s"):
        vals = ", ".join(repr(v) for v in cols[name])
        out.append(f'        "{name}": [{vals}],')
    out.append("    },")
    return "\n".join(out)


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1
               else "/tmp/tjl/src/engine/gasdata.jl")
    fun = src.with_name("gasfun.jl")
    n, r, hform, cols = parse(src, fun)

    dst = (Path(__file__).resolve().parent.parent
           / "tasopt_py/gas/tables.py")
    text = dst.read_text()
    entry = emit(n, r, hform, cols)

    if '"H2": {' in text:
        # Replace the existing entry rather than appending a second one.
        text = re.sub(r'    "H2": \{.*?\n    \},\n', entry + "\n",
                      text, count=1, flags=re.S)
    else:
        # GASES is sorted by key; H2 sorts after C8H18 and before H2O.
        text = text.replace('    "H2O": {', entry + '\n    "H2O": {', 1)
    dst.write_text(text)
    print(f"wrote H2 ({n} points, r = {r}, hform = {hform}) into {dst}")
