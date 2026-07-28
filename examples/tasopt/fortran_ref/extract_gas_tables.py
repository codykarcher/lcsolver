"""Extract the tabulated gas property data from TASOPT ``src/gasfun.f``.

Each ``gas_XX`` subroutine carries six parallel DATA arrays (t, tl, cp, cpt,
h, s) plus scalars r and hform. Transcribing ~600 numbers per gas by hand
across 11 gases would be a reliable source of typos, so they are parsed
straight out of the Fortran and emitted as a Python module.

Usage::

    python extract_gas_tables.py <path-to-gasfun.f> > ../tasopt_py/gas/tables.py
"""
from __future__ import annotations

import re
import sys

SUB_RE = re.compile(r"^\s{6}subroutine\s+(gas_\w+)\s*\(", re.IGNORECASE)
END_RE = re.compile(r"^\s{6}end\s*!", re.IGNORECASE)
NDIM_RE = re.compile(r"parameter\s*\(\s*ndim\s*=\s*(\d+)\s*\)", re.IGNORECASE)


def unfold(lines: list[str]) -> list[str]:
    """Join Fortran-77 continuation lines (any non-blank in column 6)."""
    out: list[str] = []
    for ln in lines:
        body = ln.rstrip("\n")
        if len(body) > 5 and body[5] not in (" ", "0") and body[:5].strip() == "":
            if out:
                out[-1] += " " + body[6:].strip()
                continue
        out.append(body)
    return out


def numbers(text: str) -> list[float]:
    """Parse Fortran reals, including the e0/E0 and D exponent forms."""
    toks = re.findall(r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?", text)
    return [float(t.replace("d", "e").replace("D", "e")) for t in toks]


def parse(path: str) -> dict:
    with open(path) as fh:
        lines = unfold(fh.readlines())

    gases: dict = {}
    name = None
    buf: list[str] = []
    for ln in lines:
        m = SUB_RE.match(ln)
        if m:
            name, buf = m.group(1), []
            continue
        if name is None:
            continue
        if END_RE.match(ln):
            gases[name] = parse_body(buf)
            name = None
            continue
        buf.append(ln)
    return {k: v for k, v in gases.items() if v}


def parse_body(buf: list[str]) -> dict | None:
    text = "\n".join(buf)
    m = NDIM_RE.search(text)
    if not m:
        return None
    ndim = int(m.group(1))

    out: dict = {"ndim": ndim}
    for ln in buf:
        s = ln.strip()
        if not s.lower().startswith("data "):
            continue
        m2 = re.match(r"data\s+([\w\s,]+?)\s*/(.*)/\s*$", s, re.IGNORECASE)
        if not m2:
            continue
        names = [n.strip().lower() for n in m2.group(1).split(",")]
        vals = numbers(m2.group(2))
        if names == ["r", "hform"]:
            out["r"], out["hform"] = vals[0], vals[1]
        elif len(names) == 1 and names[0] in ("t", "tl", "cp", "cpt", "h", "s"):
            if len(vals) != ndim:
                raise SystemExit(
                    f"array {names[0]}: got {len(vals)} values, ndim={ndim}")
            out[names[0]] = vals
    need = {"t", "tl", "cp", "cpt", "h", "s", "r", "hform"}
    return out if need <= set(out) else None


def main() -> None:
    path = sys.argv[1]
    gases = parse(path)

    print('"""Gas property tables — generated from TASOPT src/gasfun.f.')
    print()
    print("DO NOT EDIT BY HAND. Regenerate with:")
    print("    python fortran_ref/extract_gas_tables.py <gasfun.f> "
          "> tasopt_py/gas/tables.py")
    print()
    print("Each entry holds the tabulation used by TASOPT's cubic Hermite")
    print("interpolation: temperature t [K], its log tl, specific heat cp")
    print("[J/kg-K] with slope cpt [J/kg-K^2], enthalpy h [J/kg] and entropy")
    print("complement s [J/kg-K], plus the gas constant r and formation")
    print("enthalpy hform.")
    print('"""')
    print()
    print("GASES = {")
    for name in sorted(gases):
        g = gases[name]
        key = name[4:].upper()
        print(f'    "{key}": {{')
        print(f'        "ndim": {g["ndim"]}, "r": {g["r"]!r}, "hform": {g["hform"]!r},')
        for arr in ("t", "tl", "cp", "cpt", "h", "s"):
            print(f'        "{arr}": {g[arr]!r},')
        print("    },")
    print("}")
    print()
    print("__all__ = [\"GASES\"]")

    print(f"\n# extracted {len(gases)} gases: {', '.join(sorted(gases))}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
