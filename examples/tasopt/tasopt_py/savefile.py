"""Optimiser restart files -- ``getsave.f``.

An optimisation is expensive: every objective evaluation is a whole aircraft,
which in this port is about fifteen seconds. The ``.sav`` file lets a search
be picked up where it stopped, by storing the *entire simplex* -- all
``nvar + 1`` vertices -- for each point of the ``i``/``j`` parameter grid,
along with enough of the setup to refuse a file that does not match the case
being run.

Format
------
::

    nvar ispars jspars                  1x,60i4
    iovar(1..nvar)                      1x,60i4     -- cparo indices
    <blank>                             one per grid point from here
    i j                                 1x,60i4
    vertex 1: v(1..nvar)                1x,60g22.14
    vertex 2: v(1..nvar)
    ...
    vertex nvar+1: v(1..nvar)

The variable list is stored as 1-based indices into ``cparo``, so a file
records *which* variables were being optimised, not just how many.
:func:`read_sav` refuses a file whose variable count, sweep parameters or
variable selection differ from the case in hand, and warns if the grid size
does, exactly as ``getsave.f`` does.

The shipped program cannot write a usable one
---------------------------------------------
``tasopt.f`` opens the ``.sav`` file, writes the two header lines with
``wrtsave0``, and then **closes it** -- before the optimisation that produces
the data has run. Every later ``wrtsave1`` call therefore writes to a closed
unit 8, which gfortran silently redirects to a file called ``fort.8`` in the
working directory. The result is a ``.sav`` holding only its header, an
unexpected ``fort.8`` holding all the data, and a restart that fails on the
grid-size check because it read no grid points at all.

That is a bug in the *driver*, not in these routines, and this port does not
reproduce it: :func:`write_sav` writes the header and the body to the same
file. The routines themselves are ported as written, and both halves are
checked against real Fortran output -- the header against a ``.sav``, the body
against the ``fort.8`` the same run left behind.

Verified against the compiled Fortran; see ``tests/test_savefile.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .model import indices as I
from .output import _gfmt

__all__ = ["read_sav", "write_sav", "SaveFile", "SaveMismatch"]

#: ``getsave.f``'s local array bound on the number of variables.
NVMAX = 50


class SaveMismatch(Exception):
    """The save file does not describe the case being run.

    ``getsave.f`` sets its ``ferr`` flag and returns; the caller then falls
    back to a cold start. Raising is the equivalent, and lets a caller do the
    same by catching it.
    """


@dataclass
class SaveFile:
    """One ``.sav`` file: the simplex at every grid point."""
    variables: list = field(default_factory=list)     # cparo keywords
    ispars: str = ""
    jspars: str = ""
    #: ``{(i, j): [vertex, ...]}``, 1-based grid indices, ``nvar+1`` vertices.
    simplexes: dict = field(default_factory=dict)

    @property
    def ni(self):
        return max((i for i, _ in self.simplexes), default=0)

    @property
    def nj(self):
        return max((j for _, j in self.simplexes), default=0)


def _iovar_indices(names):
    """``cparo`` keywords to the 1-based indices the file stores."""
    out = []
    for n in names:
        if n not in I.CPARO:
            raise SaveMismatch(f"{n!r} is not an optimisation variable")
        out.append(I.CPARO.index(n) + 1)
    return out


def _i4(values):
    """``format(1x,60i4)``."""
    return " " + "".join(f"{int(v):4d}" for v in values)


def _g22(values):
    """``format(1x,60g22.14)``."""
    return " " + "".join(_gfmt(v, 22, 14) for v in values)


def write_sav(path, variables, ispars, jspars, simplexes) -> None:
    """Write a restart file.

    ``simplexes`` is ``{(i, j): [vertex, ...]}`` with ``nvar+1`` vertices per
    point, each a list of ``nvar`` values in ``variables`` order. Grid points
    are written in ``i`` then ``j`` order, as ``wrtsave`` walks them.
    """
    nvar = len(variables)
    iov = _iovar_indices(variables)
    isp = I.CPARS.index(ispars) + 1 if ispars else 0
    jsp = I.CPARS.index(jspars) + 1 if jspars else 0

    lines = [_i4([nvar, isp, jsp]), _i4(iov)]
    for i in sorted({k[0] for k in simplexes}):
        for j in sorted({k[1] for k in simplexes if k[0] == i}):
            verts = simplexes[(i, j)]
            if len(verts) != nvar + 1:
                raise ValueError(
                    f"grid point ({i},{j}) has {len(verts)} vertices, "
                    f"expected {nvar + 1}")
            # `write(lu,*) ' '` -- list-directed, so a blank plus the blank.
            lines.append("  ")
            lines.append(_i4([i, j]))
            for v in verts:
                lines.append(_g22(v))
    Path(path).write_text("".join(ln + "\n" for ln in lines))


def read_sav(path, variables, ispars, jspars, ni=None, nj=None) -> SaveFile:
    """Read a restart file, checking it describes the case in hand.

    Raises :class:`SaveMismatch` where ``getsave.f`` sets ``ferr`` and
    returns: a different number of variables, different sweep parameters, a
    different variable selection, or more variables than it can hold. A grid
    size that differs from ``ni``/``nj`` is also a mismatch -- the Fortran
    reports it after reading, so the data is loaded first and can be inspected
    on the exception.
    """
    nvar = len(variables)
    want_iov = _iovar_indices(variables)
    want_isp = I.CPARS.index(ispars) + 1 if ispars else 0
    want_jsp = I.CPARS.index(jspars) + 1 if jspars else 0

    tok = Path(path).read_text().split()
    pos = 0

    def take(n, cast=float):
        nonlocal pos
        if pos + n > len(tok):
            raise SaveMismatch("unexpected end of save file")
        v = [cast(t) for t in tok[pos:pos + n]]
        pos += n
        return v

    nvar1, isp1, jsp1 = take(3, int)
    problems = []
    if nvar1 != nvar:
        problems.append("number of optimisation variables "
                        f"({nvar1} in the file, {nvar} in the run)")
    if isp1 != want_isp:
        problems.append("type of i parameter")
    if jsp1 != want_jsp:
        problems.append("type of j parameter")
    if problems:
        raise SaveMismatch("save file mismatch in " + "; ".join(problems))

    if nvar > NVMAX:
        raise SaveMismatch(f"getsave: local array overflow, increase nvmax "
                           f"to {nvar}")

    iov1 = take(nvar, int)
    if iov1 != want_iov:
        raise SaveMismatch("save file mismatch in optimisation variable "
                           "selection")

    out = SaveFile(variables=list(variables), ispars=ispars, jspars=jspars)
    while pos < len(tok):
        i1, j1 = take(2, int)
        verts = [take(nvar) for _ in range(nvar + 1)]
        out.simplexes[(i1, j1)] = verts

    if ni is not None and out.ni != ni:
        raise SaveMismatch(f"save file i number mismatch: {ni} in the run, "
                           f"{out.ni} in the file")
    if nj is not None and out.nj != nj:
        raise SaveMismatch(f"save file j number mismatch: {nj} in the run, "
                           f"{out.nj} in the file")
    return out
