"""Reading TASOPT's ``.tas`` input files -- ``getparm.f`` and ``getval.f``.

A ``.tas`` file is a **positional** list of values, one per line, with
everything after ``!`` a comment and lines starting with ``#``, ``%`` or ``!``
skipped entirely. There are no keywords for most of it: the reader and the
file agree on the order of about three hundred numbers, and a line inserted or
deleted anywhere shifts everything after it. The handful of keyword-driven
sections -- the ``i``/``j`` parameter sweeps and the optimisation variables --
are the exception.

This module reproduces that reader, and the array initialisation ``tasopt.f``
does around it, so :func:`read_tas` produces exactly the state the shipped
program hands to ``wsize``.

Two conventions from the format worth knowing
---------------------------------------------
**Values carry their own unit conversion.** ``77.0 * 0.0254`` is how the file
says "77 inches". The reader finds the first ``*`` or ``/`` on the line and
applies whatever follows it as a factor, so ``124.0 * 0.3048`` and
``3000.0 / 1.0`` are both single values. Only one operator is honoured, and
``*`` wins if both are present.

**Unset means huge, not zero.** ``tasopt.f`` fills every parameter array with
``2**1023`` before reading, so anything the file does not set and no routine
computes stays at that value rather than silently reading as zero. This module
does the same. It is why so much of ``para``/``pare`` in a dumped state is
``8.98846567431158e+307``.

Things in the source worth knowing
----------------------------------
* ``getrval`` and friends read into a ``character*128`` buffer, so a line
  longer than 128 characters is silently truncated. No shipped case comes
  close -- ``737.tas``'s longest line is 82 -- but the truncation is
  reproduced.
* ``parg(igsweeph)`` is read from the file and then immediately overwritten
  with ``parg(igsweep)``, under a ``###`` comment. The tail sweep in a
  ``.tas`` file is dead input.
* ``parg(ignfweb)`` is hard-wired to 1.0; its read is commented out.
* The number of airfoil databases is hard-wired to 1 and then to 2, with the
  second set to the same filename as the first, so the same file is read
  twice into two identical tables. This port reads it once.
* The three mission-varying excrescence factors are inside ``if(.false.)``.
* ``getrkey`` searches for its keyword *anywhere* in the line, although its
  own comment says "at the beginning".

Verified against the compiled Fortran; see ``tests/test_tasfile.py``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .atmosphere import atmos
from .model import Aircraft
from .model import indices as I

__all__ = ["read_tas", "apply_sweep", "TasCase", "RunSettings",
           "TasFormatError", "BIGNUM", "LINE_WIDTH", "NMISX",
           "read_tase", "EngineGrid"]

#: ``tasopt.f``'s fill value for "not set". Every parameter array starts here.
BIGNUM = 2.0 ** 1023
#: The ``character*128`` buffer every read goes through.
LINE_WIDTH = 128
#: ``arrs.inc``'s ``nmisx`` -- the most missions a case may carry. It is not
#: in ``index.inc``, so ``tools/gen_indices.py`` does not pick it up.
NMISX = 5
GEE = 9.81


class TasFormatError(Exception):
    """Raised where the Fortran prints a message and calls ``stop``."""


@dataclass
class RunSettings:
    """The execution-control block at the top of the file."""
    Litprint: bool = False
    iterwmax: int = 50
    iterfmax: int = 15
    wrlx1: float = 0.5      # hard-wired in getparm.f; its reads are commented
    wrlx2: float = 0.9
    wrlx3: float = 0.5
    Lopt: bool = False
    Loprint: bool = False
    istepmax: int = 0
    istepout: int = 0
    Wftol: float = 0.0
    LlBFcon: bool = False
    LWfmaxcon: bool = False
    Lbmaxcon: bool = False
    Lgtoccon: bool = False
    Loutwrite: bool = False
    Lfblwrite: bool = False
    Ltrpwrite: bool = False
    Lsavwrite: bool = False
    Laswwrite: bool = False


@dataclass
class TasCase:
    """One ``.tas`` file, read.

    ``missions`` holds one :class:`~tasopt_py.model.Aircraft` per mission, all
    sharing the same ``pari`` and ``parg`` objects -- which is how TASOPT
    stores them, ``parm``/``para``/``pare`` being per-mission and the flags and
    geometry global. ``missions[0]`` is the design mission, the one ``wsize``
    sizes for.
    """
    configname: str = ""
    casename: tuple = ()
    settings: RunSettings = field(default_factory=RunSettings)
    airfoil_file: str = ""
    #: The name exactly as the file gives it, which is what the
    #: report echoes back.
    airfoil_name: str = ""
    nmission: int = 1
    pari: object = None
    parg: object = None
    missions: list = field(default_factory=list)
    #: Optimisation-variable perturbations, by ``cparo`` keyword. Zero means
    #: "do not optimise this one".
    dvarso: dict = field(default_factory=dict)
    #: Which parameter, if any, the i and j sweeps run over (empty if none).
    ispars: str = ""
    jspars: str = ""
    parsi: list = field(default_factory=list)
    parsj: list = field(default_factory=list)

    @property
    def design(self):
        """``(pari, parg, parm, para, pare)`` for the design mission."""
        m = self.missions[0]
        return self.pari, self.parg, m.parm, m.para, m.pare


# --------------------------------------------------------------------------
# getval.f
# --------------------------------------------------------------------------

def _fortran_float(tok: str) -> float:
    """Fortran list-directed real. ``1.0d0`` and ``1.0D+3`` are legal."""
    return float(re.sub(r"[dD]", "e", tok.strip()))


def _split(s: str):
    """Whitespace/comma separated fields, as ``getflt`` counts them."""
    if re.search(r",\s*,", s):
        # A null field leaves the target array element unchanged in Fortran
        # list-directed input. Nothing shipped does it, and silently guessing
        # would be worse than refusing.
        raise TasFormatError(f"null list field is not supported: {s!r}")
    return [t for t in re.split(r"[\s,]+", s.strip()) if t]


class _Reader:
    """The ``lu``/``iline`` pair the ``get*val`` routines thread through."""

    def __init__(self, text: str):
        self._lines = text.splitlines()
        self._i = 0          # 0-based cursor into _lines
        self.iline = 0       # the Fortran line counter, for messages

    # -- the shared skip-and-decomment logic of every get*val routine ------
    def _raw(self):
        while True:
            if self._i >= len(self._lines):
                raise TasFormatError(
                    f"unexpected end-of-file after line {self.iline}")
            line = self._lines[self._i][:LINE_WIDTH]
            self._i += 1
            self.iline += 1
            if not line.strip():
                continue
            if line[:1] in "#%!":
                continue
            kbang = line.find("!")
            body = line if kbang < 0 else line[:kbang]
            if not body.strip():
                continue
            return body

    def back(self):
        """``backspace(lu)``, with the line counter wound back too."""
        self._i -= 1
        self.iline -= 1

    def logical(self) -> bool:
        tok = _split(self._raw())[0].lstrip(".").lower()
        if tok.startswith("t"):
            return True
        if tok.startswith("f"):
            return False
        raise TasFormatError(f"line {self.iline}: not a logical: {tok!r}")

    def integer(self) -> int:
        return int(_split(self._raw())[0])

    def real(self) -> float:
        """``getrval``: one value, with an optional ``* fac`` or ``/ fac``."""
        body = self._raw()
        kmul, kdiv = body.find("*"), body.find("/")
        if kmul >= 0:
            fac = _fortran_float(_split(body[kmul + 1:])[0])
            body = body[:kmul]
        elif kdiv >= 0:
            fac = 1.0 / _fortran_float(_split(body[kdiv + 1:])[0])
            body = body[:kdiv]
        else:
            fac = 1.0
        return _fortran_float(_split(body)[0]) * fac

    def text(self) -> str:
        return self._raw()


def _rkey(line: str, key: str, nval: int):
    """``getrkey``. Returns ``(values, status)``.

    ``status`` is the Fortran's returned ``nval``: -2 keyword absent, 0
    keyword present with no values, >0 that many values read.
    """
    kend = len(line)
    for ch in "#%!":
        k = line.find(ch)
        if k >= 0:
            kend = min(k, kend)
    if kend <= 0:
        return [], -1
    body = line[:kend]

    starts_with_number = body[:1] in "0123456789-+."
    if starts_with_number or not key.strip():
        data = body
    else:
        # The keyword is matched with its trailing blank, anywhere in the
        # line -- not only at the start, whatever the source comment says.
        k = body.find(key + " ")
        if k < 0:
            return [], -2
        data = body[k + len(key) + 1:]

    if not data:
        return [], 0

    kmul, kdiv = data.find("*"), data.find("/")
    if kmul >= 0:
        fac = _fortran_float(_split(data[kmul + 1:])[0])
        data = data[:kmul]
    elif kdiv >= 0:
        fac = 1.0 / _fortran_float(_split(data[kdiv + 1:])[0])
        data = data[:kdiv]
    else:
        fac = 1.0

    toks = _split(data)
    if not toks:
        return [], 0
    if nval > 0:
        toks = toks[:nval]
    vals = [_fortran_float(t) * fac for t in toks]
    return vals, len(vals)


# --------------------------------------------------------------------------
# getparm.f
# --------------------------------------------------------------------------

def read_tas(path) -> TasCase:
    """Read a ``.tas`` file into the state ``wsize`` expects.

    The airfoil filename inside the file is relative to the file's own
    directory, as it is when TASOPT is run from the case directory; the
    resolved path is returned in ``airfoil_file`` but the database is not
    loaded here -- pass it to :func:`tasopt_py.aero.airfoil.airtable`.
    """
    path = Path(path)
    r = _Reader(path.read_text())

    case = TasCase()
    case.configname = r.text().strip()
    case.casename = (r.text().strip(), r.text().strip())

    s = case.settings
    s.Litprint = r.logical()
    s.iterwmax = r.integer()
    s.iterfmax = r.integer()
    # wrlx1/2/3 are hard-wired here; their three reads are commented out.
    s.wrlx1, s.wrlx2, s.wrlx3 = 0.5, 0.9, 0.5
    s.Lopt = r.logical()
    s.Loprint = r.logical()
    s.istepmax = r.integer()
    s.istepout = r.integer()
    s.Wftol = r.real()
    s.LlBFcon = r.logical()
    s.LWfmaxcon = r.logical()
    s.Lbmaxcon = r.logical()
    s.Lgtoccon = r.logical()
    s.Loutwrite = r.logical()
    s.Lfblwrite = r.logical()
    s.Ltrpwrite = r.logical()
    s.Lsavwrite = r.logical()
    s.Laswwrite = r.logical()

    # ---- i and j parameter sweeps ---------------------------------------
    for which in ("i", "j"):
        line = r.text()
        for name in I.CPARS:
            vals, n = _rkey(line, name, 999)
            if n == -1:
                raise TasFormatError(f"read error on line {r.iline}: {line!r}")
            if n >= 0:
                if which == "i":
                    case.ispars, case.parsi = (name if n else ""), vals
                else:
                    case.jspars, case.parsj = (name if n else ""), vals
                break
        else:
            raise TasFormatError(
                f"unrecognised {which}-parameter keyword on line {r.iline}: "
                f"{line!r}. Valid keywords are {I.CPARS}")
        r.text()                      # MATLAB plot label
        if which == "i":
            r.text()                  # line colours, i sweep only

    # ---- optimisation variables ------------------------------------------
    case.dvarso = {k: 0.0 for k in I.CPARO}
    while True:
        line = r.text()
        stripped = line.lstrip()
        if not stripped[:1].isalpha():
            # Not a keyword line -- put it back and move on.
            r.back()
            break
        for name in I.CPARO:
            vals, n = _rkey(line, name, 2)
            if n == -1:
                raise TasFormatError(f"read error on line {r.iline}: {line!r}")
            if n > 0:
                case.dvarso[name] = vals[0]
                break
            if n == 0:
                break               # valid variable, but not to be optimised
        else:
            raise TasFormatError(
                f"unrecognised optimised-variable keyword on line {r.iline}: "
                f"{line!r}. Valid keywords are {I.CPARO}")

    # ---- missions ---------------------------------------------------------
    def _row(nmax):
        """One line of per-mission values, padded with the last one given."""
        vals, n = _rkey(r.text(), " ", nmax)
        if n < 0:
            raise TasFormatError(f"read error on line {r.iline}")
        return vals

    wopt = _row(NMISX + 20)
    nmission = min(len(wopt), NMISX)
    case.nmission = nmission

    pari = Aircraft().pari
    parg = Aircraft().parg
    for k in range(1, I.IGTOTAL + 1):
        parg[k] = BIGNUM
    case.pari, case.parg = pari, parg
    case.missions = [Aircraft(pari=pari, parg=parg) for _ in range(nmission)]
    for m in case.missions:
        for k in range(1, I.IMTOTAL + 1):
            m.parm[k] = BIGNUM
        for ip in range(1, I.IPTOTAL + 1):
            for k in range(1, I.IATOTAL + 1):
                m.para[k, ip] = BIGNUM
            for k in range(1, I.IETOTAL + 1):
                m.pare[k, ip] = BIGNUM

    def _pad(vals):
        return [vals[min(k, len(vals) - 1)] for k in range(nmission)]

    wsum = sum(wopt[:nmission])
    if wsum == 0.0:
        wopt = [1.0] + wopt[1:]
        wsum = 1.0
    for km, m in enumerate(case.missions):
        m.parm[I.IMWOPT] = wopt[km] / wsum

    for idx, vals in ((I.IMRANGE, _pad(_row(NMISX))),):
        for km, m in enumerate(case.missions):
            m.parm[idx] = vals[km]

    Npax = _pad(_row(NMISX))
    Wpax = _pad(_row(NMISX))
    for km, m in enumerate(case.missions):
        m.parm[I.IMWPAY] = Npax[km] * Wpax[km]

    for idx in (I.IMALTTO, I.IMT0TO):
        vals = _pad(_row(NMISX))
        for km, m in enumerate(case.missions):
            m.parm[idx] = vals[km]

    # The three mission-varying excrescence factors sit inside `if(.false.)`.

    altCR = r.real()
    clpmax = r.real()
    for m in case.missions:
        altTO = m.parm[I.IMALTTO]
        for ip in range(I.IPSTATIC, I.IPCUTBACK + 1):
            m.para[I.IAALT, ip] = altTO
            m.para[I.IACLPMAX, ip] = clpmax
        m.para[I.IAALT, I.IPCLIMB1] = altTO
        m.para[I.IACLPMAX, I.IPCLIMB1] = clpmax
        m.para[I.IAALT, I.IPCRUISE1] = altCR
        m.para[I.IAALT, I.IPDESCENTN] = altTO
        m.para[I.IACLPMAX, I.IPDESCENTN] = clpmax

    for idx in (I.IGCDEFAN, I.IGCDGEAR, I.IGCDSPOIL, I.IGMUROLL,
                I.IGMUBRAKE, I.IGHOBST, I.IGLBFMAX, I.IGGTOCMIN,
                I.IGDBSLMAX, I.IGDBCBMAX):
        parg[idx] = r.real()

    tCBdeg, gCBdeg, gDE1deg, gDEndeg = (r.real() for _ in range(4))
    for m in case.missions:
        m.parm[I.IMTHCB] = tCBdeg * math.pi / 180.0
        m.parm[I.IMGAMVCB] = gCBdeg * math.pi / 180.0
        m.parm[I.IMGAMVDE1] = gDE1deg * math.pi / 180.0
        m.parm[I.IMGAMVDEN] = gDEndeg * math.pi / 180.0

    for idx in (I.IGNLIFT, I.IGNLAND, I.IGVNE):
        parg[idx] = r.real()
    parg[I.IGPCABIN] = atmos(r.real() / 1000.0).p

    CL, Mach = r.real(), r.real()
    for m in case.missions:
        for ip in range(I.IPCLIMB1 + 1, I.IPDESCENTN):
            m.para[I.IACL, ip] = CL
        for ip in range(I.IPCLIMBN, I.IPDESCENT1 + 1):
            m.para[I.IAMACH, ip] = Mach

    for idx in (I.IGSWEEP, I.IGAR, I.IGBMAX, I.IGLAMBDAS, I.IGLAMBDAT):
        parg[idx] = r.real()
    parg[I.IGLAMBDAS] = max(parg[I.IGLAMBDAS], 0.001)
    parg[I.IGLAMBDAT] = max(parg[I.IGLAMBDAT], 0.001)

    pari[I.IIWPLAN] = r.integer()
    pari[I.IIFWCEN] = r.integer()

    for idx in (I.IGRWFMAX, I.IGFLO, I.IGFLT, I.IGZS):
        parg[idx] = r.real()
    parg[I.IGBO] = 2.0 * r.real()          # file gives the half-span yo
    parg[I.IGETAS] = r.real()
    parg[I.IGRVSTRUT] = r.real()
    parg[I.IGCLHNRAT] = r.real()

    # Spanload parameters, in three blocks: takeoff, cruise, landing.
    for lo, hi in ((1, I.IPCLIMB1),
                   (I.IPCLIMB1 + 1, I.IPDESCENTN - 1),
                   (I.IPDESCENTN, I.IPDESCENTN)):
        vals = [r.real() for _ in range(5)]
        for m in case.missions:
            for ip in range(lo, hi + 1):
                for idx, v in zip((I.IARCLS, I.IARCLT, I.IACMPO, I.IACMPS,
                                   I.IACMPT), vals):
                    m.para[idx, ip] = v

    for idx in (I.IGWBOX, I.IGHBOXO, I.IGHBOXS, I.IGRH, I.IGXAXIS,
                I.IGHSTRUT,
                I.IGFFLAP, I.IGFSLAT, I.IGFAILE, I.IGFLETE, I.IGFRIBS,
                I.IGFSPOI, I.IGFWATT):
        parg[idx] = r.real()

    pari[I.IIHTSIZE] = r.integer()
    parg[I.IGVH] = r.real()
    parg[I.IGCLHCGFWD] = r.real()
    pari[I.IIVTSIZE] = r.integer()
    parg[I.IGVV] = r.real()
    parg[I.IGCLVEOUT] = r.real()
    pari[I.IIXWMOVE] = r.integer()
    for idx in (I.IGCLHSPEC, I.IGSMMIN, I.IGDEPSDA, I.IGDCLNDA,
                I.IGARH, I.IGARV, I.IGLAMBDAH, I.IGLAMBDAV, I.IGSWEEPH):
        parg[idx] = r.real()
    # The tail sweep just read is thrown away -- marked ### in the source.
    parg[I.IGSWEEPH] = parg[I.IGSWEEP]
    parg[I.IGSWEEPV] = r.real()
    parg[I.IGBOH] = 2.0 * r.real()
    parg[I.IGBOV] = 2.0 * r.real()
    for idx in (I.IGFCDHCEN, I.IGCLHMAX, I.IGCLVMAX, I.IGFHADD, I.IGFVADD,
                I.IGWBOXH, I.IGWBOXV, I.IGHBOXH, I.IGHBOXV, I.IGRHH,
                I.IGRHV, I.IGNVTAIL,
                I.IGRFUSE, I.IGDRFUSE, I.IGWFB):
        parg[idx] = r.real()
    parg[I.IGNFWEB] = 1.0                  # its read is commented out
    for idx in (I.IGHFLOOR, I.IGANOSE, I.IGBTAIL,
                I.IGXNOSE, I.IGXEND, I.IGXBLEND1, I.IGXBLEND2,
                I.IGXSHELL1, I.IGXSHELL2, I.IGXCONEND,
                I.IGXWBOX, I.IGXHBOX, I.IGXVBOX,
                I.IGZWING, I.IGZHTAIL):
        parg[idx] = r.real()
    pari[I.IIENGLOC] = r.integer()
    for idx in (I.IGXENG, I.IGYENG, I.IGNENG, I.IGLAMBDAC,
                I.IGFSTRING, I.IGFFRAME, I.IGFFADD, I.IGWFIX, I.IGXFIX,
                I.IGWPWINDOW, I.IGWPPINSUL, I.IGWPPFLOOR,
                I.IGRMH, I.IGRMV):
        parg[idx] = r.real()
    pari[I.IIFCLOSE] = r.integer()
    parg[I.IGCMVF1] = r.real()
    parg[I.IGCLMF0] = r.real()

    fduo, fdus, fdut = (r.real() for _ in range(3))
    for m in case.missions:
        for ip in range(1, I.IPTOTAL + 1):
            m.para[I.IAFDUO, ip] = fduo
            m.para[I.IAFDUS, ip] = fdus
            m.para[I.IAFDUT, ip] = fdut

    for idx in (I.IGXHPESYS, I.IGXLGNOSE, I.IGDXLGMAIN,
                I.IGFHPESYS, I.IGFLGNOSE, I.IGFLGMAIN,
                I.IGXAPU, I.IGFAPU, I.IGFSEAT, I.IGFPADD,
                I.IGFEADD, I.IGFPYLON, I.IGFRESERVE,
                I.IGSIGFAC, I.IGSIGSKIN, I.IGSIGBEND, I.IGSIGCAP,
                I.IGTAUWEB, I.IGSIGSTRUT, I.IGRESHELL,
                I.IGECAP, I.IGESTRUT,
                I.IGRHOSKIN, I.IGRHOBEND, I.IGRHOCAP, I.IGRHOWEB,
                I.IGRHOSTRUT):
        parg[idx] = r.real()

    # One airfoil database, read twice into two identical tables upstream.
    case.airfoil_name = r.text().strip()
    case.airfoil_file = str((path.parent / case.airfoil_name).resolve())

    drag = [r.real() for _ in range(13)]
    (cdfw, cdpw, Rerefw, cdft, cdpt, Rereft, cdfs, cdps, Rerefs,
     aRexp, fexcdw, fexcdt, fexcdf) = drag
    parg[I.IGFBLIW] = r.real()
    parg[I.IGFBLIF] = r.real()
    pari[I.IIBLIC] = r.integer()
    for m in case.missions:
        for ip in range(1, I.IPTOTAL + 1):
            for idx, v in ((I.IACDFW, cdfw), (I.IACDPW, cdpw),
                           (I.IAREREFW, Rerefw), (I.IACDFT, cdft),
                           (I.IACDPT, cdpt), (I.IAREREFT, Rereft),
                           (I.IACDFS, cdfs), (I.IACDPS, cdps),
                           (I.IAREREFS, Rerefs), (I.IAAREXP, aRexp),
                           (I.IAFEXCDW, fexcdw), (I.IAFEXCDT, fexcdt),
                           (I.IAFEXCDF, fexcdf)):
                m.para[idx, ip] = v

    pari[I.IIFUEL] = r.integer()
    parg[I.IGRHOFUEL] = r.real()
    Tfuel = r.real()
    for m in case.missions:
        for ip in range(1, I.IPTOTAL + 1):
            m.pare[I.IETFUEL, ip] = Tfuel

    Tmetal, Tt4TO = r.real(), r.real()
    parg[I.IGFTT4CL1] = r.real()
    parg[I.IGFTT4CLN] = r.real()
    Tt4CR = r.real()
    dTstrk, Mtexit, StA, efilm, tfilm, M4a, ruc = (r.real() for _ in range(7))
    parg[I.IGTMETAL] = Tmetal
    for m in case.missions:
        T0TO = m.parm[I.IMT0TO]
        for ip in range(1, I.IPTOTAL + 1):
            for idx, v in ((I.IETT4, Tt4CR), (I.IEM4A, M4a), (I.IERUC, ruc),
                           (I.IEDTSTRK, dTstrk), (I.IEMTEXIT, Mtexit),
                           (I.IESTA, StA), (I.IEEFILM, efilm),
                           (I.IETFILM, tfilm)):
                m.pare[idx, ip] = v
        for ip in (I.IPSTATIC, I.IPROTATE, I.IPTAKEOFF):
            m.pare[I.IET0, ip] = T0TO
            m.pare[I.IETT4, ip] = Tt4TO

    OPR, pilc = r.real(), r.real()
    pihc = OPR / pilc
    (pif, pid, pib, pifn, pitn, epolf, epollc, epolhc, epolht,
     epollt) = (r.real() for _ in range(10))
    etab = r.real()
    pifK, epfK = r.real(), r.real()
    BPR, Gearf, HTRf, HTRlc, HTRhc = (r.real() for _ in range(5))
    M2, M25 = r.real(), r.real()
    epsl, epsh = r.real(), r.real()
    engine = ((I.IEPID, pid), (I.IEPIB, pib), (I.IEPIFN, pifn),
              (I.IEPITN, pitn), (I.IEPIF, pif), (I.IEPILC, pilc),
              (I.IEPIHC, pihc), (I.IEEPOLF, epolf), (I.IEEPOLLC, epollc),
              (I.IEEPOLHC, epolhc), (I.IEEPOLHT, epolht),
              (I.IEEPOLLT, epollt), (I.IEETAB, etab), (I.IEPIFK, pifK),
              (I.IEEPFK, epfK), (I.IEBPR, BPR), (I.IEM2, M2),
              (I.IEM25, M25), (I.IEEPSL, epsl), (I.IEEPSH, epsh))
    for m in case.missions:
        for ip in range(1, I.IPTOTAL + 1):
            for idx, v in engine:
                m.pare[idx, ip] = v

    mofftpax, mofftmMTO, Pofftpax, PofftmMTO = (r.real() for _ in range(4))
    parg[I.IGMOFWPAY] = mofftpax / Wpax[0]
    parg[I.IGMOFWMTO] = mofftmMTO / GEE
    parg[I.IGPOFWPAY] = Pofftpax / Wpax[0]
    parg[I.IGPOFWMTO] = PofftmMTO / GEE

    Tt9, pt9 = r.real(), r.real()
    for m in case.missions:
        for ip in range(1, I.IPTOTAL + 1):
            m.pare[I.IETT9, ip] = Tt9
            m.pare[I.IEPT9, ip] = pt9

    A7 = [r.real() for _ in range(7)]      # static TO cutback cl1 cln de1 den
    A5 = [r.real() for _ in range(7)]
    for m in case.missions:
        for Afac, idx in ((A7, I.IEA7FAC), (A5, I.IEA5FAC)):
            st, to, cb, c1, cn, d1, dn = Afac
            m.pare[idx, I.IPSTATIC] = st
            m.pare[idx, I.IPROTATE] = to
            m.pare[idx, I.IPTAKEOFF] = to
            m.pare[idx, I.IPCUTBACK] = cb
            for ip in range(I.IPCLIMB1, I.IPCLIMBN + 1):
                f = float(ip - I.IPCLIMB1) / float(I.IPCLIMBN - I.IPCLIMB1)
                m.pare[idx, ip] = c1 * (1.0 - f) + cn * f
            for ip in range(I.IPCRUISE1, I.IPCRUISEN + 1):
                m.pare[idx, ip] = 1.0
            for ip in range(I.IPDESCENT1, I.IPDESCENTN + 1):
                f = (float(ip - I.IPDESCENT1)
                     / float(I.IPDESCENTN - I.IPDESCENT1))
                m.pare[idx, ip] = d1 * (1.0 - f) + dn * f
            m.pare[idx, I.IPTEST] = st

    parg[I.IGGEARF] = Gearf
    parg[I.IGHTRF] = HTRf
    parg[I.IGHTRLC] = HTRlc
    parg[I.IGHTRHC] = HTRhc

    parg[I.IGRSNACE] = r.real()
    parg[I.IGRVNACE] = r.real()
    pari[I.IIENGWGT] = r.integer()

    # tasopt.f, after getparm: the first mission sizes the airframe.
    parg[I.IGRANGE] = case.missions[0].parm[I.IMRANGE]
    parg[I.IGWPAY] = case.missions[0].parm[I.IMWPAY]

    return case


def apply_sweep(case, name: str, value: float) -> None:
    """Set one swept parameter, as ``tasopt.f`` does per ``i``/``j`` point.

    A ``.tas`` file may put a list of values after an ``i`` or ``j`` keyword,
    and the program then sizes an aircraft at every combination. This applies
    one such value. The keywords are ``cpars``; several of them reach more
    than one place, and ``OPR`` reaches an odd one -- see below.
    """
    parg, missions = case.parg, case.missions
    if name == "Range":
        # Only the *design* mission's range moves; the loop that would have
        # changed the others is commented out.
        parg[I.IGRANGE] = value
        missions[0].parm[I.IMRANGE] = value
    elif name == "Mach":
        for m in missions:
            for ip in range(I.IPCLIMBN, I.IPDESCENT1 + 1):
                m.para[I.IAMACH, ip] = value
    elif name == "Nmax":
        parg[I.IGNLIFT] = value
    elif name == "sigfac":
        parg[I.IGSIGFAC] = value
    elif name == "CL":
        for m in missions:
            for ip in range(I.IPCRUISE1, I.IPCRUISEN + 1):
                m.para[I.IACL, ip] = value
    elif name == "AR":
        parg[I.IGAR] = value
    elif name == "sweep":
        # Only the wing; the line that would also set the tail sweep is
        # commented out -- and getparm overwrites the tail sweep anyway.
        parg[I.IGSWEEP] = value
    elif name == "etas":
        parg[I.IGETAS] = value
    elif name == "Tt4CR":
        for m in missions:
            for ip in range(I.IPCRUISE1, I.IPCRUISEN + 1):
                m.pare[I.IETT4, ip] = value
    elif name == "Tt4TO":
        for m in missions:
            for ip in (I.IPSTATIC, I.IPROTATE, I.IPTAKEOFF):
                m.pare[I.IETT4, ip] = value
    elif name == "Tmetal":
        parg[I.IGTMETAL] = value
    elif name == "OPR":
        # The HPC pressure ratio is worked out *once*, from the LPC ratio at
        # the start of cruise of the first mission, and then written to every
        # point of every mission. The per-point form is commented out beside
        # it, so a point whose LPC ratio differs does not get the OPR asked
        # for.
        pihc = value / missions[0].pare[I.IEPILC, I.IPCRUISE1]
        for m in missions:
            for ip in range(1, I.IPTOTAL + 1):
                m.pare[I.IEPIHC, ip] = pihc
    elif name == "FPR":
        for m in missions:
            for ip in range(1, I.IPTOTAL + 1):
                m.pare[I.IEPIF, ip] = value
    elif name == "lBFmax":
        parg[I.IGLBFMAX] = value
    elif name == "bmax":
        parg[I.IGBMAX] = value
    elif name == "alt":
        for m in missions:
            m.para[I.IAALT, I.IPCRUISE1] = value
    else:
        raise TasFormatError(f"cannot sweep over {name!r}; "
                             f"valid keywords are {I.CPARS}")


# --------------------------------------------------------------------------
# The engine operating-point grid -- tasopt.f's optional `.tase` file
# --------------------------------------------------------------------------

#: ``tasopt.f``'s array bounds on the three grid axes.
NEADIM = NEMDIM = NEFDIM = 21


@dataclass
class EngineGrid:
    """The altitude x Mach x throttle grid an ``.oute`` engine deck is run on.

    Read from a separate ``<case>.tase`` file, which ``tasopt.f`` opens next
    to the ``.tas`` and quietly ignores if it is not there. Three lines, one
    per axis, each read with ``getrkey`` -- so each takes the ``* factor``
    and ``/ factor`` suffixes, which is how the 737's altitudes are written
    in feet and converted to metres on the spot.
    """
    #: Altitudes, metres.
    alt: list = field(default_factory=list)
    mach: list = field(default_factory=list)
    #: Throttle settings, as a *fraction of the maximum thrust available at
    #: that altitude and Mach* -- not of sea-level static thrust.
    fset: list = field(default_factory=list)

    def __bool__(self):
        """``Lengoper``: all three axes must be non-empty."""
        return bool(self.alt) and bool(self.mach) and bool(self.fset)

    @property
    def shape(self):
        return len(self.alt), len(self.mach), len(self.fset)

    @property
    def npoints(self):
        n, m, f = self.shape
        return n * m * f


def read_tase(path) -> EngineGrid:
    """Read a ``.tase`` engine operating-point grid.

    A missing file gives an empty grid rather than an error, which is what
    ``tasopt.f`` does with its ``err=15`` branch -- the deck is optional.
    """
    path = Path(path)
    if not path.exists():
        return EngineGrid()

    r = _Reader(path.read_text())
    axes = []
    for nmax in (NEADIM, NEMDIM, NEFDIM):
        vals, n = _rkey(r.text(), " ", nmax)
        axes.append(vals if n > 0 else [])
    return EngineGrid(alt=axes[0], mach=axes[1], fset=axes[2])
