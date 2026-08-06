#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""What solvers this installation actually has, and which one will be used.

Three questions this module answers, none of which the solve path answers on
its own:

1. Which ``ipopt`` binary wins? More than one can be on ``PATH`` --- a conda
   environment puts its own in ``$CONDA_PREFIX/bin``, which ``conda activate``
   prepends, so a source build of IPOPT that a user made specifically to get
   MA27 can be shadowed by the prebuilt MUMPS one without any error at all.
   The symptom is slower and flakier solves, which reads as "lcsolver is
   flaky".

2. Which linear solver does that binary carry? This is fixed at IPOPT build
   time and is the single largest determinant of behaviour on a geometric or
   signomial program (see ``docs/ipopt.rst``). It cannot be read out of a
   version string; it has to be probed by solving something.

3. Is cyipopt --- the in-process route, and the only one that can evaluate a
   black-box constraint --- present, and is it linked against the *same* IPOPT
   as the executable? A conda cyipopt and a source-built MA27 executable are a
   perfectly common and perfectly silent mismatch.

``LCSOLVER_IPOPT_EXECUTABLE``
    Set this to an absolute path to pin the executable regardless of ``PATH``
    order. It exists because telling a user to reorder ``PATH`` against
    ``conda activate`` is advice that works until the next shell.
"""

import contextlib
import io
import logging
import os
import shutil
import sys

IPOPT_EXECUTABLE_ENV = 'LCSOLVER_IPOPT_EXECUTABLE'
AUTOSELECT_ENV = 'LCSOLVER_IPOPT_AUTOSELECT'

#: Where the installer records an IPOPT it built. A source build lands
#: somewhere like ``~/software/ipopt/build/bin``, which is on nobody's PATH,
#: and the alternative to remembering it is telling every user to paste an
#: export into a shell profile -- a step that is easy to skip and silently
#: leaves the build unused.
STATE_FILE = os.path.join(os.path.expanduser('~'), '.config', 'lcsolver',
                          'solvers.json')

# Probing costs a solve, so cache per (executable, linear_solver) for the
# session. Nothing here changes while a process is running.
_PROBE_CACHE = {}

# Resolution is cached too: it can involve probing, and it is consulted once
# per solve -- including inside the SLCP and SIA loops, which solve thousands
# of subproblems.
_RESOLVED = None


# --------------------------------------------------------------------------
# locating the executable
# --------------------------------------------------------------------------
def recorded_ipopt():
    """The IPOPT the installer last built, if any."""
    try:
        import json
        with open(STATE_FILE) as handle:
            path = json.load(handle).get('ipopt_executable')
    except Exception:
        return None
    return path if path and os.path.isfile(path) else None


def record_ipopt(path):
    """Remember a built IPOPT, so no shell profile has to be edited."""
    import json

    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    state = {}
    try:
        with open(STATE_FILE) as handle:
            state = json.load(handle)
    except Exception:
        pass
    state['ipopt_executable'] = path
    with open(STATE_FILE, 'w') as handle:
        json.dump(state, handle, indent=2)
    return STATE_FILE


def ipopt_choice():
    """``(path, reason)`` for the ``ipopt`` LCsolver will use.

    The order is deliberate, and the middle of it is the part worth explaining.

    1. ``LCSOLVER_IPOPT_EXECUTABLE``, if set. Always wins, even if broken ---
       a pin that points at nothing is a mistake to report, not to route around.
    2. Otherwise, among every candidate (``PATH``, plus whatever the installer
       recorded), **an MA27 build beats one without it**, regardless of order.

    That second rule is the whole reason this function exists. ``conda
    activate`` prepends ``$CONDA_PREFIX/bin`` to ``PATH`` in every new shell,
    so a prebuilt MUMPS IPOPT shadows a source build made specifically to get
    MA27 --- silently, since both solve, one just worse. Honouring ``PATH``
    strictly means that machine stays misconfigured until somebody notices the
    performance and reads the docs. Preferring MA27 means it never happens.

    The probe only runs when there is more than one candidate, which on most
    machines is never. ``LCSOLVER_IPOPT_AUTOSELECT=0`` restores strict
    ``PATH`` order.
    """
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED

    pinned = os.environ.get(IPOPT_EXECUTABLE_ENV)
    if pinned:
        _RESOLVED = (pinned, f'pinned by {IPOPT_EXECUTABLE_ENV}')
        return _RESOLVED

    candidates = list(ipopt_executables_on_path())
    recorded = recorded_ipopt()
    if recorded and not any(os.path.realpath(c) == os.path.realpath(recorded)
                            for c in candidates):
        candidates.append(recorded)

    if not candidates:
        _RESOLVED = (None, 'no ipopt on PATH and none recorded')
    elif len(candidates) == 1:
        only = candidates[0]
        source = ('recorded by lcsolver-install-solvers'
                  if recorded and os.path.realpath(only)
                  == os.path.realpath(recorded) else 'found on PATH')
        _RESOLVED = (only, source)
    elif os.environ.get(AUTOSELECT_ENV) == '0':
        _RESOLVED = (candidates[0], f'first on PATH ({AUTOSELECT_ENV}=0)')
    else:
        best = next((c for c in candidates
                     if linear_solver_available('ma27', c)), None)
        if best is None:
            _RESOLVED = (candidates[0],
                         f'first of {len(candidates)}; none has MA27')
        elif os.path.realpath(best) == os.path.realpath(candidates[0]):
            _RESOLVED = (best, f'first of {len(candidates)}, and has MA27')
        else:
            _RESOLVED = (best, f'chosen over {candidates[0]}, which has no MA27')

    return _RESOLVED


def ipopt_executable():
    """Path to the ``ipopt`` binary LCsolver will use, or ``None``."""
    return ipopt_choice()[0]


def _forget_resolution():
    """Drop the cached choice. For tests, and for callers that change PATH."""
    global _RESOLVED
    _RESOLVED = None
    _PROBE_CACHE.clear()


def ipopt_available():
    """Is there any usable IPOPT here --- the executable, or cyipopt?

    Not cached: Pyomo's own availability check is a PATH lookup, and caching
    would make an IPOPT installed mid-session invisible.
    """
    try:
        from lcsolver.solvers.ipopt.NLP import _executable_available
        if _executable_available('ipopt'):
            return True
    except Exception:
        pass
    try:
        import pyomo.environ as pyo
        return bool(pyo.SolverFactory('cyipopt').available(exception_flag=False))
    except Exception:
        return False


def ipopt_solver_factory(executable=None):
    """``SolverFactory('ipopt')`` that honours the executable pin.

    Every route that drives the AMPL interface should go through this rather
    than constructing the factory itself, or ``LCSOLVER_IPOPT_EXECUTABLE`` ends
    up respected on some code paths and ignored on others --- which is worse
    than not having it, because the SLCP and SIA loops (the ones that run
    thousands of IPOPT solves and care most about the linear solver) are
    exactly the paths that would ignore it.
    """
    import pyomo.environ as pyo

    exe = executable or ipopt_executable()
    return (pyo.SolverFactory('ipopt', executable=exe) if exe
            else pyo.SolverFactory('ipopt'))


def ipopt_executables_on_path():
    """Every ``ipopt`` on ``PATH``, in the order the shell would find them.

    Used to detect shadowing: if entry 0 lacks MA27 and a later entry has it,
    the user built IPOPT for nothing.
    """
    found = []
    seen = set()
    for directory in os.environ.get('PATH', '').split(os.pathsep):
        if not directory:
            continue
        for name in ('ipopt', 'ipopt.exe'):
            candidate = os.path.join(directory, name)
            real = os.path.realpath(candidate)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK) \
                    and real not in seen:
                seen.add(real)
                found.append(candidate)
    return found


# --------------------------------------------------------------------------
# probing what a build contains
# --------------------------------------------------------------------------
@contextlib.contextmanager
def _quiet():
    """Swallow the noise a deliberately-failing probe makes.

    A probe for a linear solver the build does not have is *expected* to fail,
    and IPOPT is loud about it -- Pyomo logs "Solver returned non-zero return
    code" at ERROR and the option documentation gets echoed in full. Printing
    that while answering "does this build have MA27?" reads as a crash.

    Three layers, because the noise comes from three places: the Pyomo logger,
    Python-level stdout/stderr, and --- for the in-process route --- IPOPT's
    own C++ output, which writes to the file descriptors directly and ignores
    ``sys.stdout`` entirely. Only ``dup2`` reaches that last one.
    """
    logger = logging.getLogger('pyomo')
    previous = logger.level
    logger.setLevel(logging.CRITICAL)

    sink = io.StringIO()
    devnull = saved_out = saved_err = None
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved_out = os.dup(1)
        saved_err = os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
    except Exception:
        # No usable file descriptors (embedded interpreter, some notebook
        # kernels). The Python-level redirect below still applies; a chattier
        # report is better than a failed one.
        pass

    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield
    finally:
        logger.setLevel(previous)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        for fd, saved in ((1, saved_out), (2, saved_err)):
            if saved is not None:
                os.dup2(saved, fd)
                os.close(saved)
        if devnull is not None:
            os.close(devnull)


def linear_solver_available(name, executable=None):
    """Does this IPOPT build carry the ``name`` linear solver?

    Probed by solving a one-variable problem with ``linear_solver <name>``: a
    build without it rejects the option and fails. There is no reliable way to
    ask the binary directly --- ``--print-options`` does not list the compiled
    set on every version.
    """
    executable = executable or ipopt_executable()
    key = (os.path.realpath(executable) if executable else None, name)
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]

    result = False
    try:
        import pyomo.environ as pyo
        from pyomo.opt import TerminationCondition

        probe = pyo.ConcreteModel()
        probe.x = pyo.Var(initialize=1.0, bounds=(0.5, None))
        probe.o = pyo.Objective(expr=probe.x)
        opt = (pyo.SolverFactory('ipopt', executable=executable)
               if executable else pyo.SolverFactory('ipopt'))
        opt.options['linear_solver'] = name
        with _quiet():
            res = opt.solve(probe, tee=False, load_solutions=False)
        result = (res.solver.termination_condition
                  == TerminationCondition.optimal)
    except Exception:
        result = False

    _PROBE_CACHE[key] = result
    return result


# Run in a child interpreter, for a reason that is not obvious: refusing an
# option makes IPOPT's C++ journalist emit the whole option documentation, and
# it flushes that at *process exit*, long after any dup2 redirection has been
# undone. In-process there is no point at which it can be silenced. A child
# process has its own exit, and its output goes to a pipe.
#
# All three answers come from one child, so this costs one interpreter start
# rather than three.
_CYIPOPT_PROBE = r'''
import json, sys
out = {"baseline": False, "ma27": None, "mumps": None}
try:
    import pyomo.environ as pyo
    from pyomo.opt import TerminationCondition

    def attempt(options):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.0, bounds=(0.5, None))
        m.o = pyo.Objective(expr=m.x)
        opt = pyo.SolverFactory("cyipopt")
        if not opt.available(exception_flag=False):
            return None
        # PyomoCyIpoptSolver takes options here; it has no `options` mapping.
        r = opt.solve(m, tee=False, options=options)
        return r.solver.termination_condition == TerminationCondition.optimal

    out["baseline"] = attempt({}) is True
    if out["baseline"]:
        for name in ("ma27", "mumps"):
            try:
                out[name] = attempt({"linear_solver": name}) is True
            except Exception:
                out[name] = False
except Exception:
    pass
sys.__stdout__.write("\n@@LCSOLVER@@" + json.dumps(out) + "\n")
'''


def _cyipopt_probe():
    """Ask a child interpreter what cyipopt's IPOPT was built with."""
    key = ('cyipopt', '_all')
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]

    result = {'baseline': False, 'ma27': None, 'mumps': None}
    try:
        import subprocess

        proc = subprocess.run([sys.executable, '-c', _CYIPOPT_PROBE],
                              capture_output=True, text=True, timeout=300)
        for line in proc.stdout.splitlines():
            if line.startswith('@@LCSOLVER@@'):
                import json
                result = json.loads(line[len('@@LCSOLVER@@'):])
                break
    except Exception:
        pass

    _PROBE_CACHE[key] = result
    return result


def cyipopt_linear_solver_available(name):
    """Same question for the in-process route, which links its own IPOPT.

    Returns ``True``/``False``, or ``None`` when the probe could not run at
    all --- cyipopt absent, or present but unable to solve even a one-variable
    model. That is a different fact from "this build lacks MA27", and
    reporting it as ``False`` produces a confidently wrong diagnosis.
    """
    probe = _cyipopt_probe()
    if not probe.get('baseline'):
        return None
    return probe.get(name)


def _pynumero_asl_available():
    """Is the compiled PyNumero ASL library present and loadable?"""
    try:
        from pyomo.contrib.pynumero.asl import AmplInterface
        return bool(AmplInterface.available())
    except Exception:
        return False


def _ipopt_version(executable):
    try:
        import subprocess
        out = subprocess.run([executable, '--version'], capture_output=True,
                             text=True, timeout=30)
        first = (out.stdout or out.stderr).strip().splitlines()
        return first[0].strip() if first else None
    except Exception:
        return None


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------
def check_solvers(probe=True):
    """Collect the solver situation as a plain dict.

    Parameters
    ----------
    probe : bool
        Run the linear-solver probes. Each is a real (tiny) solve; turning
        them off makes the report fast and leaves the ``ma27``/``mumps`` fields
        ``None``.
    """
    report = {
        'cvxopt': {'available': False, 'version': None},
        'ipopt': {'executable': None, 'pinned': False, 'version': None,
                  'ma27': None, 'mumps': None, 'shadowed_by': None,
                  'all_on_path': [], 'reason': None},
        'cyipopt': {'available': False, 'version': None, 'pynumero_asl': None,
                    'ma27': None, 'mumps': None},
        'probed': bool(probe),
        'warnings': [],
    }

    # ---- cvxopt ----------------------------------------------------------
    try:
        import cvxopt
        report['cvxopt']['available'] = True
        report['cvxopt']['version'] = getattr(cvxopt, '__version__', 'unknown')
    except Exception:
        report['warnings'].append(
            'cvxopt is not importable. It is a declared dependency, so this '
            'install is incomplete: `pip install cvxopt`.')

    # ---- ipopt executable ------------------------------------------------
    exe, reason = ipopt_choice()
    on_path = ipopt_executables_on_path()
    report['ipopt']['all_on_path'] = on_path
    report['ipopt']['pinned'] = bool(os.environ.get(IPOPT_EXECUTABLE_ENV))
    report['ipopt']['executable'] = exe
    report['ipopt']['reason'] = reason

    if exe and report['ipopt']['pinned'] and not os.path.isfile(exe):
        report['warnings'].append(
            f'{IPOPT_EXECUTABLE_ENV} points at {exe}, which does not exist. '
            f'Unset it or correct it; nothing else will override it.')
    elif exe:
        report['ipopt']['version'] = _ipopt_version(exe)
        if probe:
            report['ipopt']['ma27'] = linear_solver_available('ma27', exe)
            report['ipopt']['mumps'] = linear_solver_available('mumps', exe)

            # An MA27 build losing to one without it. Resolution prefers MA27,
            # so reaching here means something overrode that: a pin, or
            # LCSOLVER_IPOPT_AUTOSELECT=0. Either is deliberate, and either
            # deserves saying out loud rather than silently costing robustness.
            if not report['ipopt']['ma27']:
                for other in list(on_path) + [recorded_ipopt()]:
                    if not other or (os.path.realpath(other)
                                     == os.path.realpath(exe)):
                        continue
                    if linear_solver_available('ma27', other):
                        report['ipopt']['shadowed_by'] = other
                        report['warnings'].append(
                            f'the ipopt being used ({exe}) has no MA27, but '
                            f'{other} does, and would normally be preferred. '
                            f'Reason this one was chosen: {reason}.')
                        break
            if not report['ipopt']['ma27'] and not report['ipopt']['shadowed_by']:
                report['warnings'].append(
                    'this IPOPT is a MUMPS build. It works; MA27 is markedly '
                    'more robust on geometric and signomial programs. Upgrade '
                    'with `lcsolver-install-solvers --ma27 <path-to-ma27>` '
                    '(see docs/ipopt.rst).')
    else:
        report['warnings'].append(
            'no ipopt executable found. IPOPT is the default convex backend, '
            'so this install cannot solve. Run `lcsolver-install-solvers`.')

    # ---- cyipopt ---------------------------------------------------------
    try:
        import cyipopt
        report['cyipopt']['available'] = True
        report['cyipopt']['version'] = getattr(cyipopt, '__version__', 'unknown')

        # cyipopt alone is not a working in-process route. Pyomo builds the
        # NLP through PyNumero, whose ASL shared library ships separately from
        # both packages; without it every black-box solve fails with "Cannot
        # load the PyNumero ASL interface", which names a component most users
        # have never heard of.
        report['cyipopt']['pynumero_asl'] = _pynumero_asl_available()
        if not report['cyipopt']['pynumero_asl']:
            report['warnings'].append(
                "cyipopt is installed but Pyomo's PyNumero ASL library is "
                'not, so black-box (grey-box) models cannot be solved at all. '
                'Fix with `pyomo download-extensions`, or '
                '`lcsolver-install-solvers`, which does it for you.')

        if probe:
            report['cyipopt']['ma27'] = cyipopt_linear_solver_available('ma27')
            report['cyipopt']['mumps'] = cyipopt_linear_solver_available('mumps')
    except Exception:
        report['warnings'].append(
            'cyipopt is not installed. Black-box (grey-box) constraints '
            'require it -- the AMPL executable route cannot evaluate a Python '
            'callback. Run `lcsolver-install-solvers`.')

    # A mismatch is not an error, but it is worth naming: the two routes will
    # behave differently on the same model and that is otherwise mystifying.
    if (probe and report['ipopt']['ma27'] is True
            and report['cyipopt']['available']
            and report['cyipopt']['ma27'] is False):
        report['warnings'].append(
            'the ipopt executable has MA27 but cyipopt does not, so black-box '
            'models (which must use cyipopt) run on MUMPS while everything '
            'else runs on MA27. Rebuild cyipopt against the MA27 build: '
            '`lcsolver-install-solvers --relink-cyipopt`.')

    return report


def _fmt(report):
    """Render the report for a terminal."""
    lines = []
    add = lines.append

    add('LCsolver solver environment')
    add('=' * 60)

    c = report['cvxopt']
    add(f"cvxopt        {'yes  ' + str(c['version']) if c['available'] else 'MISSING'}")

    i = report['ipopt']
    if i['executable']:
        add(f"ipopt         {i['executable']}")
        if i['reason']:
            # Never leave "why this binary?" to be inferred: with several
            # installed, the answer is the difference between a fast solve and
            # a flaky one.
            add(f"              {i['reason']}")
        if i['version']:
            add(f"              {i['version']}")
        if i['ma27'] is not None:
            solvers = [n for n, ok in (('ma27', i['ma27']), ('mumps', i['mumps'])) if ok]
            add(f"              linear solvers: {', '.join(solvers) or 'none detected'}")
        others = [p for p in i['all_on_path']
                  if os.path.realpath(p) != os.path.realpath(i['executable'])]
        if others:
            add(f"              also on PATH: {', '.join(others)}")
    else:
        add('ipopt         MISSING')

    y = report['cyipopt']
    if y['available']:
        add(f"cyipopt       yes  {y['version']}")
        if y.get('pynumero_asl') is False:
            add('              PyNumero ASL library: MISSING '
                '(black-box models cannot solve)')
        if report['probed']:
            if y['ma27'] is None and y['mumps'] is None:
                add('              linear solvers: could not probe '
                    '(cyipopt cannot solve a trivial model)')
            else:
                solvers = [n for n, ok in (('ma27', y['ma27']),
                                           ('mumps', y['mumps'])) if ok]
                add(f"              linear solvers: "
                    f"{', '.join(solvers) or 'none detected'}")
    else:
        add('cyipopt       MISSING  (needed for black-box constraints)')

    if report['warnings']:
        add('')
        for w in report['warnings']:
            add(f'! {w}')
    else:
        add('')
        add('No problems found.')

    return '\n'.join(lines)


#: Exit status for "the report ran fine and the install is incomplete". Kept
#: distinct from 1 so a caller can tell a missing solver from this command
#: falling over, which is otherwise indistinguishable -- an uncaught exception
#: also exits 1.
EXIT_INCOMPLETE = 3


def main(argv=None):
    """``lcsolver-check-solvers`` entry point.

    Exit status: 0 if the install can solve, ``EXIT_INCOMPLETE`` if something
    required is missing, 1 if this command itself failed.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog='lcsolver-check-solvers',
        description='Report which solvers this LCsolver install will use.')
    parser.add_argument(
        '--no-probe', action='store_true',
        help='skip the linear-solver probes (faster; omits ma27/mumps)')
    args = parser.parse_args(argv)

    report = check_solvers(probe=not args.no_probe)
    print(_fmt(report))
    incomplete = (report['ipopt']['executable'] is None
                  or not report['cvxopt']['available'])
    return EXIT_INCOMPLETE if incomplete else 0


if __name__ == '__main__':
    raise SystemExit(main())
