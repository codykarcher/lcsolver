#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""What solvers this installation actually has, and which one will be used.

Answers three questions the solve path doesn't:
1. Which ``ipopt`` binary wins? conda activate prepends $CONDA_PREFIX/bin,
   so a MUMPS build can silently shadow a source-built MA27 one.
2. Which linear solver does that binary carry? Fixed at IPOPT build time and
   the biggest determinant of GP/SP behaviour; has to be probed by solving.
3. Is cyipopt (the in-process route, required for black-box constraints)
   present, and linked against the same IPOPT as the executable?

Set ``LCSOLVER_IPOPT_EXECUTABLE`` to pin the executable regardless of PATH
order (reordering PATH against conda activate only lasts until the next shell).
"""

import contextlib
import io
import logging
import os
import shutil
import sys

IPOPT_EXECUTABLE_ENV = 'LCSOLVER_IPOPT_EXECUTABLE'
AUTOSELECT_ENV = 'LCSOLVER_IPOPT_AUTOSELECT'

# Where the installer records an IPOPT it built. Source builds land off-PATH,
# and asking users to edit a shell profile silently leaves the build unused.
STATE_FILE = os.path.join(os.path.expanduser('~'), '.config', 'lcsolver',
                          'solvers.json')

# Probing costs a solve, so cache per (executable, linear_solver) for the
# session. Nothing here changes while a process is running.
_PROBE_CACHE = {}

# Cache the resolution too: it can probe, and it's consulted once per solve,
# including inside the SLCP/SIA loops.
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

    LCSOLVER_IPOPT_EXECUTABLE always wins (even if broken -- a bad pin is a
    mistake to report, not route around). Otherwise an MA27 build beats one
    without it regardless of PATH order: conda activate prepends
    $CONDA_PREFIX/bin, so a prebuilt MUMPS ipopt silently shadows a source
    build made specifically for MA27. Probes only when there is more than one
    candidate; LCSOLVER_IPOPT_AUTOSELECT=0 restores strict PATH order.
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
    """Is there any usable IPOPT here -- the executable, or cyipopt?

    Not cached: caching would make an IPOPT installed mid-session invisible.
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


def ensure_own_libs_first(executable):
    """Make the executable load ITS OWN libipopt, not a shadowed one.

    A loader path (DYLD_LIBRARY_PATH) pinned to one install shadows every
    other libipopt by name -- once made three MUMPS binaries probe as
    MA27-only. Prepend the executable's sibling lib dir; no-op without one.
    """
    if not executable:
        return
    var = 'DYLD_LIBRARY_PATH' if sys.platform == 'darwin' else 'LD_LIBRARY_PATH'
    try:
        libdir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.realpath(executable)),
                         os.pardir, 'lib'))
        import glob
        if not glob.glob(os.path.join(libdir, 'libipopt*')):
            return
        current = os.environ.get(var, '')
        parts = [p for p in current.split(os.pathsep) if p]
        if parts and os.path.normpath(parts[0]) == libdir:
            return
        os.environ[var] = os.pathsep.join(
            [libdir] + [p for p in parts if os.path.normpath(p) != libdir])
    except Exception:
        pass


def ipopt_solver_factory(executable=None):
    """``SolverFactory('ipopt')`` that honours the executable pin.

    All AMPL-route callers should use this, or LCSOLVER_IPOPT_EXECUTABLE gets
    respected on some paths and ignored on others (notably the SLCP/SIA loops).
    """
    import pyomo.environ as pyo

    exe = executable or ipopt_executable()
    ensure_own_libs_first(exe)
    return (pyo.SolverFactory('ipopt', executable=exe) if exe
            else pyo.SolverFactory('ipopt'))


def ipopt_executables_on_path():
    """Every ``ipopt`` on PATH, in shell lookup order. Used to detect shadowing."""
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

    Three layers: the Pyomo logger, Python-level stdout/stderr, and IPOPT's
    C++ output, which writes to the file descriptors directly -- only dup2
    reaches that one.
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
        # No usable fds (embedded interpreter, some notebook kernels); the
        # Python-level redirect below still applies.
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


# Every linear solver an upstream IPOPT can be built against.  The headline
# three (mumps, ma27, pardiso) compose the availability message
KNOWN_IPOPT_LINEAR_SOLVERS = (
    'mumps', 'ma27', 'ma57', 'ma77', 'ma86', 'ma97',
    'pardiso', 'pardisomkl', 'spral', 'wsmp',
)

# Solvers IPOPT dlopens at runtime, and the option naming their library.
# No rebuild needed: hand hsllib a full CoinHSL (ma57/77/86/97) or
# pardisolib a Panua Pardiso and the stock build runs them
RUNTIME_LOADED_LINEAR_SOLVERS = {
    'ma57': 'hsllib', 'ma77': 'hsllib', 'ma86': 'hsllib', 'ma97': 'hsllib',
    'pardiso': 'pardisolib',
}


def linear_solver_library_option(name):
    """'hsllib' or 'pardisolib' for a runtime-loaded solver, else None"""
    return RUNTIME_LOADED_LINEAR_SOLVERS.get(str(name).strip().lower())


def _spral_runtime_env():
    """SPRAL's required OpenMP settings, set process-wide (the ipopt
    executable inherits them); explicit user settings are left alone"""
    os.environ.setdefault('OMP_CANCELLATION', 'TRUE')
    os.environ.setdefault('OMP_PROC_BIND', 'TRUE')


def require_linear_solver(name, route='pyomo', executable=None,
                          library=None):
    """Validate and probe a requested IPOPT ``linear_solver``.

    Returns the normalized name if the build on ``route`` carries it.
    ValueError for an unknown name; SolverUnavailable (naming what IS
    available) when the build lacks it, so it reads as an install gap rather
    than IPOPT dying mid-solve. An inconclusive cyipopt probe passes the name
    through untested -- guessing False would refuse working builds.
    ``library`` is the shared lib for a runtime-loaded solver (hsllib/
    pardisolib); refused for a compiled-in one.
    """
    from lcsolver.core.errors import SolverUnavailable

    key = str(name).strip().lower()
    if key not in KNOWN_IPOPT_LINEAR_SOLVERS:
        raise ValueError(
            'unknown linear_solver %r. IPOPT linear solvers are: %s'
            % (name, ', '.join(KNOWN_IPOPT_LINEAR_SOLVERS)))

    libopt = linear_solver_library_option(key)
    if library is not None:
        if libopt is None:
            raise ValueError(
                'linear_solver_library was given, but %r is not a '
                'runtime-loaded solver (those are: %s); the library would '
                'be ignored' % (key,
                                ', '.join(sorted(
                                    RUNTIME_LOADED_LINEAR_SOLVERS))))
        library = os.path.expanduser(str(library))
        if not os.path.exists(library):
            raise SolverUnavailable(
                'the linear_solver_library for %r does not exist: %s'
                % (key, library))

    if key == 'spral':
        _spral_runtime_env()

    if route == 'pyomo':
        ok = linear_solver_available(key, executable, library=library)
    else:
        ok = cyipopt_linear_solver_available(key)
        if ok is None:
            return key
    if not ok:
        if route == 'pyomo':
            present = [s for s in ('ma27', 'mumps', 'pardiso')
                       if linear_solver_available(s, executable)]
        else:
            present = [s for s in ('ma27', 'mumps', 'pardiso')
                       if cyipopt_linear_solver_available(s)]
        _lib_note = ''
        if libopt is not None and library is None:
            _lib_note = (' %r is a runtime-loaded solver: pass '
                         'linear_solver_library=<path to the %s library> '
                         'if you have one.' % (key,
                                               'CoinHSL' if libopt == 'hsllib'
                                               else 'Pardiso'))
        raise SolverUnavailable(
            "the linear solver %r is not available in this IPOPT build "
            "(route: %s%s). Probed as available here: %s.%s See "
            "docs/linear_solvers.rst for building IPOPT with additional "
            "linear solvers."
            % (key, route,
               f', executable {executable}' if executable else '',
               ', '.join(present) if present else 'none of ma27/mumps/pardiso',
               _lib_note))
    return key


# IPOPT options LCsolver sets whenever a linear solver is selected, unless
# the caller set them. SPRAL's stock settings (IPOPT mirrors MA97's:
# u=1e-8, matching-based scaling) let it declare the D8's KKT systems
# singular and then die on the regularized re-factorization -- the bundled
# crash sentinel examples/data/d8_spral_crash.nl. MC64 scaling is the one
# setting that cleared the whole deck (358 sub-problems, 0 crashes, and
# faster: 86 s vs 103 s); spral_small and spral_u only moved the crashes.
# Measured 2026-09-15, docs/linear_solvers.rst.
LINEAR_SOLVER_DEFAULT_OPTIONS = {
    'spral': {'spral_scaling': 'mc64'},
}


def linear_solver_default_options(name):
    """LCsolver's option overrides for ``name`` (a fresh dict; {} for a
    solver with none)."""
    return dict(LINEAR_SOLVER_DEFAULT_OPTIONS.get(
        str(name or '').strip().lower(), {}))


def apply_linear_solver_defaults(options, name):
    """setdefault linear_solver_default_options(name) into ``options`` (an
    IPOPT option mapping) and return it; an explicit user setting wins."""
    for k, v in linear_solver_default_options(name).items():
        options.setdefault(k, v)
    return options


class IpoptCrashed(RuntimeError):
    """The ipopt executable died -- a signal (SIGBUS, SIGSEGV) or a non-zero
    exit, not an IPOPT status. A RuntimeError on purpose: the sequential
    solvers already treat one as a failed sub-problem (shrink, restore,
    retry) instead of aborting the whole solve."""


@contextlib.contextmanager
def ipopt_launch(linear_solver=None, executable=None, recoverable=False):
    """Run ONE pyomo ipopt solve inside this block.

    pyomo reports a dead executable as raw ERROR log lines (the return code
    and the whole solver log) plus a bare ApplicationError. Here the log is
    captured instead of printed and the failure comes back as IpoptCrashed
    naming the signal, the linear solver, and the MA27 note. recoverable=
    True (the SIA/SLCP sub-problem loops, which carry on from a failed
    sub-problem) also files an [LC-W313] warning so the event shows in the
    post-solve report instead of scrolling past. Anything pyomo logged on
    a solve that did NOT crash is re-emitted untouched.
    """
    import re
    import signal
    import warnings

    from pyomo.common.errors import ApplicationError

    log = logging.getLogger('pyomo.opt')
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    propagate = log.propagate
    log.addHandler(handler)
    log.propagate = False
    try:
        yield
    except ApplicationError as exc:
        rc = None
        for r in records:
            hit = re.search(r'non-zero return code \((-?\d+)\)',
                            r.getMessage())
            if hit:
                rc = int(hit.group(1))
        if rc is not None and rc < 0:
            try:
                how = f'{signal.Signals(-rc).name} (signal {-rc})'
            except ValueError:
                how = f'signal {-rc}'
        elif rc is not None:
            how = f'exit code {rc}'
        else:
            how = 'no exit status'
        msg = ('the ipopt executable crashed with %s%s.%s'
               % (how,
                  f' under linear_solver={linear_solver!r}'
                  if linear_solver else '',
                  linear_solver_failure_note(linear_solver, executable)))
        if recoverable:
            warnings.warn(
                '[LC-W313] %s Treated as a failed sub-problem: the loop '
                'retries under MA27 where the build has it, otherwise '
                'shrinks its step and carries on.' % msg, stacklevel=3)
        raise IpoptCrashed(msg) from exc
    else:
        log.removeHandler(handler)
        log.propagate = propagate
        for r in records:
            log.handle(r)
    finally:
        if handler in log.handlers:
            log.removeHandler(handler)
        log.propagate = propagate


def crash_fallback_solver(linear_solver, executable=None):
    """The linear solver to retry a crashed sub-problem under: MA27 when the
    build has it and it was not already in use, else None."""
    if str(linear_solver or 'ma27').lower() == 'ma27':
        return None
    try:
        return 'ma27' if linear_solver_available('ma27', executable) else None
    except Exception:
        return None


def linear_solver_failure_note(linear_solver, executable=None):
    """Advisory to append when a solve FAILS under a non-MA27 linear solver.

    MA27 is the measured most-robust solver on this problem class (the D8
    sentinel breaks MUMPS outright -- docs/linear_solvers.rst), so a failure
    under anything else names the retry.  Empty for MA27 itself, and for an
    unspecified solver on an MA27 build (that failure is not about the
    linear solver).  Never raises: this decorates an error path.
    """
    key = str(linear_solver).strip().lower() if linear_solver else None
    if key in (None, '', 'ma27'):
        return ''
    try:
        have_ma27 = bool(linear_solver_available('ma27', executable))
    except Exception:
        have_ma27 = False
    if have_ma27:
        return (' Note: this solve used linear_solver=%r; MA27 is the most '
                'robust linear solver on this problem class -- retry with '
                "linear_solver='ma27'." % key)
    return (' Note: this solve used linear_solver=%r, and this IPOPT build '
            'has no MA27 -- the most robust linear solver on this problem '
            'class. Build one with `lcsolver-install-solvers --ma27 <path>` '
            '(see docs/linear_solvers.rst).' % key)


def linear_solver_available(name, executable=None, library=None):
    """Does this IPOPT build carry the ``name`` linear solver?

    Probed by solving a one-variable problem with ``linear_solver <name>`` --
    there is no reliable way to ask the binary directly. ``library`` rides
    along as hsllib/pardisolib for a runtime-loaded solver.
    """
    executable = executable or ipopt_executable()
    key = (os.path.realpath(executable) if executable else None, name,
           os.path.realpath(library) if library else None)
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]

    result = False
    try:
        import pyomo.environ as pyo
        from pyomo.opt import TerminationCondition

        probe = pyo.ConcreteModel()
        probe.x = pyo.Var(initialize=1.0, bounds=(0.5, None))
        probe.o = pyo.Objective(expr=probe.x)
        # Probe the binary's OWN library, not whatever a loader path
        # shadows it with -- see ensure_own_libs_first.
        ensure_own_libs_first(executable)
        opt = (pyo.SolverFactory('ipopt', executable=executable)
               if executable else pyo.SolverFactory('ipopt'))
        opt.options['linear_solver'] = name
        _libopt = linear_solver_library_option(name)
        if library and _libopt:
            opt.options[_libopt] = library
        if name == 'spral':
            _spral_runtime_env()
        with _quiet():
            res = opt.solve(probe, tee=False, load_solutions=False)
        result = (res.solver.termination_condition
                  == TerminationCondition.optimal)
    except Exception:
        result = False

    _PROBE_CACHE[key] = result
    return result


# Run in a child interpreter: IPOPT's C++ journalist flushes its option-error
# dump at process exit, after any dup2 redirection is undone, so in-process it
# cannot be silenced. One child answers all three questions.
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

    True/False, or None when the probe could not run at all (cyipopt absent
    or broken) -- a different fact from "this build lacks MA27".
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
    """Collect the solver situation as a plain dict. ``probe=False`` skips
    the linear-solver probes (each is a tiny real solve) and leaves the
    ma27/mumps fields None."""
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

            # An MA27 build losing to one without it: resolution prefers MA27,
            # so a pin or AUTOSELECT=0 overrode it -- say so out loud.
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

        # cyipopt alone is not a working in-process route: PyNumero's ASL
        # library ships separately, and without it every black-box solve fails.
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

    # A mismatch isn't an error, but the two routes behaving differently on
    # the same model is otherwise mystifying.
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
            # always say why this binary won; with several installed it matters
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


# Exit status for "report ran fine, install is incomplete". Distinct from 1
# so a missing solver is distinguishable from this command falling over.
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
