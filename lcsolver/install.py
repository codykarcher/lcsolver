#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Get a working solver stack: cvxopt, IPOPT, cyipopt.

pip cannot finish the job: there is no IPOPT executable on PyPI, and cyipopt
is sdist-only, so it compiles against an IPOPT already on the machine. IPOPT
has to come from a package manager (conda-forge, apt, Homebrew) or source.
Python rather than shell so it behaves the same on Windows.

Default install is a MUMPS build (the only redistributable linear solver).
MA27 -- what you actually want under a GP/SP, see docs/ipopt.rst -- is free
for academic use but cannot be redistributed, so:
  ``--ma27 <path>``       build IPOPT against MA27 sources, relink cyipopt
  ``--relink-cyipopt``    just the cyipopt half, against an existing MA27 build
Both run the shipped install_ipopt.sh.

Nothing here modifies an environment without printing the plan and asking.
"""

import os
import shutil
import subprocess
import sys

from lcsolver.environment import (
    IPOPT_EXECUTABLE_ENV,
    check_solvers,
    ipopt_executable,
    linear_solver_available,
    record_ipopt,
)

# Places MA27 sources plausibly sit after a manual download. Checked only to
# offer the upgrade; never used without saying so.
_MA27_SEARCH = (
    '$MA27_SOURCE',
    '$HSL_SOURCE',
    '~/software/MA27/ma27-1.0.0',
    './MA27/ma27-1.0.0',
    '~/MA27/ma27-1.0.0',
    '~/Downloads/ma27-1.0.0',
    '~/Downloads/coinhsl-2023.11.17',
)


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------
class Step:
    """One command plus a human sentence, so the whole plan can be printed
    and approved before any of it runs."""

    def __init__(self, description, command=None, action=None, env=None,
                 optional=False):
        self.description = description
        self.command = command
        self.action = action
        self.env = env
        # A step whose failure is worth reporting but not worth aborting
        # the install over.
        self.optional = optional

    def show(self):
        if self.command:
            return f'  {self.description}\n      $ {" ".join(self.command)}'
        return f'  {self.description}'

    def run(self):
        if self.action is not None:
            # print it too: some steps are instructions (anything needing root)
            # and would otherwise run silently
            print(f'\n==> {self.description}')
            return self.action()
        environ = dict(os.environ)
        environ.update(self.env or {})
        print(f'\n==> {self.description}')
        print(f'    $ {" ".join(self.command)}\n')
        return subprocess.run(self.command, env=environ).returncode


def _conda():
    """The conda executable and the environment it would modify, or None.

    CONDA_PREFIX first: the question is which environment is active now --
    the one an unqualified ``conda install`` writes into.
    """
    exe = shutil.which('conda') or shutil.which('mamba') or shutil.which('micromamba')
    if not exe:
        return None

    prefix = os.environ.get('CONDA_PREFIX')
    if not prefix:
        # conda installed but nothing activated: ask conda where an install
        # would land rather than sending the user to a source build
        try:
            import json
            out = subprocess.run([exe, 'info', '--json'], capture_output=True,
                                 text=True, timeout=120)
            info = json.loads(out.stdout)
            prefix = info.get('active_prefix') or info.get('default_prefix')
        except Exception:
            prefix = None
    if not prefix:
        return None

    return {'exe': exe, 'prefix': prefix,
            'name': os.environ.get('CONDA_DEFAULT_ENV', os.path.basename(prefix)),
            'activated': bool(os.environ.get('CONDA_PREFIX'))}


def _packaged_script(name):
    """Absolute path to a shipped helper script. Looked up inside the
    installed package, not the repo -- a wheel install has no utilities/."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, 'scripts', name)
    if not os.path.isfile(path):
        raise RuntimeError(
            f'{name} is missing from the installed package (looked in '
            f'{os.path.dirname(path)}). This is a packaging bug.')
    return path


def _ipopt_build_prefix(executable):
    """``<prefix>/bin/ipopt`` -> ``<prefix>``, which is what cyipopt links to."""
    return os.path.dirname(os.path.dirname(os.path.realpath(executable)))


def _looks_like_ma27(path):
    """Is there actually MA27 Fortran under here? The HSL build happily
    produces a library with no solver in it -- catch that up front."""
    if not path or not os.path.isdir(path):
        return False
    for root in (path, os.path.join(path, 'src'), os.path.join(path, 'ma27')):
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            low = entry.lower()
            if low.startswith('ma27') and low.endswith(('.f', '.f90', '.F')):
                return True
    return False


def _find_ma27_sources():
    for candidate in _MA27_SEARCH:
        path = os.path.expanduser(os.path.expandvars(candidate))
        if _looks_like_ma27(path):
            return os.path.abspath(path)
    return None


def _confirm(prompt, assume_yes):
    if assume_yes:
        return True
    try:
        answer = input(f'{prompt} [y/N] ').strip().lower()
    except EOFError:
        return False
    return answer in ('y', 'yes')


# --------------------------------------------------------------------------
# plan construction
# --------------------------------------------------------------------------
def _plan_default(report, conda, args):
    """Steps for the plain install: cvxopt, IPOPT+MUMPS, cyipopt.

    Returns ``(steps, source_build_root)``; the root is set only when IPOPT
    is built from source, so the caller can print the exports (a source build
    is on nobody's PATH).
    """
    steps = []
    source_root = None

    def finish(collected):
        """Append the PyNumero ASL fetch, whatever else was decided. A helper
        because every early return above needs it too -- an env that already
        has ipopt+cyipopt may be missing only the ASL library."""
        if not args.skip_cyipopt and not _pynumero_asl_available():
            collected.append(_plan_pynumero_asl())
        return collected

    if not report['cvxopt']['available'] and not args.skip_cvxopt:
        steps.append(Step(
            'install cvxopt (wheels exist for every supported platform)',
            command=[sys.executable, '-m', 'pip', 'install', 'cvxopt']))

    exe = report['ipopt']['executable']
    have_ipopt = bool(exe) and os.path.isfile(exe)
    have_cyipopt = report['cyipopt']['available']
    want_ipopt = not have_ipopt and not args.skip_ipopt
    want_cyipopt = not have_cyipopt and not args.skip_cyipopt

    # MA27 executable already here, cyipopt missing: build cyipopt against
    # THAT IPOPT. `conda install cyipopt` would link conda's MUMPS one, so
    # black-box models would quietly run MUMPS while everything else ran MA27.
    if want_cyipopt and have_ipopt and report['ipopt'].get('ma27') is True:
        steps.extend(_plan_relink(_ipopt_build_prefix(exe)))
        want_cyipopt = False

    if not (want_ipopt or want_cyipopt):
        return finish(steps), source_root

    if conda:
        packages = []
        if want_ipopt:
            packages.append('ipopt')
        if want_cyipopt:
            packages.append('cyipopt')
        steps.append(Step(
            f'install {" and ".join(packages)} from conda-forge into '
            f'{conda["prefix"]}',
            command=[conda['exe'], 'install', '-y', '-c', 'conda-forge'] + packages))
        return finish(steps), source_root

    # No conda. IPOPT then has to come from the system package manager, and
    # cyipopt has to compile against it.
    if want_ipopt:
        if sys.platform == 'darwin' and shutil.which('brew'):
            steps.append(Step(
                'install IPOPT (MUMPS build) from Homebrew',
                command=['brew', 'install', 'ipopt']))
        elif sys.platform.startswith('linux') and shutil.which('apt-get'):
            # Not run: it needs root, and this script should not be the thing
            # that asks for root.
            steps.append(Step(
                'IPOPT must be installed with apt, which needs root. Run this '
                'yourself, then re-run lcsolver-install-solvers:\n'
                '      $ sudo apt-get install -y coinor-libipopt-dev coinor-libipopt1v5',
                action=lambda: 0))
        elif sys.platform.startswith('win'):
            steps.append(Step(
                'IPOPT on Windows without conda is not something this script '
                'can do. Install conda (https://conda-forge.org/download/) '
                'and re-run.',
                action=lambda: 0))
        else:
            # last resort: MUMPS is redistributable, so unlike MA27 this
            # needs no manual download and can run unattended
            source_root = _ma27_build_root(args)
            steps.append(Step(
                f'build IPOPT against MUMPS from source into '
                f'{source_root}/ipopt\n'
                f'      (no conda or Homebrew here to install a prebuilt one; '
                f'needs gfortran, and takes a while)',
                command=['bash', _packaged_script('install_ipopt.sh'),
                         source_root],
                env={'IPOPT_BUILD_MUMPS': '1'}))

    if want_cyipopt:
        if source_root:
            # point at what was just built; the source build isn't on PATH yet
            steps.extend(_plan_relink(os.path.join(source_root, 'ipopt', 'build')))
        else:
            steps.append(Step(
                'install cyipopt (compiles from source against the IPOPT '
                'above; needs a C compiler and pkg-config)',
                command=[sys.executable, '-m', 'pip', 'install', 'cyipopt']))

    return finish(steps), source_root


def _pynumero_asl_available():
    """Can Pyomo's in-process NLP interface actually load its ASL library?

    The pynumero_ASL shared library ships with neither pyomo nor cyipopt;
    without it every black-box solve dies with "Cannot load the PyNumero ASL
    interface".
    """
    try:
        from pyomo.contrib.pynumero.asl import AmplInterface
        return bool(AmplInterface.available())
    except Exception:
        return False


def _plan_pynumero_asl():
    """Build the PyNumero ASL library. Two dead ends that look like the
    answer: `pyomo download-extensions` does not provide it (exits 0 having
    changed nothing), and conda-forge's pynumero_libraries needs Python <=3.8.
    So it has to be compiled (cmake + C compiler) -- optional, since a build
    failure only loses the black-box models, not the executable route.
    """
    return Step(
        "build Pyomo's PyNumero ASL library, so cyipopt can evaluate a model "
        "at all (needs cmake and a C compiler)",
        # call the PyNumero builder directly: `pyomo build-extensions` fails
        # as a whole when any extension fails (MC++ wanting pybind11 kills it
        # on stock macOS before PyNumero is even reached)
        command=[sys.executable, '-c',
                 'from pyomo.contrib.pynumero.build import build_pynumero; '
                 'build_pynumero()'],
        optional=True)


def _ma27_build_root(args):
    return os.path.abspath(os.path.expanduser(args.ma27_root))


def _plan_ma27(sources, args):
    """Steps to build IPOPT against MA27 and point cyipopt at it. Delegates
    the build to the shipped install_ipopt.sh."""
    if sys.platform.startswith('win'):
        raise RuntimeError(
            'building IPOPT against MA27 needs bash and autotools, which this '
            'script cannot drive on Windows. Build under WSL, or stay on the '
            'conda-forge MUMPS build.')

    root = _ma27_build_root(args)
    script = _packaged_script('install_ipopt.sh')
    build = os.path.join(root, 'ipopt', 'build')

    steps = [Step(
        f'build IPOPT against MA27 from {sources}\n'
        f'      (into {root}/ipopt -- clones IPOPT, ASL and HSL, then compiles; '
        f'this takes a while)',
        command=['bash', script, root],
        # MA27_SRC lets the script take sources from wherever the user put
        # them, without changing its one-argument form
        env={'MA27_SRC': sources})]

    if not args.skip_cyipopt:
        steps.extend(_plan_relink(build))
        # --ma27 is a complete install in its own right and skips the default
        # planner -- without this the documented quickstart built everything
        # and still couldn't evaluate a black box
        if not _pynumero_asl_available():
            steps.append(_plan_pynumero_asl())

    return steps


def _plan_relink(build):
    """Rebuild cyipopt against a specific IPOPT build. cyipopt links whatever
    it was compiled against -- a conda cyipopt stays on conda's MUMPS forever,
    and black-box models are stuck with it."""
    include = os.path.join(build, 'include', 'coin-or')
    lib = os.path.join(build, 'lib')
    env = {
        'PKG_CONFIG_PATH': os.path.join(lib, 'pkgconfig')
                           + os.pathsep + os.environ.get('PKG_CONFIG_PATH', ''),
        # cyipopt has used more than one spelling across releases; setting all
        # of them costs nothing and avoids a version-dependent failure.
        'IPOPT_INCLUDE_DIR': include,
        'IPOPT_LIBRARY_DIR': lib,
        'IPOPT_INCLUDE_DIRS': include,
        'IPOPT_LIB_DIRS': lib,
    }
    libvar = 'DYLD_LIBRARY_PATH' if sys.platform == 'darwin' else 'LD_LIBRARY_PATH'
    env[libvar] = lib + os.pathsep + os.environ.get(libvar, '')

    return [Step(
        f'build cyipopt against {build}, so the in-process route (the only one '
        f'that can evaluate a black box) uses the same linear solver as the '
        f'executable',
        command=[sys.executable, '-m', 'pip', 'install', '--force-reinstall',
                 '--no-cache-dir', '--no-binary', 'cyipopt', 'cyipopt'],
        env=env)]


# --------------------------------------------------------------------------
# after the fact
# --------------------------------------------------------------------------
def _finish_source_build(root, linear_solver='MA27'):
    """Record the build and say what, if anything, is left to do. LCsolver
    records the build and prefers it, so no shell-profile export is needed."""
    build = os.path.join(root, 'ipopt', 'build')
    exe = os.path.join(build, 'bin', 'ipopt')

    print('\n' + '-' * 74)
    print(f'IPOPT with {linear_solver} is built: {exe}')

    if not os.path.isfile(exe):
        print('\nwarning: the build finished but no binary is at that path.')
        return

    try:
        state = record_ipopt(exe)
        print(f'\nRecorded in {state}. LCsolver will use this build from any '
              f'environment,\nwithout a PATH change -- it prefers an MA27 '
              f'build over a prebuilt MUMPS one\neven when conda puts conda\'s '
              f'first.')
    except Exception as exc:
        # Not fatal: the pin still works, it is just manual again.
        print(f'\nwarning: could not record the build ({exc}). To make '
              f'LCsolver use it, set\n'
              f'    export {IPOPT_EXECUTABLE_ENV}="{exe}"')

    print('\nCheck it:\n    lcsolver-check-solvers')
    print('-' * 74)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog='lcsolver-install-solvers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Install the LCsolver solver stack: cvxopt, IPOPT, cyipopt.',
        epilog='With no options: installs cvxopt and a MUMPS build of IPOPT '
               'plus cyipopt, and reports MA27 as the available upgrade.')
    parser.add_argument('--ma27', metavar='PATH', default=None,
                        help='build IPOPT against the MA27 sources at PATH '
                             '(free for academic use; download from '
                             'https://www.hsl.rl.ac.uk/download/MA27/1.0.0/)')
    parser.add_argument('--ma27-root', metavar='DIR', default='~/software',
                        help='where to build IPOPT+MA27 (default: ~/software)')
    parser.add_argument('--relink-cyipopt', action='store_true',
                        help='rebuild cyipopt against an existing MA27 build '
                             'of IPOPT, and do nothing else')
    parser.add_argument('--skip-ipopt', action='store_true',
                        help='do not touch IPOPT')
    parser.add_argument('--skip-cyipopt', action='store_true',
                        help='do not install or rebuild cyipopt')
    parser.add_argument('--skip-cvxopt', action='store_true',
                        help='do not install cvxopt')
    parser.add_argument('-n', '--dry-run', action='store_true',
                        help='print the plan and exit without running it')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='do not ask for confirmation')
    args = parser.parse_args(argv)

    print('Checking what is already installed...\n')
    report = check_solvers(probe=True)
    from lcsolver.environment import _fmt
    print(_fmt(report))
    print()

    # A broken pin has to be fixed by the user; it outranks anything
    # installed here.
    pinned = os.environ.get(IPOPT_EXECUTABLE_ENV)
    if pinned and not os.path.isfile(pinned):
        print(f'error: {IPOPT_EXECUTABLE_ENV} is set to {pinned}, which does '
              f'not exist. LCsolver honours it over PATH, so nothing installed '
              f'here would be used. Unset or correct it, then re-run.',
              file=sys.stderr)
        return 1

    conda = _conda()

    # ---- decide what to do ------------------------------------------------
    try:
        if args.relink_cyipopt:
            exe = ipopt_executable()
            if not exe or not os.path.isfile(exe):
                print('error: no ipopt executable to link against.', file=sys.stderr)
                return 1
            build = _ipopt_build_prefix(exe)
            if not linear_solver_available('ma27', exe):
                print(f'error: {exe} does not have MA27, so relinking cyipopt '
                      f'against it would gain nothing. Build one first with '
                      f'--ma27 <path>.', file=sys.stderr)
                return 1
            steps = _plan_relink(build)
            built_root = built_solver = None
        elif args.ma27:
            sources = os.path.abspath(os.path.expanduser(args.ma27))
            if not _looks_like_ma27(sources):
                print(f'error: no MA27 Fortran sources found at {sources}. '
                      f'Expected the extracted archive, containing ma27*.f '
                      f'either at the top level or under src/.', file=sys.stderr)
                return 1
            steps = _plan_ma27(sources, args)
            built_root, built_solver = _ma27_build_root(args), 'MA27'
        else:
            steps, built_root = _plan_default(report, conda, args)
            built_solver = 'MUMPS'
    except RuntimeError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1

    if not steps:
        print('Nothing to do -- everything this script installs is present.')
        if report['ipopt']['ma27'] is False:
            found = _find_ma27_sources()
            if found:
                print(f'\nMA27 sources found at {found}. To upgrade off MUMPS:\n'
                      f'    lcsolver-install-solvers --ma27 {found}')
            else:
                print('\nThe IPOPT here is a MUMPS build. MA27 is the upgrade; '
                      'see docs/ipopt.rst.')
        return 0

    # ---- show it, then ask ------------------------------------------------
    print('Plan:')
    for step in steps:
        print(step.show())
    if conda and not args.ma27 and not args.relink_cyipopt:
        if conda.get('activated'):
            where = f'the active conda environment "{conda["name"]}"'
        else:
            # no env active means conda picks base, a choice people regret
            where = (f'the conda environment at {conda["prefix"]} -- no '
                     f'environment is currently activated, so this is conda\'s '
                     f'default')
        print(f'\nThis modifies {where}.\n'
              f'To use a separate one instead, create and activate it first:\n'
              f'    conda create -n lcsolver -c conda-forge python=3.12\n'
              f'    conda activate lcsolver')

    if args.dry_run:
        print('\n(dry run -- nothing was executed)')
        return 0
    if not _confirm('\nProceed?', args.yes):
        print('Aborted.')
        return 1

    # ---- run --------------------------------------------------------------
    for step in steps:
        code = step.run()
        if code and step.optional:
            print(f'\nwarning: this step failed ({code}) and the install '
                  f'continues without it:\n  {step.description}',
                  file=sys.stderr)
        elif code:
            print(f'\nerror: step failed ({code}): {step.description}',
                  file=sys.stderr)
            return code

    # ---- report ------------------------------------------------------------
    if built_root:
        # a source build is not on PATH; the exports are a necessary step
        _finish_source_build(built_root, built_solver)
        return 0

    print('\nDone. Re-checking...\n')
    from lcsolver import environment
    environment._PROBE_CACHE.clear()
    report = check_solvers(probe=True)
    print(environment._fmt(report))

    if report['ipopt']['ma27'] is False:
        found = _find_ma27_sources()
        print('\nThis is a MUMPS build, which is a working install. MA27 is '
              'more robust on geometric and signomial programs '
              '(see docs/ipopt.rst).')
        if found:
            print(f'MA27 sources are already on this machine ({found}):\n'
                  f'    lcsolver-install-solvers --ma27 {found}')
        else:
            print('Register at https://www.hsl.rl.ac.uk/download/MA27/1.0.0/, '
                  'extract, then:\n'
                  '    lcsolver-install-solvers --ma27 <extracted-path>')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
