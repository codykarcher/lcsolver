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

Default install builds IPOPT from source with the open-source linear
solvers, MUMPS and SPRAL (SPRAL is then the default; it certifies
everything MA27 does -- docs/linear_solvers.rst), and cyipopt against it.
MA27 is faster, free for academic use, and not redistributable, so:
  ``--ma27 <path>``       include MA27 in the initial build
  ``--add-ma27 <path>``   add MA27 to a build made earlier without it
  ``--relink-cyipopt``    just the cyipopt half, against the existing build
  ``--prebuilt``          a conda/Homebrew/apt IPOPT instead of a source build
All source builds run the shipped install_ipopt.sh.

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
MA27_SEARCH = (
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


def conda_environment():
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


def packaged_script(name):
    """Absolute path to a shipped helper script. Looked up inside the
    installed package, not the repo -- a wheel install has no utilities/."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, 'scripts', name)
    if not os.path.isfile(path):
        raise RuntimeError(
            f'{name} is missing from the installed package (looked in '
            f'{os.path.dirname(path)}). This is a packaging bug.')
    return path


def ipopt_build_prefix(executable):
    """``<prefix>/bin/ipopt`` -> ``<prefix>``, which is what cyipopt links to."""
    return os.path.dirname(os.path.dirname(os.path.realpath(executable)))


def looks_like_ma27(path):
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


def find_ma27_sources():
    for candidate in MA27_SEARCH:
        path = os.path.expanduser(os.path.expandvars(candidate))
        if looks_like_ma27(path):
            return os.path.abspath(path)
    return None


def confirm(prompt, assume_yes):
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
def is_source_build(exe):
    """An IPOPT this script built (or one laid out like it): its prefix
    carries the pkgconfig cyipopt links against."""
    return os.path.isfile(os.path.join(ipopt_build_prefix(exe), 'lib',
                                       'pkgconfig', 'ipopt.pc'))


def can_source_build():
    """bash, git and a Fortran compiler: what install_ipopt.sh needs before
    it can even start (SPRAL's extra needs it checks and reports itself)."""
    if sys.platform.startswith('win'):
        return False
    have_fc = any(shutil.which(f'gfortran{s}')
                  for s in ('', '-16', '-15', '-14', '-13', '-12', '-11'))
    return bool(shutil.which('bash') and shutil.which('git') and have_fc)


def plan_source_build_step(root, args, ma27=None, add=False):
    """The install_ipopt.sh step: MUMPS + SPRAL by default, MA27 included
    (``--with-ma27``) or added to an existing root (``--add-ma27``)."""
    cmd = ['bash', packaged_script('install_ipopt.sh'), root]
    if ma27:
        cmd += ['--add-ma27' if add else '--with-ma27', ma27]
    if getattr(args, 'no_spral', False):
        cmd.append('--no-spral')
    if getattr(args, 'no_mumps', False):
        cmd.append('--no-mumps')
    if add:
        what = f'add MA27 (from {ma27}) to the IPOPT under {root}/ipopt'
    else:
        solvers = ' + '.join(s for s, on in (
            ('MUMPS', not getattr(args, 'no_mumps', False)),
            ('SPRAL', not getattr(args, 'no_spral', False)),
            ('MA27', bool(ma27))) if on)
        what = (f'build IPOPT with {solvers} from source into {root}/ipopt\n'
                f'      (clones IPOPT and the solvers, then compiles; needs '
                f'gfortran -- and for SPRAL a GCC toolchain, metis and '
                f'hwloc, else it is skipped with a note; takes a while)')
    # MA27_SRC alongside the flag: older callers of the script read the env
    return Step(what, command=cmd, env={'MA27_SRC': ma27} if ma27 else None)


def plan_default(report, conda, args):
    """Steps for the plain install: cvxopt, IPOPT (MUMPS + SPRAL from
    source), cyipopt against it. ``--prebuilt`` takes a conda/Homebrew/apt
    IPOPT instead -- MA27-only or MUMPS-only, never SPRAL.

    Returns ``(steps, source_build_root)``; the root is set only when IPOPT
    is built from source, so the caller can print the exports (a source build
    is on nobody's PATH).
    """
    steps = []
    source_root = None

    if not report['cvxopt']['available'] and not args.skip_cvxopt:
        steps.append(Step(
            'install cvxopt (wheels exist for every supported platform)',
            command=[sys.executable, '-m', 'pip', 'install', 'cvxopt']))

    exe = report['ipopt']['executable']
    have_ipopt = bool(exe) and os.path.isfile(exe)
    have_cyipopt = report['cyipopt']['available']
    want_ipopt = not have_ipopt and not args.skip_ipopt
    want_cyipopt = not have_cyipopt and not args.skip_cyipopt

    # A build of ours (or any MA27 build) already here, cyipopt missing:
    # build cyipopt against THAT IPOPT. `conda install cyipopt` would link
    # conda's own IPOPT, so black-box models would quietly run a different
    # linear solver from everything else.
    if want_cyipopt and have_ipopt and (report['ipopt'].get('ma27') is True
                                        or is_source_build(exe)):
        steps.extend(plan_relink(ipopt_build_prefix(exe)))
        want_cyipopt = False

    if not (want_ipopt or want_cyipopt):
        return with_pynumero_step(steps, args), source_root

    # The default: the open-source multi-solver build from source. Prebuilt
    # IPOPTs are MA27-only or MUMPS-only and never carry SPRAL.
    if want_ipopt and not getattr(args, 'prebuilt', False) and can_source_build():
        source_root = build_root(args)
        steps.append(plan_source_build_step(source_root, args))
        if want_cyipopt:
            steps.extend(plan_relink(os.path.join(source_root, 'ipopt', 'build')))
        return with_pynumero_step(steps, args), source_root

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
        return with_pynumero_step(steps, args), source_root

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
            # last resort: the source build anyway (the toolchain check
            # above said no, so the script will name what is missing)
            source_root = build_root(args)
            steps.append(plan_source_build_step(source_root, args))

    if want_cyipopt:
        if source_root:
            # point at what was just built; the source build isn't on PATH yet
            steps.extend(plan_relink(os.path.join(source_root, 'ipopt', 'build')))
        else:
            steps.append(Step(
                'install cyipopt (compiles from source against the IPOPT '
                'above; needs a C compiler and pkg-config)',
                command=[sys.executable, '-m', 'pip', 'install', 'cyipopt']))

    return with_pynumero_step(steps, args), source_root


def with_pynumero_step(steps, args):
    """Append the PyNumero ASL fetch to a plan, whatever else was decided:
    an env that already has ipopt+cyipopt may be missing only the ASL
    library."""
    if not args.skip_cyipopt and not pynumero_asl_available():
        steps.append(plan_pynumero_asl())
    return steps


def pynumero_asl_available():
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


def plan_pynumero_asl():
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


def build_root(args):
    """The source-build root (``--root``; ``--ma27-root`` is its old name)."""
    root = getattr(args, 'root', None) or args.ma27_root
    return os.path.abspath(os.path.expanduser(root))


def plan_ma27(sources, args, add=False, exe=None):
    """Steps for a source build carrying MA27 -- the full MUMPS + SPRAL +
    MA27 build, or (``add``) MA27 bolted onto the root an existing
    executable came from -- with cyipopt relinked against it."""
    if sys.platform.startswith('win'):
        raise RuntimeError(
            'building IPOPT from source needs bash and autotools, which this '
            'script cannot drive on Windows. Build under WSL, or use '
            '--prebuilt.')

    root = build_root(args)
    if add and exe and os.path.isfile(exe):
        # <root>/ipopt/build/bin/ipopt -> <root>, when it is one of ours
        prefix = ipopt_build_prefix(exe)
        if (os.path.basename(prefix) == 'build'
                and os.path.basename(os.path.dirname(prefix)) == 'ipopt'):
            root = os.path.dirname(os.path.dirname(prefix))
    build = os.path.join(root, 'ipopt', 'build')

    steps = [plan_source_build_step(root, args, ma27=sources, add=add)]

    if not args.skip_cyipopt:
        steps.extend(plan_relink(build))
        # a complete install in its own right, skipping the default planner
        # -- without this the documented quickstart built everything and
        # still couldn't evaluate a black box
        if not pynumero_asl_available():
            steps.append(plan_pynumero_asl())

    return steps


def plan_relink(build):
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
def finish_source_build(root, linear_solver=None):
    """Record the build and say what it carries. LCsolver records the build
    and prefers it, so no shell-profile export is needed."""
    build = os.path.join(root, 'ipopt', 'build')
    exe = os.path.join(build, 'bin', 'ipopt')

    print('\n' + '-' * 74)
    if not os.path.isfile(exe):
        print(f'warning: the build finished but no binary is at {exe}.')
        return

    from lcsolver import environment
    environment._PROBE_CACHE.clear()
    have = [s for s in ('ma27', 'mumps', 'spral')
            if linear_solver_available(s, exe)]
    default = environment.default_linear_solver('pyomo', exe)
    print(f'IPOPT is built: {exe}')
    print(f'linear solvers: {", ".join(have) or "none detected"}'
          f'{f"  (default: {default})" if default else ""}')

    try:
        state = record_ipopt(exe)
        print(f'\nRecorded in {state}. LCsolver will use this build from any '
              f'environment,\nwithout a PATH change -- it prefers a build '
              f'carrying more linear solvers\neven when conda puts conda\'s '
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
        epilog='With no options: installs cvxopt, builds IPOPT from source '
               'with MUMPS + SPRAL (SPRAL is then the default linear solver), '
               'and builds cyipopt against it. MA27 -- faster, free for '
               'academic use, not redistributable -- can be included '
               '(--ma27) or added later (--add-ma27).')
    parser.add_argument('--ma27', metavar='PATH', default=None,
                        help='include MA27 (sources at PATH) in the build '
                             '(download from '
                             'https://www.hsl.rl.ac.uk/download/MA27/1.0.0/)')
    parser.add_argument('--add-ma27', metavar='PATH', default=None,
                        help='add MA27 (sources at PATH) to the source build '
                             'already here, and relink cyipopt')
    parser.add_argument('--root', '--ma27-root', dest='ma27_root',
                        metavar='DIR', default='~/software',
                        help='where the source build lives (default: '
                             '~/software, i.e. ~/software/ipopt)')
    parser.add_argument('--prebuilt', action='store_true',
                        help='take a conda/Homebrew/apt IPOPT instead of the '
                             'source build (MA27-only or MUMPS-only; no SPRAL)')
    parser.add_argument('--no-spral', action='store_true',
                        help='leave SPRAL out of the source build')
    parser.add_argument('--no-mumps', action='store_true',
                        help='leave MUMPS out of the source build')
    parser.add_argument('--relink-cyipopt', action='store_true',
                        help='rebuild cyipopt against the existing IPOPT '
                             'build, and do nothing else')
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
    from lcsolver.environment import format_report
    print(format_report(report))
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

    conda = conda_environment()

    # ---- decide what to do ------------------------------------------------
    try:
        if args.relink_cyipopt:
            exe = ipopt_executable()
            if not exe or not os.path.isfile(exe):
                print('error: no ipopt executable to link against.', file=sys.stderr)
                return 1
            build = ipopt_build_prefix(exe)
            steps = plan_relink(build)
            built_root = built_solver = None
        elif args.ma27 or args.add_ma27:
            add = not args.ma27
            sources = os.path.abspath(os.path.expanduser(args.ma27
                                                         or args.add_ma27))
            if not looks_like_ma27(sources):
                print(f'error: no MA27 Fortran sources found at {sources}. '
                      f'Expected the extracted archive, containing ma27*.f '
                      f'either at the top level or under src/.', file=sys.stderr)
                return 1
            exe = report['ipopt']['executable']
            if add and not (exe and is_source_build(exe)):
                print('error: --add-ma27 needs a source build to add to, and '
                      'the IPOPT in use is not one. Run '
                      'lcsolver-install-solvers first (or --ma27 to build '
                      'everything at once).', file=sys.stderr)
                return 1
            steps = plan_ma27(sources, args, add=add, exe=exe)
            built_root, built_solver = build_root(args), 'MA27'
            if add:
                built_root = os.path.dirname(os.path.dirname(
                    ipopt_build_prefix(exe)))
        else:
            steps, built_root = plan_default(report, conda, args)
            built_solver = None
    except RuntimeError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1

    if not steps:
        print('Nothing to do -- everything this script installs is present.')
        suggest_ma27(report)
        return 0

    # ---- show it, then ask ------------------------------------------------
    print('Plan:')
    for step in steps:
        print(step.show())
    if (conda and not args.ma27 and not args.add_ma27
            and not args.relink_cyipopt and not built_root):
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
    if not confirm('\nProceed?', args.yes):
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
        finish_source_build(built_root, built_solver)
        return 0

    print('\nDone. Re-checking...\n')
    from lcsolver import environment
    environment._PROBE_CACHE.clear()
    report = check_solvers(probe=True)
    print(environment.format_report(report))
    suggest_ma27(report)
    return 0


def suggest_ma27(report):
    """The MA27 upsell, sized to what the build already has."""
    if report['ipopt']['ma27'] is not False:
        return
    exe = report['ipopt']['executable']
    flag = '--add-ma27' if exe and is_source_build(exe) else '--ma27'
    if report['ipopt'].get('spral'):
        print('\nThis build has no MA27; LCsolver defaults to SPRAL, which '
              'certifies the same problems. MA27 is about 3x faster on '
              'large decks (docs/linear_solvers.rst).')
    else:
        print('\nThis build carries only MUMPS, which falsely fails on real '
              'decks (docs/linear_solvers.rst). Re-run with a GCC toolchain, '
              'metis and hwloc installed to add SPRAL, or add MA27:')
    found = find_ma27_sources()
    if found:
        print(f'MA27 sources are already on this machine ({found}):\n'
              f'    lcsolver-install-solvers {flag} {found}')
    else:
        print('MA27 is free for academic use: register at '
              'https://www.hsl.rl.ac.uk/download/MA27/1.0.0/, extract, then:\n'
              f'    lcsolver-install-solvers {flag} <extracted-path>')


if __name__ == '__main__':
    raise SystemExit(main())
