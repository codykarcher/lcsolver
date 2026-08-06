#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Tests for solver discovery and the install bootstrap.

None of these install anything or run a solver. What they check is the part
that decides *what would be installed* and *which IPOPT would be used* --- the
logic that, when wrong, produces a machine that solves with the wrong linear
solver and says nothing about it.
"""

import os
import stat
import sys

import pytest

from lcsolver import environment, install


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    """Keep these tests off the developer's own machine state.

    Two things would otherwise leak: the resolution cache, which would carry a
    PATH from one test into the next, and the recorded-build file in the real
    ``~/.config``, which the tests must neither read nor write.
    """
    monkeypatch.setattr(environment, 'STATE_FILE',
                        str(tmp_path / 'state' / 'solvers.json'))
    monkeypatch.delenv(environment.IPOPT_EXECUTABLE_ENV, raising=False)
    monkeypatch.delenv(environment.AUTOSELECT_ENV, raising=False)

    # Whether this machine happens to have Pyomo's PyNumero ASL library is not
    # what any of the planner tests are about, but it changes the plan by one
    # step -- so left alone, they pass on a developer's machine (which has it)
    # and fail on a fresh runner (which does not). Pinned to present here; the
    # two tests that are about the ASL library set it themselves.
    monkeypatch.setattr(install, '_pynumero_asl_available', lambda: True)

    environment._forget_resolution()
    yield
    environment._forget_resolution()


# --------------------------------------------------------------------------
# choosing an executable
# --------------------------------------------------------------------------
def _fake_ipopt(directory, name='ipopt'):
    """An executable file that is not really IPOPT. Never run, only found."""
    path = os.path.join(directory, name)
    with open(path, 'w') as handle:
        handle.write('#!/bin/sh\nexit 0\n')
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP
             | stat.S_IXOTH)
    return path


def test_pin_beats_path(tmp_path, monkeypatch):
    """The whole point of the pin: it wins over PATH, which conda controls."""
    on_path = tmp_path / 'path_bin'
    pinned = tmp_path / 'pinned_bin'
    on_path.mkdir()
    pinned.mkdir()
    _fake_ipopt(str(on_path))
    pin = _fake_ipopt(str(pinned))

    monkeypatch.setenv('PATH', str(on_path))
    assert environment.ipopt_executable() == os.path.join(str(on_path), 'ipopt')

    monkeypatch.setenv(environment.IPOPT_EXECUTABLE_ENV, pin)
    environment._forget_resolution()
    assert environment.ipopt_executable() == pin


def test_pin_to_a_missing_file_is_reported_not_ignored(tmp_path, monkeypatch):
    """Silently falling back to PATH would hide the user's mistake."""
    monkeypatch.setenv(environment.IPOPT_EXECUTABLE_ENV,
                       str(tmp_path / 'does_not_exist'))
    assert environment.ipopt_executable() == str(tmp_path / 'does_not_exist')

    report = environment.check_solvers(probe=False)
    assert report['ipopt']['pinned'] is True
    assert any('does not exist' in w for w in report['warnings'])


def test_every_ipopt_on_path_is_found_in_order(tmp_path, monkeypatch):
    """Shadow detection needs the losers, not just the winner."""
    first = tmp_path / 'a'
    second = tmp_path / 'b'
    first.mkdir()
    second.mkdir()
    a = _fake_ipopt(str(first))
    b = _fake_ipopt(str(second))

    monkeypatch.setenv('PATH', os.pathsep.join([str(first), str(second)]))
    found = environment.ipopt_executables_on_path()
    assert found == [a, b]


def test_no_ipopt_anywhere_is_a_warning_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv('PATH', str(tmp_path))
    monkeypatch.delenv(environment.IPOPT_EXECUTABLE_ENV, raising=False)

    report = environment.check_solvers(probe=False)
    assert report['ipopt']['executable'] is None
    assert any('no ipopt executable' in w for w in report['warnings'])


def test_report_renders_without_probing(tmp_path, monkeypatch):
    """`--no-probe` must produce a readable report, not a half-filled one."""
    monkeypatch.setenv('PATH', str(tmp_path))
    text = environment._fmt(environment.check_solvers(probe=False))
    assert 'LCsolver solver environment' in text
    assert 'ipopt' in text


def test_quiet_restores_the_file_descriptors():
    """The probe silencer redirects fd 1 and 2 at the OS level to catch
    IPOPT's C++ output. If it failed to put them back, every later print in
    the process would vanish into /dev/null."""
    def identity(fd):
        info = os.fstat(fd)
        return (info.st_dev, info.st_ino)

    before = (identity(1), identity(2))
    with environment._quiet():
        print('this should not appear')
    assert (identity(1), identity(2)) == before

    # and still writable
    os.write(1, b'')


def test_an_ma27_build_wins_even_when_it_is_last_on_path(tmp_path, monkeypatch):
    """The failure this removes: `conda activate` prepends $CONDA_PREFIX/bin,
    so a prebuilt MUMPS ipopt shadows the build someone made to get MA27, and
    nothing says a word about it."""
    conda_like = tmp_path / 'conda_bin'
    source_like = tmp_path / 'source_bin'
    conda_like.mkdir()
    source_like.mkdir()
    mumps = _fake_ipopt(str(conda_like))
    ma27 = _fake_ipopt(str(source_like))

    monkeypatch.setenv('PATH', os.pathsep.join([str(conda_like),
                                                str(source_like)]))
    monkeypatch.setattr(environment, 'linear_solver_available',
                        lambda name, exe=None: name == 'ma27' and exe == ma27)
    environment._forget_resolution()

    chosen, reason = environment.ipopt_choice()
    assert chosen == ma27
    assert 'no MA27' in reason and mumps in reason


def test_strict_path_order_is_still_available(tmp_path, monkeypatch):
    """Auto-selection is a default, not a policy. Someone who wants exactly
    what their PATH says must be able to have it."""
    first = tmp_path / 'a'
    second = tmp_path / 'b'
    first.mkdir()
    second.mkdir()
    a = _fake_ipopt(str(first))
    ma27 = _fake_ipopt(str(second))

    monkeypatch.setenv('PATH', os.pathsep.join([str(first), str(second)]))
    monkeypatch.setenv(environment.AUTOSELECT_ENV, '0')
    monkeypatch.setattr(environment, 'linear_solver_available',
                        lambda name, exe=None: exe == ma27)
    environment._forget_resolution()

    chosen, reason = environment.ipopt_choice()
    assert chosen == a
    assert 'first on PATH' in reason


def test_a_recorded_build_is_used_with_nothing_on_path(tmp_path, monkeypatch):
    """A source build lands somewhere no PATH points at. Recording it is what
    replaces 'paste this export into your shell profile'."""
    build = tmp_path / 'software' / 'ipopt' / 'build' / 'bin'
    build.mkdir(parents=True)
    exe = _fake_ipopt(str(build))

    monkeypatch.setenv('PATH', str(tmp_path / 'nothing_here'))
    environment._forget_resolution()
    assert environment.ipopt_executable() is None

    environment.record_ipopt(exe)
    environment._forget_resolution()
    chosen, reason = environment.ipopt_choice()
    assert chosen == exe
    assert 'recorded' in reason


def test_a_recorded_build_that_was_deleted_is_ignored(tmp_path, monkeypatch):
    """Stale state must not turn into a pin at a path that no longer exists."""
    monkeypatch.setenv('PATH', str(tmp_path / 'nothing_here'))
    environment.record_ipopt(str(tmp_path / 'gone' / 'ipopt'))
    environment._forget_resolution()

    assert environment.recorded_ipopt() is None
    assert environment.ipopt_executable() is None


def test_the_report_says_why_this_binary(tmp_path, monkeypatch):
    """With several installed, 'which one and why' is the whole question."""
    directory = tmp_path / 'bin'
    directory.mkdir()
    _fake_ipopt(str(directory))
    monkeypatch.setenv('PATH', str(directory))
    environment._forget_resolution()

    report = environment.check_solvers(probe=False)
    assert report['ipopt']['reason']
    assert report['ipopt']['reason'] in environment._fmt(report)


# --------------------------------------------------------------------------
# recognizing MA27 sources
# --------------------------------------------------------------------------
@pytest.mark.parametrize('layout', ['top_level', 'src', 'ma27'])
def test_ma27_sources_recognized_in_each_layout(tmp_path, layout):
    """The archive has been shipped with more than one directory shape."""
    root = tmp_path / 'ma27-1.0.0'
    target = root if layout == 'top_level' else root / layout
    target.mkdir(parents=True)
    (target / 'ma27ad.f').write_text('      SUBROUTINE MA27AD\n')

    assert install._looks_like_ma27(str(root))


def test_a_directory_without_ma27_fortran_is_rejected(tmp_path):
    """The failure this prevents: HSL builds a library with no solver in it,
    and IPOPT only says so an hour later, at run time."""
    root = tmp_path / 'not_ma27'
    (root / 'src').mkdir(parents=True)
    (root / 'src' / 'readme.txt').write_text('nothing useful')

    assert not install._looks_like_ma27(str(root))
    assert not install._looks_like_ma27(str(tmp_path / 'absent'))


def test_ma27_autodiscovery_only_accepts_real_sources(tmp_path, monkeypatch):
    monkeypatch.setenv('MA27_SOURCE', str(tmp_path / 'empty'))
    (tmp_path / 'empty').mkdir()
    monkeypatch.setattr(install, '_MA27_SEARCH', ('$MA27_SOURCE',))
    assert install._find_ma27_sources() is None

    (tmp_path / 'empty' / 'ma27ad.f').write_text('      SUBROUTINE MA27AD\n')
    assert install._find_ma27_sources() == str(tmp_path / 'empty')


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------
class _Args:
    """Stand-in for the parsed argparse namespace."""

    def __init__(self, **kwargs):
        self.skip_ipopt = False
        self.skip_cyipopt = False
        self.skip_cvxopt = False
        self.ma27 = None
        self.ma27_root = '~/software'
        self.__dict__.update(kwargs)


def _report(cvxopt=True, ipopt=None, cyipopt=True):
    return {
        'cvxopt': {'available': cvxopt, 'version': '1.3.3'},
        'ipopt': {'executable': ipopt, 'pinned': False, 'version': None,
                  'ma27': None, 'mumps': None, 'shadowed_by': None,
                  'all_on_path': []},
        'cyipopt': {'available': cyipopt, 'version': '1.7.0',
                    'pynumero_asl': True, 'ma27': None, 'mumps': None},
        'probed': True,
        'warnings': [],
    }


def test_a_complete_install_plans_nothing(tmp_path):
    exe = _fake_ipopt(str(tmp_path))
    steps, source_root = install._plan_default(
        _report(ipopt=exe), conda=None, args=_Args())
    assert steps == []
    assert source_root is None


def test_conda_is_preferred_and_names_the_environment_it_would_change():
    conda = {'exe': 'conda', 'prefix': '/opt/envs/lcsolver',
             'name': 'lcsolver', 'activated': True}
    steps, source_root = install._plan_default(
        _report(ipopt=None, cyipopt=False), conda=conda, args=_Args())

    assert len(steps) == 1
    assert 'ipopt' in steps[0].command and 'cyipopt' in steps[0].command
    # The prefix has to be visible before the user approves anything.
    assert '/opt/envs/lcsolver' in steps[0].description
    assert source_root is None


def test_an_ma27_executable_makes_cyipopt_link_to_it_not_to_conda(tmp_path):
    """The trap this closes: `conda install cyipopt` drags conda's MUMPS IPOPT
    along and links against it, so the executable route runs MA27 while
    black-box models -- which have no choice but cyipopt -- run MUMPS."""
    build = tmp_path / 'ipopt' / 'build'
    (build / 'bin').mkdir(parents=True)
    exe = _fake_ipopt(str(build / 'bin'))
    conda = {'exe': 'conda', 'prefix': '/opt/envs/x', 'name': 'x',
             'activated': True}

    report = _report(ipopt=exe, cyipopt=False)
    report['ipopt']['ma27'] = True
    steps, _ = install._plan_default(report, conda=conda, args=_Args())

    assert len(steps) == 1
    assert '--no-binary' in steps[0].command          # built, not downloaded
    assert 'conda' not in steps[0].command
    assert steps[0].env['IPOPT_LIBRARY_DIR'] == str(build / 'lib')


def test_a_mumps_executable_does_not_trigger_a_relink(tmp_path):
    """Relinking against a MUMPS build would gain nothing and cost a compile."""
    exe = _fake_ipopt(str(tmp_path))
    conda = {'exe': 'conda', 'prefix': '/opt/envs/x', 'name': 'x',
             'activated': True}

    report = _report(ipopt=exe, cyipopt=False)
    report['ipopt']['ma27'] = False
    steps, _ = install._plan_default(report, conda=conda, args=_Args())

    assert len(steps) == 1
    assert steps[0].command[0] == 'conda'
    assert 'cyipopt' in steps[0].command


def test_missing_cvxopt_is_planned_even_though_it_is_a_hard_dependency():
    """It is declared, so this only happens after --no-deps or a manual
    uninstall -- but that is exactly when someone runs this script."""
    steps, _ = install._plan_default(
        _report(cvxopt=False, ipopt='/usr/bin/ipopt'), conda=None,
        args=_Args())
    assert any('cvxopt' in step.command for step in steps if step.command)


def test_the_pynumero_asl_library_is_fetched_when_missing(tmp_path, monkeypatch):
    """Installing cyipopt is not enough for the in-process route.

    Pyomo builds the NLP through PyNumero, whose compiled ASL library ships
    with neither pyomo nor cyipopt. Without it every black-box solve dies on
    "Cannot load the PyNumero ASL interface" -- which is what a conda
    environment built from environment.yml does, and it is the case that
    reaches the early return in the planner, so it is the one worth pinning.
    """
    exe = _fake_ipopt(str(tmp_path))
    monkeypatch.setattr(install, '_pynumero_asl_available', lambda: False)

    report = _report(ipopt=exe)          # ipopt and cyipopt both already here
    steps, _ = install._plan_default(report, conda=None, args=_Args())

    assert len(steps) == 1
    joined = ' '.join(steps[0].command)
    assert 'build_pynumero' in joined

    # Two routes that look right and are not. `download-extensions` fetches
    # gjh and MC++, reports success and leaves the library exactly as absent
    # as it was. conda-forge's `pynumero_libraries` does contain it, but its
    # newest build wants Python <= 3.8, so conda cannot solve for it at all on
    # a current interpreter.
    assert 'download-extensions' not in joined
    assert 'pynumero_libraries' not in joined

    # And it must not sink the whole install: an environment that cannot
    # compile it still has a working executable route.
    assert steps[0].optional is True


def test_the_asl_library_is_not_refetched_when_present(tmp_path, monkeypatch):
    exe = _fake_ipopt(str(tmp_path))
    monkeypatch.setattr(install, '_pynumero_asl_available', lambda: True)

    steps, _ = install._plan_default(_report(ipopt=exe), conda=None,
                                     args=_Args())
    assert steps == []


def test_a_missing_asl_library_is_reported_not_silent(monkeypatch):
    """It has to appear in the report too: the error it causes at solve time
    names a component the user has never heard of."""
    monkeypatch.setattr(environment, '_pynumero_asl_available', lambda: False)
    report = environment.check_solvers(probe=False)

    if report['cyipopt']['available']:   # nothing to say if cyipopt is absent
        assert report['cyipopt']['pynumero_asl'] is False
        assert any('PyNumero ASL' in w for w in report['warnings'])
        assert 'MISSING' in environment._fmt(report)


def test_relink_targets_the_build_the_executable_came_from(tmp_path):
    build = tmp_path / 'ipopt' / 'build'
    (build / 'lib' / 'pkgconfig').mkdir(parents=True)

    steps = install._plan_relink(str(build))
    assert len(steps) == 1
    step = steps[0]
    assert '--no-binary' in step.command
    assert step.env['IPOPT_INCLUDE_DIR'] == str(build / 'include' / 'coin-or')
    assert step.env['IPOPT_LIBRARY_DIR'] == str(build / 'lib')
    assert str(build / 'lib' / 'pkgconfig') in step.env['PKG_CONFIG_PATH']


@pytest.mark.skipif(sys.platform.startswith('win'),
                    reason='the MA27 build is not supported on Windows')
def test_ma27_plan_passes_the_sources_through_without_copying(tmp_path):
    sources = tmp_path / 'ma27-1.0.0'
    sources.mkdir()
    (sources / 'ma27ad.f').write_text('      SUBROUTINE MA27AD\n')

    steps = install._plan_ma27(str(sources), _Args(ma27_root=str(tmp_path)))
    build_step = steps[0]
    assert build_step.env['MA27_SRC'] == str(sources)
    assert build_step.command[0] == 'bash'
    assert build_step.command[1].endswith('install_ipopt.sh')
    # and the relink follows, or black-box models stay on the old library
    assert any('cyipopt' in step.command for step in steps[1:] if step.command)


def test_the_build_script_ships_inside_the_package():
    """It lives in the package rather than utilities/ so that it survives a
    wheel install, which is the only way --ma27 works for anyone else."""
    path = install._packaged_script('install_ipopt.sh')
    assert os.path.isfile(path)
    text = open(path).read()
    assert 'MA27_SRC' in text
    assert 'IPOPT_BUILD_MUMPS' in text
