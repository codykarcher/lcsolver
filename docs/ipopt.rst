Installing IPOPT
================

IPOPT is the one piece of LCsolver that ``pip`` cannot install for you, and the
one it needs most. It is the default convex backend --- ``solve()`` routes a
detected LP, QP, GP or SP to it unless told otherwise --- and it is the only
backend that can evaluate a black-box (grey-box) constraint. A working IPOPT is
therefore a working LCsolver.

This page covers getting one, and one choice inside it that matters more than
the rest: the sparse linear solver.


Two interfaces
--------------

LCsolver can reach IPOPT two ways, selected by ``method``:

``method='pyomo'``
    ``SolverFactory('ipopt')``, Pyomo's AMPL-based interface. Needs the
    ``ipopt`` **executable** on your ``PATH``. This is the faster route and the
    default.

``method='cyipopt'``
    ``pyomo.contrib.pynumero``, which calls the IPOPT **library** in-process.
    Required for black-box constraints, which the AMPL route cannot evaluate.

    It needs two things, and the second one surprises people. cyipopt itself,
    and Pyomo's compiled **PyNumero ASL library**, which is how PyNumero builds
    an NLP at all. That library ships with neither pyomo nor cyipopt: it is
    installed prebuilt from conda-forge's ``pynumero_libraries`` (linux-64,
    osx-64 and win-64 only) or compiled by ``pyomo build-extensions``, which
    is the only route on osx-arm64. Note that ``pyomo download-extensions``
    does *not* provide it --- that command fetches gjh and MC++ and reports
    success, which is a convincing way to believe the problem is fixed while
    nothing has changed. Without the library every solve on this route fails
    with ``Cannot load the PyNumero ASL interface (pynumero_ASL)`` --- a
    component most users have never heard of, named in an error that does
    not say how to get it.
    ``lcsolver-install-solvers`` fetches it, and ``lcsolver-check-solvers``
    reports it as missing rather than leaving you to decode that message.

``method='auto'`` (the default) prefers the Pyomo route and switches to cyipopt
when the model contains a black-box constraint. Installing both is the
comfortable position; installing neither leaves LCsolver able to build a formulation
and unable to solve it.


Why pip cannot do this for you
------------------------------

``pip install lcsolver`` installs everything except IPOPT, and no change to the
dependency metadata would fix that:

* There is no IPOPT executable on PyPI, under any name.
* cyipopt is published on PyPI as an **sdist only** --- no wheels, in any
  release to date. ``pip install cyipopt`` therefore *compiles*, against an
  IPOPT that must already be installed and discoverable by ``pkg-config``. On
  a machine with no IPOPT it fails, which is why it cannot be a dependency.

So IPOPT comes from conda-forge, from a system package manager, or from
source. ``lcsolver-install-solvers`` picks whichever of those applies, prints
the commands, and asks before running them.


The quick way, and what it costs you
------------------------------------

::

    lcsolver-install-solvers        # conda-forge if available, else brew/apt,
                                    # else a source build

or, from a clone::

    conda env create -f environment.yml
    conda activate lcsolver
    pip install -e .

Either gives you a working IPOPT in a minute or two, and it is the right first
move. Be aware of what it ships with.

IPOPT does not factorize anything itself. Every interior-point iteration solves
a symmetric indefinite KKT system, and that work is handed to a third-party
sparse linear solver. Which one you have is the single largest determinant of
how IPOPT behaves on a hard problem, and it is fixed at **build** time --- you
cannot pip-install a different one afterwards.

Prebuilt IPOPT binaries ship with **MUMPS**, because MUMPS is the only capable
option whose licence permits redistribution. Every conda-forge, apt and
Homebrew IPOPT you will encounter is a MUMPS build.


MUMPS is not the one you want
-----------------------------

MUMPS works, it is genuinely free, and for a model of a few hundred variables
you will not notice anything. Use it to get started. Do not settle on it.

The reason is not that MUMPS is a bad linear solver --- it is a good one, aimed
at a different problem. It is a general parallel multifrontal code built for
large *distributed* factorizations. IPOPT wants something else: many thousands
of factorizations of the *same sparsity pattern*, each one needing an accurate
inertia (the count of positive, negative and zero eigenvalues) so the algorithm
can tell whether its current KKT matrix is well-posed or needs regularizing.

That inertia request is where the difference shows. When the factorization
cannot deliver a reliable inertia, IPOPT falls back to adding regularization
to the Hessian and trying again. Each retry is a wasted factorization, and a
run that needs many of them converges slowly, stalls on ``Restoration Phase``,
or terminates on iteration count with a point that is nearly but not quite
feasible. On the geometric and signomial programs LCsolver generates --- which are
dense in couplings, badly scaled before presolve, and solved repeatedly inside
the SLCP and SIA loops --- this is the common failure mode, and it presents as
"the solver is flaky" rather than as "the linear algebra is the problem."

IPOPT's own default value for the ``linear_solver`` option is ``ma27``, not
``mumps``. That is the authors' recommendation stated in the source.


Use MA27
--------

MA27 is part of `HSL <https://www.hsl.rl.ac.uk/>`_, a Fortran library from the
STFC Rutherford Appleton Laboratory. It is **free for academic use** but cannot
be redistributed, which is precisely why no prebuilt IPOPT contains it and why
you have to build IPOPT yourself to get it.

It is a serial, symmetric-indefinite multifrontal code from the 1980s that has
been the reference linear solver for interior-point methods ever since. It is
small, it is fast on the sparsity patterns optimization produces, and --- the
part that matters --- it returns a trustworthy inertia, so IPOPT regularizes
only when it genuinely needs to.

Two caveats, so the recommendation is honest:

* MA27 is **serial**. On a very large problem a threaded solver (MA86, MA97, or
  MUMPS with a good BLAS) can win on wall-clock. MA27 is the right default up
  to problems considerably larger than anything in ``examples/``; it is not
  universally fastest.
* The gap is a matter of robustness and of scaling with problem size, not a
  fixed multiplier. On a small, well-scaled model the two are
  indistinguishable. Do not expect a speedup on the Hoburg UAV.

Getting it:

1. Register at https://www.hsl.rl.ac.uk/download/MA27/1.0.0/ and accept the
   academic licence.
2. Download and extract the archive anywhere.
3. Point the installer at it::

       lcsolver-install-solvers --ma27 /path/to/ma27-1.0.0

That one command is both the first-install route and the upgrade route: run on
a machine that already has the conda MUMPS build, it builds IPOPT against MA27,
relinks cyipopt against the new build, and prints the environment to export.
Nothing about the existing MUMPS install is removed --- see
`Which one am I actually using?`_ below, because with both present that
question stops being rhetorical.

If you would rather not register, stay on the MUMPS build and pass
``options={'linear_solver': 'mumps'}`` explicitly --- a MUMPS-only build will
otherwise reject IPOPT's ``ma27`` default with ``Invalid value "ma27" for
option linear_solver``.


Building IPOPT with MA27
------------------------

Under ``--ma27`` the installer runs ``lcsolver/scripts/install_ipopt.sh``,
which clones IPOPT and the ThirdParty helpers, builds ASL and HSL, and
configures IPOPT against both. It can also be run directly, which is unchanged
from how it has always worked::

    ./utilities/install_ipopt.sh ~/software

By default it expects the MA27 source at ``<root>/MA27/ma27-1.0.0``; set
``MA27_SRC`` to take it from anywhere else. Setting ``IPOPT_BUILD_MUMPS=1``
instead builds against MUMPS, which needs no manual download --- that is the
fallback the installer uses on a machine with no conda, apt or Homebrew.

It creates:

===============================  ==========================================
``<root>/ipopt/Ipopt``           clone of the IPOPT repository
``<root>/ipopt/ThirdParty-ASL``  clone of the ASL helper
``<root>/ipopt/ThirdParty-HSL``  clone of the HSL helper
``<root>/ipopt/ASL_build``       built ASL
``<root>/ipopt/HSL_build``       built HSL, containing MA27
``<root>/ipopt/build``           built IPOPT --- ``bin/ipopt`` lives here
===============================  ==========================================

(Under ``IPOPT_BUILD_MUMPS=1`` the HSL entries are replaced by
``ThirdParty-Mumps`` and ``MUMPS_build``.)

Nothing has to go in a shell profile afterwards. Two beliefs to the contrary
are worth correcting, because both used to be printed by this script:

**The library path is usually not needed.** libtool records absolute install
names for libcoinhsl and libcoinasl, so the loader finds them unaided; the
binary starts with ``DYLD_LIBRARY_PATH`` and ``LD_LIBRARY_PATH`` unset. Rather
than assert either way, the script now *tests* the binary it just built with
those variables cleared, and prints the exports only if they turn out to be
needed --- which happens if the build tree is moved after installation.

**The ``PATH`` entry is not needed either**, for LCsolver's purposes: the
installer records the build in ``~/.config/lcsolver/solvers.json`` and
LCsolver reads that. Add ``<root>/ipopt/build/bin`` to ``PATH`` if you want to
type ``ipopt`` in a terminal yourself; that is the only reason left.

``IPOPT_INCLUDE_DIR`` and ``IPOPT_LIBRARY_DIR`` are what ``pip install cyipopt``
compiles against. Export them *before* installing cyipopt if you want the
in-process route to use this build rather than a separate conda one --- or let
``lcsolver-install-solvers --relink-cyipopt`` do it, which is the same thing
with the paths filled in.


.. _Which one am I actually using?:

Which one am I actually using?
------------------------------

Once a machine has two IPOPTs --- a conda MUMPS one and a source MA27 one ---
this stops being obvious, and under plain ``PATH`` rules getting it wrong is
silent: ``conda activate`` prepends ``$CONDA_PREFIX/bin`` in every new shell, so
the conda binary wins and the build made specifically to get MA27 is used by
nothing. No error, just slower and less robust solves.

LCsolver does not follow plain ``PATH`` rules, for exactly that reason. It picks
in this order:

1. ``LCSOLVER_IPOPT_EXECUTABLE``, if set --- always, even if it points at
   nothing, because a broken pin is a mistake to report rather than route
   around.
2. Among everything else --- every ``ipopt`` on ``PATH``, plus any build
   ``lcsolver-install-solvers`` made --- **an MA27 build beats one without it**,
   whatever the order.

So a source build wins over a conda one, and a source build wins even when it is
on no ``PATH`` at all, because the installer recorded where it put it
(``~/.config/lcsolver/solvers.json``). Nothing needs to go in a shell profile.

The probe that decides this runs only when more than one candidate exists, and
is cached for the session.

To see the outcome, including which binary was passed over and why::

    lcsolver-check-solvers

It also reports whether cyipopt --- which links its own IPOPT and is unaffected
by ``PATH`` entirely --- agrees with the executable.

To override::

    export LCSOLVER_IPOPT_EXECUTABLE="<root>/ipopt/build/bin/ipopt"   # force one
    export LCSOLVER_IPOPT_AUTOSELECT=0                                # strict PATH order

Both apply on every route that drives the AMPL interface, including the SLCP and
SIA loops, which run the most solves and care about the linear solver most.


Checking by hand
----------------

::

    which ipopt
    ipopt --version

To find out which linear solvers were actually compiled in, ask for one that
does not exist --- IPOPT answers with the list::

    $ ipopt --print-options 2>/dev/null | grep -A12 'linear_solver'

A more direct test is to solve something and name the solver::

    from lcsolver.solvers.ipopt import ipopt_solve
    res = ipopt_solve(f, options={'linear_solver': 'ma27'})

``Invalid value "ma27" for option linear_solver`` means IPOPT was built without
HSL. The same message for ``mumps`` means it was built without MUMPS --- a
source build against HSL only, which is a perfectly good place to be.
