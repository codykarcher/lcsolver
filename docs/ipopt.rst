Installing IPOPT
================

IPOPT is not an optional extra in the way ``cvxopt`` is. It is the default
convex backend --- ``solve()`` routes a detected LP, QP, GP or SP to it unless
told otherwise --- and it is the only backend that can evaluate a black-box
(grey-box) constraint. A working IPOPT is therefore a working LCsolver.

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
    Needs ``pip install cyipopt``. Required for black-box constraints, which
    the AMPL route cannot evaluate.

``method='auto'`` (the default) prefers the Pyomo route and switches to cyipopt
when the model contains a black-box constraint. Installing both is the
comfortable position; installing neither leaves LCsolver able to build a formulation
and unable to solve it.


The quick way, and what it costs you
------------------------------------

::

    conda install -c conda-forge ipopt        # executable and library
    pip install cyipopt                       # the in-process route

That gives you a working IPOPT in a minute, and it is the right first move.
Be aware of what it ships with.

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
2. Download and extract the archive, so that you have a directory
   ``<root>/MA27/ma27-1.0.0``.
3. Build IPOPT against it with ``tools/install_ipopt.sh`` (next section).

If you would rather not register, stay on the conda-forge MUMPS build and pass
``options={'linear_solver': 'mumps'}`` explicitly --- a MUMPS-only build will
otherwise reject IPOPT's ``ma27`` default with ``Invalid value "ma27" for
option linear_solver``.


Building IPOPT with MA27
------------------------

``tools/install_ipopt.sh`` clones IPOPT and the ThirdParty helpers, builds ASL
and HSL, and configures IPOPT against both::

    ./tools/install_ipopt.sh ~/software

It expects the MA27 source at ``<root>/MA27/ma27-1.0.0`` as above, and creates:

===============================  ==========================================
``<root>/ipopt/Ipopt``           clone of the IPOPT repository
``<root>/ipopt/ThirdParty-ASL``  clone of the ASL helper
``<root>/ipopt/ThirdParty-HSL``  clone of the HSL helper
``<root>/ipopt/ASL_build``       built ASL
``<root>/ipopt/HSL_build``       built HSL, containing MA27
``<root>/ipopt/build``           built IPOPT --- ``bin/ipopt`` lives here
===============================  ==========================================

The script prints the environment settings to add to your shell profile
(``~/.zshenv``, ``~/.bashrc``). They are needed: the ``ipopt`` binary finds
libcoinhsl and libcoinasl at run time through the library path, and omitting
them produces a binary that exists and will not start.

::

    export PATH="<root>/ipopt/build/bin:$PATH"
    export DYLD_LIBRARY_PATH="<root>/ipopt/build/lib:$DYLD_LIBRARY_PATH"
    export DYLD_LIBRARY_PATH="<root>/ipopt/ASL_build/lib:$DYLD_LIBRARY_PATH"
    export DYLD_LIBRARY_PATH="<root>/ipopt/HSL_build/lib:$DYLD_LIBRARY_PATH"
    export IPOPT_INCLUDE_DIR="<root>/ipopt/build/include/coin-or"
    export IPOPT_LIBRARY_DIR="<root>/ipopt/build/lib"

On Linux use ``LD_LIBRARY_PATH`` in place of ``DYLD_LIBRARY_PATH``.

``IPOPT_INCLUDE_DIR`` and ``IPOPT_LIBRARY_DIR`` are what ``pip install cyipopt``
compiles against, so export them *before* installing cyipopt if you want the
in-process route to use this build rather than a separate conda one.


Checking what you have
----------------------

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
