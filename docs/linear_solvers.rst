Linear solvers for IPOPT
========================

Every IPOPT iteration factorizes one KKT system, and which sparse solver
does it decides robustness on LCsolver's problem class. This page records
what IPOPT can be built on, what we measured, how to build a multi-solver
binary, and where the effort to reduce MA27 dependence should go next.

Selecting a solver is one argument::

    lcsolver.solve(f, linear_solver='ma27')     # or 'mumps', 'pardiso', ...

validated by name before any backend runs, probed against the IPOPT build
actually in use (a missing solver raises ``SolverUnavailable`` naming what
IS available), and threaded through every route: raw NLP, log-space GP,
and the SIA/PCCP sub-problem loops. Results record which solver ran.

What upstream IPOPT supports
----------------------------

===========  =========================  =====================================
solver       license                    notes
===========  =========================  =====================================
MA27         HSL (archive, free)        compiled in; IPOPT's classic default
MA57/77/86/  HSL (free academic /       loadable at runtime from a user
MA97         commercial)                library, no rebuild needed
MUMPS        CeCILL-C (free)            the solver most binary distributions
                                        historically shipped
Pardiso      Panua (commercial) or      loadable at runtime
             MKL (free, x86 only)
SPRAL        BSD (open source)          MA97-class, GPU-capable; the modern
                                        no-license-friction option
WSMP         IBM (commercial)           rarely relevant here
===========  =========================  =====================================

UMFPACK (SuiteSparse -- the "oofpack/uufpak" from the 2026-09 call) and
PETSc are **not** supported upstream: using either means maintaining a fork
implementing ``Ipopt::SparseSymLinearSolverInterface`` (~1-2k lines of C++;
the in-tree HSL and MUMPS interfaces are the reference implementations).
Recommendation: do not fork until SPRAL has been evaluated -- it covers the
license-freedom goal inside upstream IPOPT.

Distribution reality (measured 2026-09-15, macOS arm64)
-------------------------------------------------------

Both conda-forge (``ipopt`` 3.14.20) and Homebrew now build IPOPT with
**MA27 compiled in and no MUMPS at all** -- the CoinHSL archive licensing
change reached the packagers. The folklore that "a binary install means
MUMPS" is stale on this platform; a plain ``brew install ipopt`` or conda
install is already MA27.

One hazard found while measuring: a loader path pinned to one IPOPT install
(``DYLD_LIBRARY_PATH`` on macOS, ``LD_LIBRARY_PATH`` on Linux) shadows the
shared library of *every other* install by name -- three different
MUMPS-capable binaries all probed as "MA27-only" because they silently
loaded the shadowing build's ``libipopt``. LCsolver now puts each
executable's own ``lib`` directory first when probing or solving
(``lcsolver.environment.ensure_own_libs_first``).

What we measured
----------------

All runs: Ipopt 3.14.20, one dual-solver source build (MA27 + MUMPS 5.9.1),
macOS arm64, 2026-09-15.

* **Every model in examples/** (GPs, an LP, a QP, an SP) plus an
  ill-conditioned synthetic family: identical status and objective on both
  solvers, all sub-second. Small, clean log-space models do not stress the
  factorization.
* **737-800 / TASOPT deck** (lcjetliner, 1298 variables, SIA): both
  certify in 39 iterations; MA27 110 s, MUMPS 122 s.
* **SPaircraft D8.2** (York et al., AIAA J.; 1172 variables, SIA):
  **MUMPS falsely declares the iteration-2 sub-problem infeasible**; the
  run stops uncertified at 25,102 lbf, 17 % above the answer. MA27
  certifies at 179 iterations / 35 s / 21,384.0 lbf. The failing
  sub-problem is bundled at ``examples/data/d8_sia_subproblem.nl`` and
  replayed by ``examples/linear_solver_switch.py`` and the test suite.

Verdict: **MA27 is the right default for this problem class**; MUMPS is
acceptable for small models and as a fallback, and the D8 capture is the
regression sentinel that says when that assessment should be revisited.

Re-running the assessment::

    python utilities/linear_solver_benchmark.py \
        --executable /path/to/dual/ipopt --linear-solvers ma27,mumps

(one subprocess per model/solver cell, hard per-cell timeout).

Building a multi-solver IPOPT
-----------------------------

The measured binary carries MA27 and MUMPS side by side, switchable per
solve. Recipe (macOS; Linux is the same modulo the loader variable)::

    cd ~/software/ipopt
    git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git
    cd ThirdParty-Mumps
    ./get.Mumps
    ./configure --prefix=$PWD/../MUMPS_build && make -j4 && make install

    cd .. && git clone --branch stable/3.14 \
        https://github.com/coin-or/Ipopt.git Ipopt-src
    mkdir build-dir && cd build-dir
    ../Ipopt-src/configure --prefix=$PWD/../build-mumps \
        --with-hsl-cflags="-I$PWD/../HSL_build/include/coin-or/hsl" \
        --with-hsl-lflags="-L$PWD/../HSL_build/lib -lcoinhsl" \
        --with-mumps-cflags="-I$PWD/../MUMPS_build/include/coin-or/mumps" \
        --with-mumps-lflags="-L$PWD/../MUMPS_build/lib -lcoinmumps" \
        --with-asl-cflags="-I$PWD/../ASL_build/include/coin-or/asl" \
        --with-asl-lflags="-L$PWD/../ASL_build/lib -lcoinasl" \
        --disable-java
    make -j4 && make install

(The HSL and ASL third-party builds are the ones
``utilities/install_ipopt.sh`` already produces.)

Next steps for reducing MA27 dependence
---------------------------------------

1. **SPRAL** (first): build ThirdParty-Metis, then SPRAL with OpenMP, then
   IPOPT ``--with-spral``; run the benchmark battery plus the D8 capture
   against it. If SPRAL certifies the D8, the license-freedom goal is met
   with zero fork burden.
2. **Runtime-loaded HSL** (MA57/MA86/MA97): the same coinhsl library
   already built here exposes them; benchmark MA97 on the big SPs, since it
   is the modern HSL answer for indefinite systems.
3. **Pardiso via Panua** on Apple silicon if a license materialises;
   MKL-Pardiso only on x86 machines.
4. **UMFPACK/PETSc fork**: only if 1-3 all disappoint; budget ~1-2k lines
   of C++ against ``SparseSymLinearSolverInterface`` plus permanent rebase
   cost.
