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

All runs: Ipopt 3.14.20, one triple-solver source build (MA27 + MUMPS 5.9.1
+ SPRAL v2023.03.29), macOS arm64, 2026-09-15/16.

* **Every model in examples/** (GPs, an LP, a QP, an SP) plus an
  ill-conditioned synthetic family: identical status and objective on all
  three solvers, all sub-second (SPRAL ~2x the wall time of MA27 at this
  size -- OpenMP overhead, not arithmetic).
* **The D8 sentinel sub-problem** (``examples/data/d8_sia_subproblem.nl``,
  SIA tolerances): MA27 optimal in 0.13 s, SPRAL optimal in 0.40 s,
  **MUMPS falsely declares it infeasible**. Build-specific: the linux
  conda-forge MUMPS solves this capture, so the failure belongs to the
  macOS arm64 MUMPS 5.9.1 + openblas build -- which sharpens the lesson
  (the same named solver is only as good as its build) rather than
  blunting it.
* **SPaircraft D8.2 full deck** (York et al., AIAA J.; 1172 variables,
  SIA): MA27 certifies at 179 iterations / 35 s / 21,384.0 lbf; **SPRAL
  certifies the same optimum at 149 iterations / 103 s**; MUMPS stops
  uncertified at iteration 2, 17 % above the answer.
* **737-800 / TASOPT deck** (lcjetliner, 1298 variables, SIA): MA27
  39 iterations / 110 s; MUMPS 39 / 122 s; SPRAL 23 iterations / 157 s.
  All three certify.

Verdict: **MA27 is the right default** (fastest, most robust). **SPRAL is
a working, license-free alternative**: it certifies everything MA27 does
-- including the sentinel MUMPS fails -- in consistently FEWER SIA
iterations, at 1.5-3x the wall time on one machine (its OpenMP
parallelism should close that gap on larger problems and more cores).
The MA27-dependence concern is answered without forking anything. MUMPS
remains acceptable only for small models. The D8 capture is the
regression sentinel that says when this assessment should be revisited.

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

Building SPRAL into IPOPT
-------------------------

Done and measured (see above). The recipe on macOS arm64::

    brew install metis hwloc autoconf automake libtool  # gcc for gfortran
    cd ~/software/ipopt
    git clone --depth 1 --branch v2023.03.29 \
        https://github.com/ralna/spral.git spral-src
    cd spral-src && ./autogen.sh
    CC=gcc-16 CXX=g++-16 FC=gfortran ./configure \
        --prefix=$PWD/../SPRAL_build \
        --with-blas="-L/opt/homebrew/opt/openblas/lib -lopenblas" \
        --with-lapack="-L/opt/homebrew/opt/openblas/lib -lopenblas" \
        --with-metis="-L/opt/homebrew/lib -lmetis" \
        --with-metis-inc-dir=/opt/homebrew/include
    make -j4 && make install

then add to the IPOPT configure (alongside the HSL/MUMPS flags above),
building IPOPT with the SAME gcc toolchain::

    CC=gcc-16 CXX=g++-16 FC=gfortran \
    CXXFLAGS="-O2 -fno-devirtualize-speculatively" \
    ../Ipopt-src/configure ... \
      --with-spral-cflags="-I$PWD/../SPRAL_build/include" \
      --with-spral-lflags="-L$PWD/../SPRAL_build/lib -lspral \
          -L/opt/homebrew/opt/openblas/lib -lopenblas \
          -L/opt/homebrew/lib -lmetis -lhwloc \
          -L$(dirname $(gcc-16 -print-file-name=libgomp.dylib)) \
          -lgomp -lgfortran"

Two gotchas, both hit and solved here: (1) SPRAL's C++ objects need the
same C++ runtime as IPOPT's -- build both with gcc, or the link dies on
libstdc++ symbols; (2) ``-fno-devirtualize-speculatively`` is REQUIRED
with gcc: its speculative devirtualization emits references to the
vtables of dependency-detector classes IPOPT declares but does not
compile (Ma28), and the link fails on a symbol nothing actually uses.
At runtime SPRAL needs ``OMP_CANCELLATION=TRUE`` and
``OMP_PROC_BIND=TRUE``; LCsolver sets both automatically whenever
``linear_solver='spral'`` is requested.

Remaining avenues
-----------------

1. **Runtime-loaded HSL** (MA57/MA86/MA97): machinery is in place --
   ``solve(f, linear_solver='ma97', linear_solver_library='/path/to/
   libcoinhsl.dylib')`` -- but the coinhsl here is the MA27-only archive,
   so benchmarking MA97 waits on a full CoinHSL tarball.
2. **Pardiso**: same machinery (``linear_solver_library`` maps to
   ``pardisolib``), untestable on this machine (Panua is commercial,
   MKL is x86-only) -- ready for a licensed box.
3. **UMFPACK/PETSc fork**: moot unless SPRAL disappoints at scale;
   budget ~1-2k lines of C++ against ``SparseSymLinearSolverInterface``
   plus permanent rebase cost.
