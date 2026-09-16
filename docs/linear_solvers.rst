Linear solvers for IPOPT
========================

Every IPOPT iteration factorizes one KKT system, and which sparse solver
does it decides robustness on LCsolver's problem class. This page records
what IPOPT can be built on, what we measured, how to build a multi-solver
binary, and where the effort to reduce MA27 dependence should go next.

Selecting a solver is one argument::

    lcsolver.solve(f, linear_solver='ma27')     # or 'mumps', 'spral', ...

validated by name before any backend runs, probed against the IPOPT build
actually in use (a missing solver raises ``SolverUnavailable`` naming what
IS available), and threaded through every route: raw NLP, log-space GP,
and the SIA/PCCP sub-problem loops. Results record which solver ran (the
summary's Report section says ``[linear solver: ...]``).

Told nothing, LCsolver picks explicitly, in the order **MA27, SPRAL,
MUMPS** -- the first the build carries. The default install
(``lcsolver-install-solvers``) builds MUMPS + SPRAL, so SPRAL is the
default there; adding MA27 (``--ma27`` at build time, ``--add-ma27`` after)
makes MA27 the default. In-process cyipopt never picks SPRAL (see the
OpenMP note below).

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
  certifies the same optimum** (179 iterations / 86 s with LCsolver's
  MC64-scaling default; 149 / 103 s on IPOPT's stock SPRAL settings, with
  two sub-problems crashing the executable along the way -- see the crash
  sentinel below); MUMPS stops uncertified at iteration 2, 17 % above the
  answer.
* **737-800 / TASOPT deck** (lcjetliner, 1298 variables, SIA): MA27
  39 iterations / 110 s; MUMPS 39 / 122 s; SPRAL 23 iterations / 157 s.
  All three certify.
* **Every SIA sub-problem of both decks, replayed standalone** (the
  batch comparison; one IPOPT run per file per solver):

  ==========  =======  =========  ===========  ============  =========
  deck        files    MA27       SPRAL+mc64   SPRAL stock   MA27 time
  ==========  =======  =========  ===========  ============  =========
  D8 (1e-12)  352      352 solve  352 solve    3 crash,      17 s vs
                       (4 accept) (19 accept)  1 false inf.  59 s SPRAL
  b737 (1e-9) 168      168 solve  **4 crash**  4 crash       38 s vs
                                                             139 s SPRAL
  ==========  =======  =========  ===========  ============  =========

  The four b737 crashes survive **every** SPRAL option tried (all five
  scalings, ``spral_u``, ``spral_small``, ``spral_umax``, the pivot
  method); MA27 and MUMPS solve all four. One is bundled as
  ``examples/data/b737_spral_crash.nl``.

Verdict: **MA27 is the right default when present** (fastest, most
robust: it solved every sub-problem of both decks, and 3-4x faster).
**SPRAL is the open-source default**: it certifies the same optima and
the D8 sentinel MUMPS fails, at 3-4x the wall time, with one real
capability gap -- a rank-deficient-KKT crash that no option closes, which
LCsolver survives by retrying the sub-problem under MA27 or MUMPS. MUMPS
remains the solver of last resort. The three bundled captures are the
regression sentinels that say when this assessment should be revisited.

Re-running the assessment::

    python utilities/linear_solver_benchmark.py \
        --executable /path/to/dual/ipopt --linear-solvers ma27,mumps

(one subprocess per model/solver cell, hard per-cell timeout).

The multi-solver build: what the installer does
------------------------------------------------

``lcsolver-install-solvers`` (running the shipped
``lcsolver/scripts/install_ipopt.sh``) builds, from source, one IPOPT
carrying **MUMPS + SPRAL** by default and **MA27** when asked::

    lcsolver-install-solvers                           # MUMPS + SPRAL
    lcsolver-install-solvers --ma27 <ma27-1.0.0 dir>   # ... + MA27
    lcsolver-install-solvers --add-ma27 <dir>          # MA27 onto an
                                                       # existing build

The build is idempotent per component (a failed run re-runs; ``--add-ma27``
rebuilds only the HSL component and the IPOPT link). SPRAL needs a GCC
toolchain, METIS, hwloc and autotools (macOS: ``brew install gcc metis
hwloc autoconf automake libtool``; Debian: ``gfortran g++ libmetis-dev
libhwloc-dev``); without them it is skipped with a message naming what to
install, and the build carries MUMPS (+ MA27). Prebuilt IPOPTs (conda,
Homebrew, apt; ``--prebuilt``) are MA27-only or MUMPS-only and never carry
SPRAL.

Two products from one source, and why: SPRAL is OpenMP code built with
GCC (libgomp). A conda Python already holds LLVM's OpenMP runtime
(libomp), and two OpenMP runtimes in one process is a hard abort -- so
SPRAL cannot live in the shared library cyipopt loads in-process. The
installer therefore makes

* ``<root>/ipopt/build/bin/ipopt`` -- a STATIC executable: MUMPS + SPRAL
  [+ MA27], every solver switchable per solve;
* ``<root>/ipopt/build/lib/`` -- the SHARED library for cyipopt: MUMPS
  [+ MA27].

cyipopt only evaluates grey-box functions in-process and never selects
the linear solver, so it loses nothing; ``default_linear_solver`` never
picks SPRAL on that route.

Gotchas the script carries so you do not have to: SPRAL's C++ objects
need the same C++ runtime as IPOPT's, so when SPRAL is in the build IPOPT
is built with the same GCC; with GCC, ``-fno-devirtualize-speculatively``
is REQUIRED (speculative devirtualization emits vtable references to
dependency-detector classes IPOPT declares but never compiles -- Ma28 --
and the link dies on a symbol nothing uses); at runtime SPRAL needs
``OMP_CANCELLATION=TRUE`` and ``OMP_PROC_BIND=TRUE``, which LCsolver sets
whenever SPRAL is selected (without them SPRAL aborts its first
factorization, surfacing as ``internalSolverError``).

The SPRAL crash sentinel
------------------------

The bookend to the MUMPS sentinel: ``examples/data/d8_spral_crash.nl`` is
a D8 SIA sub-problem on which SPRAL v2023.03.29 **kills the ipopt
executable** (SIGBUS or SIGSEGV, return code -10/-11 through pyomo) at
IPOPT's tolerances. The mechanism, from IPOPT's own log: SPRAL's
factorization comes back "Singular system, estimated rank 1491 of 1709",
IPOPT adds its regularization (``delta_c = 5.6e-9``) and re-factorizes,
and SPRAL dies inside that second factorization. MA27 and MUMPS solve the
identical file (MA27 in 33 iterations). Deterministic, single-threaded,
independent of the OpenMP binding and thread stack size; the trigger is
``constr_viol_tol=1e-12`` (IPOPT's stock 1e-4 never reaches the
degenerate iterate). Two of the D8's 229 SIA sub-problems hit it under
IPOPT's stock SPRAL settings; the run still certified because both fell
in Phase-I restoration solves, whose failures SIA absorbs.

Hunting a SPRAL failure deliberately had found nothing first: the
ill-conditioned GP family to 44 orders of magnitude and the Hoburg GP
with every inequality duplicated 64 times (rank-deficient by
construction) all solve identically under SPRAL and MA27. The crash
needed a real deck's degenerate interior-point iterate.

Measured on the full deck (358 sub-problems each):

==========================  ========  ==========  =============
SPRAL option                crashes   certified   wall time
==========================  ========  ==========  =============
stock (matching scaling)    2         yes         103 s
``spral_small=1e-12``       2 (moved) yes         111 s
``spral_u=0.5``             crashed at sub-problem 30
``spral_scaling=mc64``      **0**     yes         **86 s**
==========================  ========  ==========  =============

So LCsolver sets ``spral_scaling=mc64`` whenever SPRAL is selected (an
explicit user setting wins; ``lcsolver.environment.
LINEAR_SOLVER_DEFAULT_OPTIONS``).

MC64 is not the whole answer. The b737 deck's sub-problems, replayed
standalone at the deck's own 1e-9 tolerances, crash SPRAL on 4 of 168
under **every** option combination tried -- all five scalings, the pivot
thresholds, the pivot method -- while MA27 and MUMPS solve all four
(``examples/data/b737_spral_crash.nl`` is one). That is the one genuine
capability gap between the two, and it runs SPRAL -> MA27: nothing was
found that MA27 fails and SPRAL solves (the D8 and b737 batches, the
ill-conditioned family to 44 decades, 64x duplicated rows, the raw-NLP
route). A dead executable is therefore a class of failure to survive,
not a file to fix: every IPOPT launch runs inside ``ipopt_launch``, which
captures pyomo's raw ERROR log instead of printing it, raises
``IpoptCrashed`` naming the signal and the linear solver, and lets the
SIA/SLCP sub-problem loops retry that one sub-problem under the most
robust other solver the build has -- MA27, else MUMPS (which is why the
open-source build carries both) -- filing the event as ``[LC-W313]`` in
the post-solve report.

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
