Solvers and Backends
====================

LCsolver inspects a formulation, classifies its algebraic structure, and routes
it to a solver suited to that structure. This page describes the routes and
what they cost.

The routing pipeline
--------------------

The default requires no decision::

    sol = lcsolver.solve(f)

``solve`` first runs the model checks -- units, then the unbuilt-block gate and
the pre-solve gate (:doc:`checks`); a half-assembled model or an ill-posed one
stops there with a :class:`~lcsolver.solvers.solver.PresolveError` rather than
a strange answer. It then classifies the structure and dispatches:

* a detected **LP, QP or GP** goes to the convex backend -- **IPOPT by
  default**, solving in log space (below); ``cvxopt`` is the automatic
  fallback when no IPOPT is installed, and can be chosen explicitly with
  ``convex_backend='cvxopt'``;
* a detected **SP** goes to **SIA**, sequential inner approximation;
* a model with **black-box constraints** goes to SIA as well, each box
  imposed as an opaque row inside the trust-region loop -- letting the
  structured route run without them would solve a relaxation and report it
  as the optimum. ``solver='ipopt'`` remains the escape hatch to hand the
  raw model, boxes included, to IPOPT via cyipopt;
* anything else goes to the raw NLP route (``solvers/ipopt/NLP.py``).

Architecturally the tree matches this: ``cvxopt/`` and ``ipopt/`` hold the
atomic solves (``LP/QP/GP`` under cvxopt; log-space ``GP`` and the unique
``NLP`` under ipopt), while the sequential methods -- PCCP, SLCP, SIA -- live
once in ``solvers/sequential/`` and take the convex machinery underneath as a
component, not a copy.

Both convex routes solve the *same* convex problem. A geometric program is not
convex in its natural variables, so the IPOPT route applies the standard
logarithmic change of variables first. With :math:`x_j = e^{t_j}` a monomial
becomes

.. math::  c_k \prod_j x_j^{a_{kj}} = e^{\,b_k + a_k^\top t},
           \qquad b_k = \log c_k,

so a posynomial becomes a sum of exponentials of affine functions, which is
convex. The global-optimality guarantee is therefore preserved; only the
numerical machinery differs.

This matters. Handing a geometric program to a general nonlinear solver *in
its natural variables* forfeits that guarantee, and on realistic models it
often fails outright -- see below.

Cost Comparison
---------------

``utilities/benchmark_solvers.py`` runs every backend on identical models.
Since all of them solve the same problem, disagreement in the objective would
be a correctness failure, and the timings are a like-for-like comparison.

Median of five runs, on the geometric programs of the SLCP apparatus
(:doc:`examples`):

.. list-table::
   :header-rows: 1
   :widths: 26 18 18 18 20

   * - Backend
     - simple (2 var)
     - K-O (18 var)
     - Hoburg (61 var)
     - Guarantee
   * - ``cvxopt`` (convex)
     - **12.8 ms**
     - **26.3 ms**
     - 65.7 ms
     - global
   * - ``ipopt`` (convex)
     - 15.9 ms
     - 26.6 ms
     - **51.3 ms**
     - global
   * - ``ipopt`` (raw NLP)
     - 13.9 ms
     - *fails*
     - *fails*
     - local only
   * - SLCP
     - 191 ms
     - 893 ms
     - 1628 ms
     - global on the GP part

Three things to take from this.

**The two convex backends cross over with problem size.** cvxopt's dense
approach has lower fixed overhead and wins on the smallest models; IPOPT's
sparse linear algebra wins once the model is large enough to pay for it. They
are within 1% of each other at eighteen variables. IPOPT is the default
anyway: it tolerates the wide variable boxes realistic models carry, it solves
models cvxopt returns ``status='unknown'`` on, and it is the only route that
can evaluate a black-box constraint. cvxopt remains available, is the
automatic fallback, and is still the faster choice on a small, well-scaled
program.

**The raw-NLP route fails on the larger models**, and the reason is
conditioning rather than any deficiency in IPOPT. At the Hoburg optimum the
natural variables run from :math:`\bar{I}_{cap} \approx 2 \times 10^{-5}` to
:math:`Re_2 \approx 10^{7}` -- a range of :math:`5 \times 10^{11}`, or 11.7
decades. A general solver must cope with that directly, and here it exhausts
its iteration limit. The log transform maps the same range onto a span of 27
in the transformed variables *and* makes the problem convex. Both convex
backends then solve it in tens of milliseconds.

That is the practical argument for detecting structure: not only the
global-optimality guarantee, but solvability at all.

**Sequential methods are far slower on these problems, and should be.** SLCP
(and SIA, its certified successor) are algorithms for signomial programs;
applying one to a pure geometric program means paying for a sequence of convex
subproblems where one would do. They earn their cost only when the model is
*not* fully GP-compatible -- a signomial row, a black box.

What the sequential path does to the model first
------------------------------------------------

The sequential solvers take variable bounds natively, so declared bounds
carried as constraint rows are pure cost: one more log-sum-exp to build and
differentiate every iteration, for a statement the solver already has. They are
folded into native bounds on the way in, whether or not ``presolve`` is on --
``presolve=False`` opts out of the *column* reductions, which can change the
trajectory, not out of the exact singleton fold. On a 4,160-row aircraft model
that is 2,596 rows removed and 1,338 s down to 188 s for the identical
38-iteration solve.

Only **declared-bound** rows fold. A single-variable model row -- a span gate,
say -- stays a row, because the elastic relaxation can put slack on a row but
not on a hard bound, and an active gate is exactly the constraint that needs
slack mid-trajectory; folding all of them stalled the same model at the
iteration cap.

Monomial-equality elimination is **off** on this path, though it remains
default-on in :func:`~lcsolver.presolve.reductions.presolve` for other callers.
Substituting away 706 variables shrank the same model to 592 variables and
3,454 rows and took the solve from 39 iterations / 136 s to 62 iterations /
2,504 s: the early iterations stay cheap and the trajectory then enters an
expensive restoration phase the unsubstituted problem never visits. Smaller is
not faster for the sequential solvers. See :doc:`PRESOLVE`.

When infeasibility is the answer
--------------------------------

A structured backend that *proves* infeasibility is not a backend failure, and
LCsolver no longer falls through to the raw NLP route on one. Raw IPOPT on the
same rows either fails its own restoration phase or returns numbers for a
design that does not exist, and both outcomes read like solver trouble rather
than like the model being wrong. The same holds inside SIA: a run whose Phase I
never found a feasible point, and which did not recover, raises with the
elastic report naming the rows that cannot close, rather than returning its
best infeasible iterate as though it were a design.

Writing the Solution Back
-------------------------

Every backend applies its solution to the model, so after a solve the
variables hold their optimal values::

    sol = lcsolver.solve(f)
    print(pyo.value(f.x))       # the optimum, not the initial guess

``solve`` returns a :class:`~lcsolver.solvers.solver.SolveResult` carrying the
same information as an object -- the summary, named accessors with units, the
captured messages -- see :doc:`results`. If only the raw result is wanted,
write-back can be disabled on the backend entry points themselves:
``cvxopt_solve(m, write_back=False)``, and ``ipopt_solve(m,
load_solutions=False)`` on the raw NLP route.

A note on ``cvxopt``: it can return a non-converged point with
``status='unknown'`` and raise nothing. LCsolver flags that (``LC-W205``)
rather than letting the point pass for a solution. If you see it, the IPOPT
convex backend is the fix, and when no IPOPT is installed at all the final
error says exactly that.
