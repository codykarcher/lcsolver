Solvers and Backends
====================

LCsolver inspects a formulation, classifies its algebraic structure, and routes it to
a backend suited to that structure. This page describes the available routes and
what they cost.

Choosing a Backend
------------------

The default requires no decision::

    from lcsolver.solvers.solver import solve

    result = solve(f)

``solve`` detects the structure and dispatches. A linear, quadratic or geometric
program goes to ``cvxopt``; a signomial program goes to the successive-monomial
solver; anything else, including any model containing a black-box constraint,
goes to IPOPT.

For a structured problem the convex backend can be switched::

    solve(f, convex_backend='ipopt')     # log-space convex solve with IPOPT
    solve(f, solver='ipopt-convex')      # equivalent, stated explicitly

Both routes solve the *same* convex problem. A geometric program is not convex in
its natural variables, so the IPOPT route applies the standard logarithmic change
of variables first. With :math:`x_j = e^{t_j}` a monomial becomes

.. math::  c_k \prod_j x_j^{a_{kj}} = e^{\,b_k + a_k^\top t},
           \qquad b_k = \log c_k,

so a posynomial becomes a sum of exponentials of affine functions, which is
convex. The global-optimality guarantee is therefore preserved; only the
numerical machinery differs.

This matters. Handing a geometric program to a general nonlinear solver *in its
natural variables* forfeits that guarantee, and on realistic models it often
fails outright -- see below.

Cost Comparison
---------------

``examples/benchmark_solvers.py`` runs every backend on identical models. Since
all of them solve the same problem, disagreement in the objective would be a
correctness failure, and the timings are a like-for-like comparison.

Median of five runs, on the geometric programs of :doc:`examples`:

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

**The two convex backends cross over with problem size.** cvxopt's dense approach
has lower fixed overhead and wins on the smallest model; IPOPT's sparse linear
algebra wins once the model is large enough to pay for it. They are within 1% of
each other at eighteen variables. Neither is universally better, which is why
both are available and cvxopt remains the default.

**The raw-NLP route fails on the larger models**, and the reason is conditioning
rather than any deficiency in IPOPT. At the Hoburg optimum the natural variables
run from :math:`\bar{I}_{cap} \approx 2 \times 10^{-5}` to
:math:`Re_2 \approx 10^{7}` -- a range of :math:`5 \times 10^{11}`, or 11.7
decades. A general solver must cope with that directly, and here it exhausts its
iteration limit. The log transform maps the same range onto a span of 27 in the
transformed variables *and* makes the problem convex. Both convex backends then
solve it in tens of milliseconds.

That is the practical argument for detecting structure: not only the
global-optimality guarantee, but solvability at all.

**SLCP is far slower on these problems, and should be.** It is a general
algorithm for signomial programs; applying it to a pure geometric program means
paying for a sequence of convex subproblems where one would do. It earns its cost
only when the model is *not* fully GP-compatible -- see :doc:`slcp`.

Writing the Solution Back
-------------------------

Every backend applies its solution to the model, so after a solve the variables
hold their optimal values::

    solve(f)
    print(pyo.value(f.x))       # the optimum, not the initial guess

This can be disabled with ``write_back=False`` if only the result dictionary is
wanted.

A note on ``cvxopt``: it can return a non-converged point with
``status='unknown'`` and raise nothing. LCsolver issues a ``RuntimeWarning`` in that
case rather than letting the point pass for a solution. If you see it, the
IPOPT convex backend is worth trying.
