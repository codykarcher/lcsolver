Sequential Log-Convex Programming
=================================

Most engineering design models are *almost* geometric programs. A wing-sizing
model might be entirely GP-compatible except for one constraint whose available
fuel volume is bounded below by a sum of terms, or except for the one place where
an external aerodynamic code supplies a coefficient. Structure detection will
correctly report such a model as a signomial program, and the convex backends
will decline it.

Sequential Log-Convex Programming (SLCP) is for exactly that situation. It does
not require the whole model to be GP-compatible -- only that some useful fraction
of it is.

The Idea
--------

Constraints are split in two. Posynomials :math:`p(x) \le 1` and monomials
:math:`m(x) = 1` become convex *exactly* under the log transform, so they are
imposed directly. Everything else, :math:`g(x) \le 1` and :math:`h(x) = 1`, is
linearised in log space as SQP would linearise in the natural variables.

.. math::

    \begin{aligned}
    \underset{d}{\text{minimize}} \quad
      & \log f(x_k)
      + \tfrac{1}{f(x_k)}\left(x_k \odot \nabla f(x_k)\right)^{\!\top} d
      + \tfrac12 d^\top \nabla^2 \mathcal{L}_R(y_k)\, d
      + K \textstyle\sum_i \sigma_i^2 \\
    \text{subject to} \quad
      & \log\!\left(\textstyle\sum_j \exp(P_j(d + \log x_k) + q_j)\right)
        \le \sigma_i \\
      & A_m (d + \log x_k) + b_m \le \sigma_i \\
      & \log g(x_k)
        + \tfrac{1}{g(x_k)}\left(x_k \odot \nabla g(x_k)\right)^{\!\top} d
        \le \sigma_i \\
      & \log h(x_k)
        + \tfrac{1}{h(x_k)}\left(x_k \odot \nabla h(x_k)\right)^{\!\top} d
        = \sigma_i
    \end{aligned}

The subproblem is therefore log-convex rather than quadratic, which is what
distinguishes SLCP from a log-space SQP. Keeping the posynomials exact stops the
subproblem from stepping outside a constraint that a linear model would have
badly underestimated -- the failure mode that costs the linearised method
iterations.

The relaxation variables :math:`\sigma_i` and their penalty :math:`K` handle the
inconsistent-linearisation problem familiar from SQP: without them the subproblem
can be infeasible even when the true problem is not.

Two details matter more than they look:

**The Reduced Lagrangian.** The BFGS approximation is built from
:math:`\mathcal{L}_R = \log f + \lambda \log g + \lambda \log h`, which *omits*
the constraints that are represented exactly. Their curvature is already captured
by the subproblem, so approximating it again sets the approximation fighting the
truth. Including them performs worse than not imposing the constraints exactly at
all.

**The watchdog.** A step that raises the merit function is not necessarily a bad
step: an algorithm that has overshot a constraint must often pass through worse
merit values on its way back to feasibility. Insisting on monotone decrease every
iteration stalls it there. A bounded run of non-monotone steps is allowed.

Usage
-----

SLCP has its own problem description, since it needs to know which constraints
are GP-compatible and needs value/gradient callbacks for the ones that are not::

    import numpy as np
    from edi.solvers.ipopt.slcp import (
        Constraint, Posynomial, Problem, Signomial, Options, solve)

    # minimise x*y  subject to  x >= 1, y >= 2
    objective = Posynomial([(1.0, [1, 1])], n=2)
    constraints = [
        Constraint(Posynomial([(1.0, [-1, 0])], 2), '<='),   # 1/x <= 1
        Constraint(Posynomial([(2.0, [0, -1])], 2), '<='),   # 2/y <= 1
    ]

    problem = Problem(2, objective, constraints, names=['x', 'y'])
    result = solve(problem, x0=[3.0, 3.0], method='slcp')

    print(result.objective, result.x, result.iterations)

A constraint that is not GP-compatible is supplied as a :class:`Signomial`, which
wraps any callable returning ``(value, gradient)`` in the natural variables::

    def fuel_volume(x):
        """V_avail / (V_wing + V_fuse) <= 1 -- not GP-compatible."""
        denom = x[iw] + x[if_]
        g = np.zeros(len(x))
        g[ia]  = 1.0 / denom
        g[iw]  = g[if_] = -x[ia] / denom ** 2
        return x[ia] / denom, g

    Constraint(Signomial(fuel_volume, n), '<=')

That callback signature is the whole black-box interface. An external analysis
code -- a panel method, a CFD run, a lookup table with finite differences -- is
wrapped the same way, since the algorithm only ever needs a value and a gradient
at the current iterate.

Baselines
---------

The same driver implements two comparison algorithms, selected by ``method``:

``'slcp'``
    As described above.
``'lsqp'``
    Every constraint linearised in log space. This is what the SLCP subproblem
    degenerates to when no posynomial constraints are present.
``'sqp'``
    Every constraint linearised in the natural variables, no log transform.

Options
-------

Defaults follow the reference implementation:

.. list-table::
   :header-rows: 1
   :widths: 34 14 52

   * - Option
     - Default
     - Meaning
   * - ``max_iterations``
     - 500
     - Outer iteration cap
   * - ``penalty_constant``
     - 1e15
     - :math:`K` on the :math:`\sigma` relaxation
   * - ``lagrangian_gradient_tolerance``
     - 1e-6
     - First-order optimality test
   * - ``step_magnitude_tolerance``
     - 1e-4
     - Convergence on :math:`\lVert d \rVert`
   * - ``watchdog_iterations``
     - 5
     - Consecutive non-monotone steps permitted
   * - ``eta``, ``rho``
     - 1e-4, 0.8
     - Armijo parameter and backtracking factor
   * - ``x_min``
     - 1e-9
     - Positivity floor

Limitations
-----------

The log transform requires every variable to be strictly positive, and every
constraint function to stay strictly positive throughout the solve. For
engineering models this is usually not intrusive -- there is no negative mass or
area -- but it is a real restriction, and the line search shortens any step that
would violate it rather than failing.

The algorithm is also considerably more expensive than a direct convex solve when
the model happens to be a pure geometric program; see :doc:`solvers`. Use it when
the model is *not* fully GP-compatible.

Validation
----------

``examples/run_slcp.py`` reproduces the published results. On the two-variable
example from :math:`(0.3, 0.05)`, SLCP converges in 13 iterations and the
fully-linearised baseline in 17, matching the reference implementation; the two
are identical through iteration 5 and diverge at iteration 6, where the
linearised method overshoots the constraint to 1.387 while SLCP holds it at
0.999999.

Across multiple starting points the advantage grows with distance from the
optimum, and shrinks as posynomial constraints are replaced by black boxes:

.. list-table:: Mean iterations, 12 random starts per band
   :header-rows: 1
   :widths: 34 22 22 22

   * - Hoburg variant
     - ±50% SLCP/LSQP
     - ±80% SLCP/LSQP
     - SLCP advantage
   * - 0 black boxes
     - 15.8 / 17.1
     - 16.4 / 18.4
     - 7-11%
   * - 1 black box
     - 15.8 / 16.9
     - 16.7 / 17.7
     - 6-7%
   * - 3 black boxes
     - 15.8 / 16.6
     - 16.4 / 17.7
     - 5-7%

Both methods converge from every start in every band; the difference is cost, not
reliability. Results within ±10% of the optimum are not reported because at this
sample size they are not stable across random seeds.

Reference
---------

Karcher, C. and Haimes, R., "A Method of Sequential Log-Convex Programming for
Engineering Design", *Optimization and Engineering*, 2022.
https://doi.org/10.1007/s11081-022-09750-3
