Sensitivities
=============

After a solve, EDI can report how strongly the optimum responds to each
``Constant`` in the model. This is the same diagnostic ``GPkit`` prints for a
geometric program, and it is usually the most informative thing you get out of a
design optimization: it ranks the assumptions in your model by how much they
actually matter.

Basic use
---------

.. code-block:: python

    from edi import Formulation
    from edi.solvers.solver import solve

    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='', description='x')
    y = f.Variable(name='y', guess=1.0, units='', description='y')
    a = f.Constant(name='a', value=2.0, units='', description='a')

    f.Objective(x + y)
    f.ConstraintList([x * y >= a])

    solve(f)
    f.print_sensitivities()

which gives

.. code-block:: text

    ========================================================================
    Sensitivities to constants    [d log(f*) / d log(c)]
    objective = 2.82843    duals: kkt
    ========================================================================
      a                               +0.5000   +++++++++++
    ========================================================================

Here ``f* = 2*sqrt(a)``, so the sensitivity is exactly ``1/2``: a 1% increase in
``a`` raises the optimum by about 0.5%.

To work with the numbers rather than print them::

    result = f.sensitivities()
    result['sensitivities']      # {'a': 0.5}
    result['objective']          # 2.8284271
    result['method']             # where the duals came from


What is reported
----------------

By default the reported quantity is the **log-log sensitivity**, or elasticity,

.. math::

    s_c = \frac{d \log f^*}{d \log c} = \frac{c}{f^*}\frac{d f^*}{d c}

Elasticities are unitless. That is the point: a model mixing densities, areas
and Reynolds numbers produces raw derivatives whose magnitudes say more about
the units than about the design, and they cannot be ranked against one another.
Elasticities can. A sensitivity of ``+2`` means a 1% increase in that constant
costs 2% of objective; ``-0.5`` means relaxing it *helps*.

Pass ``normalized=False`` for the raw derivative :math:`d f^*/d c`, in units of
``[objective]/[constant]``::

    f.sensitivities(normalized=False)


How it is computed, and why it is fast
--------------------------------------

The obvious way to get these numbers is to perturb each constant and re-solve,
costing ``2N`` solves for ``N`` constants. EDI does not do this. It uses the
envelope theorem, which extracts the same information from the duals of a single
solve:

.. math::

    \frac{d f^*}{d \theta} = \frac{\partial f}{\partial \theta}
      - \sum_i \lambda_i
        \frac{\partial(\mathrm{body}_i - \mathrm{bound}_i)}{\partial \theta}

Because the primal variables are stationary at the optimum, only the *partial*
derivatives at fixed :math:`x^*` survive, and each of those is taken
symbolically with Pyomo's reverse-mode differentiation. Two consequences:

* **Cost is independent of the number of constants.** One solve, plus a few
  expression walks. On the aircraft GP in ``examples/aircraft_gp.py`` (13
  constants) this is about 20x faster than perturb-and-re-solve, and the gap
  widens linearly as constants are added.
* **There is no truncation error.** The result is the exact derivative of the
  model, not a difference quotient, so there is no step size to tune. A finite
  difference on the same problem needs its step chosen carefully: too small and
  solver noise dominates, too large and truncation does.

Supported problem classes
-------------------------

Sensitivities are available for every convex structure EDI detects, on either
solver core:

.. list-table::
   :header-rows: 1

   * - Structure
     - cvxopt core
     - IPOPT core
     - Exact?
   * - Linear program
     - yes
     - yes
     - exact
   * - Quadratic program
     - yes
     - yes
     - exact
   * - Geometric program
     - yes
     - yes
     - exact
   * - Signomial program (SP / SLCP)
     - yes
     - yes
     - local approximation
   * - General NLP / black-box
     - --
     - yes
     - exact at the local optimum

For a signomial program the duals belong to the **final convex subproblem** of
the sequence, so the sensitivities describe the neighborhood of the returned
point rather than a global property. This is the intended reading, and the
result carries ``result['approximate'] == True`` so it is never silently
presented as exact.

Where the duals come from
-------------------------

The formula needs duals keyed by Pyomo constraint. EDI obtains them in one of
two ways, reported as ``result['method']``:

``suffix``
    The IPOPT route imports duals directly from the solver into Pyomo's ``dual``
    Suffix. Nothing has to be reconstructed.

``kkt``
    The cvxopt backends solve a transformed problem and return only a raw
    vector, so duals are recovered from the primal solution by solving the KKT
    stationarity condition in the least-squares sense over the active set. This
    depends on nothing but the primal solution, which is why it works uniformly
    across the LP, QP, GP and SP backends.

The KKT route reproduces IPOPT's duals to within a relative 1e-5 on the
aircraft GP, and both routes agree with a finite-difference re-solve.

Reliability
-----------

``result['stationarity_residual']`` reports how well the recovered duals satisfy
stationarity, relative to the size of the objective gradient. A converged solve
gives something around 1e-6. If it exceeds 1e-3, or if no binding constraints
were found at all, a ``RuntimeWarning`` is raised: sensitivities built on duals
that do not satisfy stationarity are not sensitivities, and a table of quiet
zeros from a diverged solve is worse than an error.

Two things to keep in mind:

* Call it **after** a solve. Every EDI backend writes the solution back onto the
  model, so ``solve(f)`` followed by ``f.sensitivities()`` is the intended
  sequence.
* An indexed ``Constant`` reports one sensitivity per element, named as
  ``c[0]``, ``c[1]``, and so on, matching how GPkit reports vectors.

API
---

.. code-block:: python

    f.sensitivities(normalized=True, method='auto', rtol=1e-4)
    f.print_sensitivities()

    from edi import sensitivities, constraint_duals, format_sensitivities

    sensitivities(model, normalized=True, method='auto')
    constraint_duals(model, method='auto')      # {constraint: dual}
    format_sensitivities(result)                # the printable table
