Examples
========

A Geometric Program for Aircraft Design
---------------------------------------

.. literalinclude:: ../examples/aircraft_gp.py
    :language: python


Problems from the SLCP Literature
---------------------------------

``examples/slcp_cases.py`` and ``examples/slcp_formulations.py`` contain the four
test problems used to develop and validate Sequential Log-Convex Programming
(:doc:`slcp`). They are useful independently of that solver: between them they
cover geometric programs, signomial programs, and models with black-box
constraints, at sizes from two to sixty-one variables.

.. list-table::
   :header-rows: 1
   :widths: 22 12 12 54

   * - Problem
     - Variables
     - Constraints
     - Character
   * - ``simple``
     - 2
     - 1
     - A geometric program small enough to plot. The worked example below.
   * - ``floudas``
     - 8
     - 6
     - Heat exchanger. Five of six constraints carry negative coefficients, so
       they are signomials rather than posynomials.
   * - ``ko``
     - 18
     - 17
     - Aircraft sizing. Signomial, because available fuel volume is a monomial
       bounded below by a posynomial.
   * - ``hoburg``
     - 61
     - 58
     - UAV conceptual design over three mission segments. Fully GP-compatible,
       with variants that replace the profile-drag model with a black box.

Run them all against every algorithm::

    python examples/run_slcp.py
    python examples/run_slcp.py hoburg --trend

The Two-Variable Example
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

    \begin{aligned}
    \underset{x,y}{\text{minimize}} \quad
      & x^{-0.1} + 15x^{0.01} + y^{-0.1} + 15y^{0.01} \\
    \text{subject to} \quad
      & 0.01x^{-1.1} + x^{0.1} + y \le 1
    \end{aligned}

Written as an LCsolver ``Formulation``:

.. code-block:: python

    from lcsolver import Formulation
    from pyomo.environ import units

    f = Formulation()
    f.Variable(name='x', guess=0.3,  units='', description='x')
    f.Variable(name='y', guess=0.05, units='', description='y')

    f.Objective(f.x ** -0.1 + 15 * f.x ** 0.01
                + f.y ** -0.1 + 15 * f.y ** 0.01)

    f.ConstraintList([
        0.01 * f.x ** -1.1 + f.x ** 0.1 + f.y <= 1.0 * units.dimensionless,
    ])

Every term is a monomial with a positive coefficient, so LCsolver detects a geometric
program and routes it to a convex backend. The optimum is
:math:`f^\ast = 31.8115934` at :math:`(x, y) = (0.0593208, 0.0224878)`.

The Hoburg UAV Problem
~~~~~~~~~~~~~~~~~~~~~~

The largest model in the set: 61 variables and 58 constraints covering an
outbound leg, a return leg, and a sprint condition that sizes the powerplant.
It is written out in full in ``examples/slcp_formulations.py``. Its solution has
been checked against the published values:

.. list-table::
   :header-rows: 1
   :widths: 30 25 25

   * - Quantity
     - Published
     - LCsolver
   * - :math:`P_{max}` [kW]
     - 1186.1
     - 1186.09
   * - :math:`T_{sprint}` [N]
     - 2209.6
     - 2209.61
   * - :math:`W_{eng}` [N]
     - 2805.8
     - 2805.82
   * - :math:`C_{D_p,\text{sprint}}`
     - 0.005732
     - 0.005732

Agreement is within 0.006% on every tabulated quantity.

Black-Box Variants
~~~~~~~~~~~~~~~~~~

The Hoburg profile-drag model can be replaced by an opaque analysis code on any
subset of the three mission segments::

    from examples.slcp_cases import hoburg

    problem, x0, _ = hoburg(n_blackbox=0)   # explicit posynomial fit
    problem, x0, _ = hoburg(n_blackbox=1)   # sprint segment black-boxed
    problem, x0, _ = hoburg(n_blackbox=3)   # all three black-boxed

The black box solves the same drag fit implicitly, so the optimum is unchanged;
only the solver's visibility into the structure differs. That makes these
variants a controlled measurement of what exploiting GP structure is worth.

``examples/slcp_xfoil.py`` goes further and substitutes XFOIL itself, evaluated
through metafoil's in-memory interface. Unlike the implicit fit, this genuinely
changes the answer, since XFOIL is a different drag model. It requires
``metafoil`` and ``scikit-learn``, and is slow -- a single solve runs several
thousand XFOIL polars.
