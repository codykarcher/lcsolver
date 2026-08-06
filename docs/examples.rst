Examples
========

Every file in ``examples/`` is standalone: run it with ``python`` and it
builds its model, solves, and prints the summary. The test suite imports each
one, so they cannot drift from the package.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - What it shows
   * - ``readme_example.py``
     - The quickstart model: a black-box function with units that differ
       from the model's, solved through SIA. See :doc:`quickstart`.
   * - ``aircraft_gp.py``
     - A simple aircraft sizing problem as a geometric program, with full
       units, Constants, and the sensitivity accessors.
   * - ``kirschen_ozturk.py``
     - Aircraft sizing as a **signomial** program: one fuel-volume row is a
       monomial bounded below by a posynomial, so the router sends it to SIA.
       States the range in km and the TSFC in 1/hr, exercising the unit
       corrector. Solves to :math:`W_f = 870.80` N.
   * - ``hoburg.py``
     - The Hoburg UAV problem as a pure geometric program: 61 variables over
       three mission segments, validated against the published values below.
   * - ``hoburg_blackbox.py``
     - The same UAV with the profile-drag posynomial hidden inside a black
       box. The optimum is unchanged -- only the solver's visibility into
       the structure changes -- and the Post Solve Report earns its keep:
       the SIA duals fail stationarity (``LC-W302``), and finite
       differencing confirms the reported sensitivities would be wrong.

A Geometric Program for Aircraft Design
---------------------------------------

.. literalinclude:: ../examples/aircraft_gp.py
    :language: python

The Hoburg UAV Problem
----------------------

The largest model in the set: 61 variables and 58 constraints covering an
outbound leg, a return leg, and a sprint condition that sizes the powerplant
(``examples/hoburg.py``). Its solution has been checked against the published
values:

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

Agreement is within 0.006% on every tabulated quantity. The black-box variant
in ``examples/hoburg_blackbox.py`` reproduces the same optimum with the drag
model opaque, which makes the pair a controlled measurement of what exploiting
GP structure is worth.

The SLCP Paper Apparatus
------------------------

The test problems used to develop and validate Sequential Log-Convex
Programming (:doc:`slcp`) live in ``utilities/``, not ``examples/`` -- they
are solver-development apparatus rather than modeling examples.
``utilities/slcp_cases.py`` states the paper's problems (``simple``,
``floudas``, ``ko``, ``hoburg`` and its black-box variants) in low-level
``Problem`` form with published reference optima;
``utilities/slcp_formulations.py`` restates the GP-compatible ones as
user-style Formulations; ``utilities/run_slcp.py`` drives them all against
every algorithm, and ``utilities/benchmark_solvers.py`` compares backends on
identical models (the numbers in :doc:`solvers`)::

    python utilities/run_slcp.py
    python utilities/run_slcp.py hoburg --trend

Note the apparatus keeps the paper's dimensionless transcriptions exactly --
its ``ko`` case lands on 892.68 N against its own reference, where the
units-carrying ``examples/kirschen_ozturk.py`` (whose Reynolds relation must
carry :math:`\rho` to be dimensionally consistent) lands on 870.80 N. Both
numbers are right; they answer slightly different statements.

Higher-fidelity black-box variants -- XFOIL evaluated in-process, NeuralFoil
in PyTorch, airfoil design variables co-optimized with the aircraft -- live in
the `lcuav <https://github.com/codykarcher/lcuav>`_ repository, which builds
on the Hoburg model.
