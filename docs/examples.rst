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
   * - ``hydrogen_network_lp.py``
     - A delivery network as a **linear** program, and the matrix machinery:
       a ``size=[3, 4]`` variable, ``f.sum`` along either axis,
       ``broadcast_rows``/``broadcast_cols`` for per-element caps, and
       ``domain=NonNegativeReals`` in place of a bounds pair.
   * - ``isolator_stack_qp.py``
     - Static equilibrium as a **quadratic** program -- minimum potential
       energy -- with a hard stop. Slice arithmetic ``x[1:] - x[:-1]`` builds
       the spring extensions, and the stop's dual is read back as a contact
       force in newtons.
   * - ``gear_train_gp.py``
     - A gearbox as a geometric program: ``f.prod`` for the overall ratio, a
       monomial recurrence over slices for the torque, an elementwise mass
       fit, and a **holographic** constraint (the edge of the fit's data)
       that goes active and says so.
   * - ``cooling_loop_gp.py``
     - ``f.group()``: three subsystems built by three functions that pass no
       prefixes and refer to each other as ``f.loop.mdot``. Also mixes mK/W
       against K in one constraint, which only balances because the unit
       corrector runs first.
   * - ``aircraft_gp.py``
     - A simple aircraft sizing problem as a geometric program, with full
       units, Constants, and the sensitivity accessors.
   * - ``boyd.py``
     - The box design problem from Boyd's GP tutorial, and the introductory
       test problem of the SLCP paper: maximize volume under wall/floor area
       and aspect-ratio limits. Solves to 5.196e-3 1/m\ :sup:`3` at
       w = 5.774 m, h = 2.887 m, d = 11.547 m, matching the paper.
   * - ``floudas.py``
     - The Floudas heat exchanger design, SLCP paper Equation 17: eight
       variables, six constraints, five of them genuine **signomials**
       (negative coefficients). Reproduces the published optimum 7049.249 on
       every variable. Its header documents the paper's Equation 17 typo
       (x2 for x1 in the first denominator).
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

Higher-fidelity black-box variants -- XFOIL evaluated in-process, NeuralFoil
in PyTorch, airfoil design variables co-optimized with the aircraft -- live in
the `lcuav <https://github.com/codykarcher/lcuav>`_ repository, which builds
on the Hoburg model.
