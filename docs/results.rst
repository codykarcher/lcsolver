Results and Reporting
=====================

``solve`` returns a :class:`~lcsolver.solvers.solver.SolveResult`. It *is* a
dictionary -- every key the backends have always returned is still there
(``res['x']``, ``res['status']``, ``res['solution']``) -- but it also carries
the result as an object, so a notebook does not have to know which key holds
what::

    sol = lcsolver.solve(f)

    print(sol.summary())        # the full table, same as f.solution.summary()
    sol.report                  # how the problem was classified and solved
    sol.objective               # the objective, with units (a pint quantity)
    sol.optimality_status       # True only for a certified optimum
    sol.messages                # everything the solve wanted to warn about

Named access with units
-----------------------

Four accessors share one calling convention: no argument returns the full flat
dict keyed by display name (``'wing.AR'``, never nested sub-dicts); a string
returns the single quantity; a list returns a dict of those names. Values are
**pint quantities** (from pyomo's own registry), so they print readably,
convert with ``.to()``, and strip with ``.magnitude``; dimensionless values
come back as plain floats::

    sol.variables()                    # {'A': 10.95, 'S': <Quantity(10.95, 'meter ** 2')>}
    sol.variables('S').to('ft^2')      # unit conversion on the result
    sol.variables(['A', 'S'])          # just those two
    sol.constants('rho')               # same convention over the Constants

    sol.sensitivities()                # log-log elasticities, plain floats
    sol.dimensioned_sensitivities('c') # d(objective)/d(constant), with units

``dimensioned_sensitivities`` converts the elasticity :math:`s_c` to the raw
derivative :math:`df^*/dc = s_c \, f^*/c` and carries it in
``[objective units]/[constant units]``. See :doc:`sensitivities` for what the
numbers mean.

Messages and the code registry
------------------------------

A solve never prints warnings by default. Everything it would have said --
holographic hits, unreliable sensitivities, solver fallbacks, SIA remedies --
is captured into ``sol.messages``, each tagged with a code from the master
registry in :mod:`lcsolver.core.codes`::

    from lcsolver.core import codes

    if any(codes.HOLOGRAPHIC_ACTIVE in m for m in sol.messages):
        ...   # detect a condition without parsing prose

Pass ``quiet=False`` to ``solve`` to have the warnings emitted normally as
well. Errors always raise regardless.

``E`` codes are errors, ``W`` codes are warnings; the hundreds digit groups by
pipeline stage (1xx pre-solve, 2xx solve-time, 3xx post-solve):

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Code
     - Meaning
   * - ``LC-E001``
     - Pre-solve gate: the stated problem is ill-posed (see :doc:`checks`)
   * - ``LC-E002``
     - Units do not balance
   * - ``LC-E101``
     - The problem was proven infeasible
   * - ``LC-W101``
     - Pre-solve findings, demoted to a warning (``diagnostics='warn'``)
   * - ``LC-W102``
     - Structure detection failed; solved as a raw NLP
   * - ``LC-W201``
     - The structured backend failed; fell back
   * - ``LC-W202``
     - No usable IPOPT; solved with cvxopt
   * - ``LC-W203``
     - SIA returned a best iterate, not a certified optimum
   * - ``LC-W204``
     - A GP form reported optimality its verification solve contradicted
   * - ``LC-W205``
     - cvxopt returned a non-optimal status without raising
   * - ``LC-W206``
     - The solve succeeded but write-back onto the model failed
   * - ``LC-W301``
     - Holographic constraints are active at the solution (:doc:`holographic`)
   * - ``LC-W302``
     - Recovered duals fail stationarity; every sensitivity is unreliable
   * - ``LC-W303``
     - Degenerate active set; the named sensitivities are not determined
   * - ``LC-W304``
     - Variables resting on the solver's positivity floor

The Report and the Post Solve Report
------------------------------------

``sol.summary()`` opens with a **Report** stating, accurately, how the problem
was handled -- "auto-detected" is claimed only when the router actually chose,
and a prescribed solver or backend is reported as prescribed::

    Report
    ------
       Problem auto-detected as a signomial program (SP), with 3 black-box constraints
       Solved with ipopt (SIA, sequential inner approximation)

and closes with a **Post Solve Report** owning the status line and every
captured message::

    Post Solve Report
    -----------------
       Status: optimal
       [LC-W301] 1 of 1 holographic constraints are ACTIVE at the solution --
           constraint_2: x <= m (at 1, margin +8.80e-09). ...

``sol.report`` returns the top section as a string; ``sol.messages`` holds the
message list.

The relationship to ``f.solution``
----------------------------------

The solve writes values back onto the model, and ``f.solution`` builds the
same rich :class:`~lcsolver.objects.solution.Solution` from them --
``sol.solution`` and ``f.solution`` are interchangeable, and
``sol.summary()`` equals ``f.solution.summary()``. The dict key
``res['solution']`` remains what it has always been: the flat
``{name: value}`` write-back record.
