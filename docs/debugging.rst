Debugging a model
=================

``lcsolver.solve(f)`` runs a chain -- unit correction, structure detection,
pre-solve checks, then a backend -- and when a model is broken the fastest
route to the fix is to run the failing stage **by itself** and read its
report, rather than re-solving and deciphering tracebacks. Each stage has a
tool, each tool catches its own class of mistake, and they are meant to be
run in the order the chain runs them.

The tools
---------

``lcsolver.unit_check(f)``
    Do the units balance? Nothing downstream means anything before this
    passes. The report names the offending constraint, prints both sides'
    units, and -- for a genuine dimension mismatch -- says exactly what
    factor the short side is missing, which is usually the whole diagnosis
    (a dropped dynamic pressure, a forgotten gravitational acceleration).

``f.structure_report()``
    What kind of problem is this, and which specific constraints stop it
    being a simpler one? A model that "is an SP" is an SP *because of
    particular rows*, and they are usually one reformulation away from
    posynomial -- the classic slip being a posynomial **equality** where an
    inequality costs nothing. By default the report describes what the
    solver will see after presolve; the blockers table names each offending
    constraint with its reason.

``structure_detector(unit_corrector(f))``
    The detector itself -- what ``solve()`` actually routes on. Its
    ``structures['blockers']`` dict records why each problem class was
    refused, including declaration-level causes such as a **zero or negative
    variable bound** (a log-space variable is strictly positive, so
    ``bounds=[0.0, ...]`` silently costs the model its GP/SP class). On a
    grey-box model that misclassification also reroutes the solve to the raw
    IPOPT path, discarding any ``SIAOptions`` -- ``solve()`` warns
    ``[LC-W207]`` when that happens, and the blockers say why.

``f.optimization_check()``
    Structure plus presolve in one report: variables that appear in no
    constraint (nothing can price them), variables with no bounds, rows that
    are infeasible as written, degenerate constraint pairs. These are the
    mistakes that units and structure cannot see because they are not wrong,
    just meaningless.

``lcsolver.solve(f)`` and what it hands back
    The solve itself runs *quiet* by default: every warning it raises --
    holographic constraints that came out active, unreliable sensitivities,
    backend fallbacks, discarded options -- is captured into
    ``sol.messages`` rather than lost to a scrolled terminal. Reading that
    list after a suspicious solve is stage five of debugging.

The walkthrough
---------------

The example below builds one small wing-sizing GP, breaks it four different
ways, and lets each tool catch its own mistake; running the file prints
every report shown above.

.. literalinclude:: ../examples/debugging_a_gp.py
   :language: python
