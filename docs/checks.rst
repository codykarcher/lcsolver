Model Checks
============

Every solve runs a pipeline of checks around the solver itself: units are
validated first, the structure is classified, the stated problem is gated for
well-posedness, and -- after a converged solve -- the solution is examined for
the failure modes that look exactly like answers. Each check also has an
independent entry point for interactive use.

Units, first and batched
------------------------

``unit_corrector`` walks every objective and constraint and collects **every**
dimensional inconsistency before raising a single ``UnitMismatch`` naming them
all, with the correction for each. A model whose constraints do not balance
dimensionally has no meaning to solve for, so nothing downstream runs.

Structure detection
-------------------

Inside ``solve`` the detector is silent: it classifies the model (LP, QP, GP,
SP, or none) and the router acts on the result (:doc:`solvers`). Asked
independently, it explains itself::

    print(f.structure_report())

which states what class the model is and, more usefully, what stops it being a
simpler one -- "almost a GP except these constraints", naming them.

The pre-solve gate
------------------

Findings that make the stated problem ill-posed stop the solve **before any
solver runs**, batched into one :class:`~lcsolver.solvers.solver.PresolveError`
(code ``LC-E001``) that names every offender and the fix::

    PresolveError: [LC-E001] pre-solve check: 2 variables are not lower
    bounded (x, y) -- add bounds=[lo, hi] to the Variable or a constraint
    that limits them. ...

The error class is deliberately small: variables that appear in no constraint,
and variables unbounded above or below. Variables computed by a black-box
constraint are accounted for -- the algebraic checks cannot see grey-box rows,
but the gate can, so they do not trip it.

``diagnostics`` controls the gate: ``'error'`` (the default) raises;
``'warn'`` demotes to a ``RuntimeWarning`` (code ``LC-W101``); ``'print'``
prints the full report; ``'off'`` skips the checks.

Unbuilt submodels
-----------------

A :doc:`submodel <submodels>` builds when its last input is assigned, so one
whose inputs never all arrive posts no variables and no constraints. That is
not a solver error -- the formulation stays solvable and answers an easier
question than the one written down -- so ``solve`` refuses to run while one is
attached, raising a ``PresolveError`` with code ``LC-E003`` that lists every
unbuilt block and every missing input at once.

This check runs unconditionally, ahead of ``diagnostics``, because a
half-assembled model is not a weaker version of the problem: it is a different
problem, and it will return a confident number for it.
:func:`~lcsolver.presolve.reductions.unbuilt_blocks` asks the same question
without solving, and ``f.<block>.get_status()`` shows what any one block is
still waiting for.

Pre- and post-solve checks, split
---------------------------------

The check machinery has two halves with separate entry points::

    from lcsolver import presolve_check, postsolve_check

    report = presolve_check(f)     # needs nothing but the model
    solve(f)
    report = postsolve_check(f)    # needs a solved model; raises otherwise

``presolve_check`` covers structure classification, empty/unbounded/fixed/
output-only variables, foldable rows, rigidity and unopposed-variable
analysis. ``postsolve_check`` covers the checks that only mean something at a
solution:

* **degenerate** -- variables the optimum does not determine: moving them
  changes neither the objective nor feasibility;
* **cancelling** -- signomial terms contributing essentially nothing to their
  constraint, so the quantity they carry is disconnected;
* **at_floor** -- variables resting on the solver's positivity floor, pinned
  by the algorithm rather than the model (code ``LC-W304``).

``optimization_check(f)`` remains as the combined call: the structural half
always, plus the post-solve half when the model has been solved.

Automatic post-solve quality checks
-----------------------------------

After a converged solve, the post-solve half runs automatically and rides back
on the result::

    sol['quality']        # {'degenerate': [...], 'cancelling': [...], 'at_floor': [...]}
    sol['quality_text']   # the same, as a readable report

Only the floor check warns (it is a symptom of the algorithm, not the model);
the rest is informational. The holographic check (:doc:`holographic`) also
runs on every solve.
