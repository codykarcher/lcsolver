Holographic Constraints
=======================

A holographic constraint is one that must hold but must **not** bind. It keeps
the problem well posed -- a ``1e-30 .. 1e30`` box, the edge of the data a fit
was made from, a numerical guard -- rather than shaping the answer. The danger
is that an active one silently invalidates the result: the solve converges,
the duals are finite, the table prints, and the optimum is sitting on a
boundary of the *model's validity* instead of the *design's*. The only defence
is to declare them in advance and check every time, so LCsolver does both.

Declaring
---------

::

    f.HolographicConstraint(x <= 1e6 * units.m)
    f.HolographicConstraintList([
        x <= cap * units.m,
        y <= cap * units.m,
    ])

A holographic constraint is imposed *exactly* as an ordinary constraint --
only LCsolver's bookkeeping differs. Groups forward the declaration, so a
sub-model declared under ``f.group('wing')`` participates normally.

The check
---------

It runs on **every** solve, with no flag and no opt-in. When any holographic
constraint is active at the solution:

* the message list gains an ``LC-W301`` entry naming each binding constraint
  with its symbolic body and numeric activity --
  ``constraint_2: x <= m (at 1, margin +8.80e-09)``;
* ``sol['holographic_active']`` carries the structured report
  (``[{name, operator, bound, value, margin, expr}]``);
* ``f.solution.holographic`` holds the same list, and the summary gains a
  dedicated section spelling out what an active one means.

Margins are **relative**, not absolute: a model spanning many orders of
magnitude cannot use absolute slack, so ``1e6`` against a bound of ``1e6``
registers as binding even though the absolute gap is huge.

Two footnotes worth knowing. A holographic **equality** is a contradiction in
terms -- an equality always binds -- and the report says so explicitly rather
than listing it as news every time. And the report reads the values currently
on the model, so it means what it says only after a solve has written them
back; ``solve`` is what calls it.

When it fires
-------------

An active holographic constraint is not an error, because the point returned
may still be useful -- but it is not an optimum of the problem you meant to
pose. The usual responses, in order of preference: extend the model (refit the
surrogate over a wider range, add the physics the box was standing in for), or
accept the boundary and promote the constraint to an ordinary one, stating the
limit as part of the design problem.
