Submodels
=========

Overview
--------
A ``SubModel`` is a reusable piece of a model: its variables, its constants,
and the constraints that tie them together, packaged in one object. A sizing
problem written as one flat script becomes unreadable somewhere around a
hundred rows; written as submodels it becomes an assembly of named blocks, each
of which can be read, tested, and reused on its own.

Everything a submodel declares is namespaced by the name it is attached under,
so the attribute path, the flat component name, the deck key, and the
sensitivity row all read the same::

    f.rotor_model.chord          # the handle
    f.rotor_model_chord          # the component, and the deck key


Attaching a submodel
--------------------

A submodel is attached to a formulation by **assignment**, and that assignment
is what hands it both the formulation and its name:

.. literalinclude:: ../tests/test_docSnippets.py
    :language: python
    :dedent: 8
    :start-after: # BEGIN: SubModels_Snippet_01
    :end-before: # END: SubModels_Snippet_01

The formulation is not passed to the constructor and the name is not spelled
out as a string. There is one place the name appears, so there is no way for
the attribute and the namespace to drift apart.

The constructor takes only the **settings** -- the things that change which
rows exist, like ``n_tanks`` above. They are available in ``self.settings``,
and each is also set as an attribute.


Inputs are assigned, not passed
-------------------------------

A block with thirty inputs makes a thirty-argument call that no reader can
check against the signature. Instead, a submodel declares what it needs and the
inputs arrive one per line::

    f.tank_model.weight_gross = W
    f.tank_model.density_fuel = rho

Each line names the input on the left and the assembly's quantity on the right,
so a mismatch is visible where it happens rather than as a position in an
argument list.

What a block needs is declared in two lists:

.. py:attribute:: SubModel.input_variables

    Names of input **design variables** -- quantities the optimizer moves. The
    block writes rows that shape them and reads solved values back.

.. py:attribute:: SubModel.input_constants

    Names of input **constants** -- quantities the assembly fixed. Each appears
    in a deck, earns a sensitivity, and nothing the block does can change it.

The two are kept apart because they are different promises to the assembly. A
quantity handed in under the wrong heading still solves, which is exactly why
it is worth stating and printing separately.

.. py:attribute:: SubModel.inputs

    Inputs whose kind is not being declared. Prefer the two lists above; this
    exists so a block written before they did keeps working.


All of the inputs, or none of them
----------------------------------

Inputs may also be passed to the constructor -- a four-input block reads fine
that way::

    f.tank_model = TankModel(n_tanks=2, weight_gross=W, density_fuel=rho)

but it is **all of them or none**. A call that names some inputs and not others
raises a ``TypeError`` naming every missing one::

    TankModel was given 1 of its 2 inputs. A block takes either ALL of its
    inputs in the constructor or NONE of them, assigned one per line
    afterwards. A partial call reads like a complete one, builds nothing,
    and posts no rows.

      1 input(s) are missing:
        input constants:
          density_fuel

      Either add the missing ones to the call, or drop the 1 that
      are there and assign every input after attaching the block:

          f.<name> = TankModel(<settings only>)
          f.<name>.density_fuel = ...

A partial call is the one reading that cannot be right. It looks like a
complete construction, so nothing about it suggests more is coming, and the
block sits there posting no rows until ``solve`` refuses it.

Settings are not part of this rule -- passing them is what the constructor is
for, and they are told apart from inputs by name.


The block builds when its last input arrives
--------------------------------------------

Assigning the final input calls ``build()``. Nothing is posted before then, so
a block that never receives an input posts **no rows at all**.

That is not a solver error. The formulation stays solvable and answers a
different, easier question than the one written down. So ``solve`` refuses to
run a formulation with an unbuilt block attached, raising a
:class:`~lcsolver.solvers.solver.PresolveError` (code ``LC-E003``) that lists
every unbuilt block and every missing input at once::

    PresolveError: [LC-E003] 1 model block(s) never received all their
    inputs, so they posted no variables and no constraints. ...

      tank_model is still waiting for 1 input(s):
        density_fuel

      Nothing about this is caught downstream. The rows those blocks
      would have posted are simply absent, so the solve succeeds and
      reports an optimum for a problem missing whole missions.
      Assign the inputs listed above, then solve again.

      f.<block>.get_status() prints every input a block needs and
      what each one is currently connected to.

The same list is available without solving, through
:func:`~lcsolver.presolve.reductions.unbuilt_blocks` and
:func:`~lcsolver.presolve.reductions.unbuilt_blocks_check`.


Looking at a block
------------------

``get_status()`` is the way to see what a block needs, what is wired to each of
those, what it declares, and what it hands back:

.. literalinclude:: ../tests/test_docSnippets.py
    :language: python
    :dedent: 8
    :start-after: # BEGIN: SubModels_Snippet_02
    :end-before: # END: SubModels_Snippet_02

::

    TankModel 'tank_model'  --  NOT BUILT, waiting on 1 input. It has posted nothing.
    ============================================================================

    Settings
       n_tanks                      = 2

    Input variables (1 of 1 connected)
       weight_gross                 <- W [N]

    Input constants (0 of 1 connected)
       density_fuel                 <- NOT CONNECTED

    Variables declared here: none yet, the block has not built

    Constants declared here: none yet, the block has not built

Once every input is connected the block builds, and the same call reports what
it declared, with sizes, units, and descriptions::

    TankModel 'tank_model'  --  BUILT, 2 constraint statements
    ==========================================================

    Settings
       n_tanks                      = 2

    Input variables (1 of 1 connected)
       weight_gross                 <- W [N]

    Input constants (1 of 1 connected)
       density_fuel                 <- rho_fuel [N/m**3]

    Variables declared here (2)
       volume            tank_model_volume [2] [m**3] -- volume of each tank
       weight            tank_model_weight [N] -- fuel carried

An input is reported by the assembly's name for the quantity connected to it,
so a block wired to the wrong handle says so. ``get_status()`` prints;
``status_text()`` returns the same report as a string.


What a block hands back
-----------------------

A submodel's declared variables and constants are visible in the component
lists and the sensitivity table. An **expression** a block exposes as an
attribute is not -- a weight rollup, a power sum -- and nothing but a docstring
would tell a reader it exists.

.. py:attribute:: SubModel.provides

    ``{name: what a caller does with it}``. Each entry is reported by
    ``get_status()`` under *Also provides*, with whether it is available yet.

.. literalinclude:: ../tests/test_docSnippets.py
    :language: python
    :dedent: 8
    :start-after: # BEGIN: SubModels_Snippet_03
    :end-before: # END: SubModels_Snippet_03

::

    Also provides
       component_weights            <expression> [N]
                                    The sum of the group weights this block
                                    owns. The assembly writes the empty-weight
                                    row itself, so that row stays in one place
                                    across every configuration.

Handing back an expression rather than a variable is a deliberate choice: it
adds no row and no degree of freedom, and it lets the assembly write the one
statement that uses it. The cost is that it is invisible, which is what
``provides`` repairs.


Declaring inside a block
------------------------

``self.Variable``, ``self.Constant``, and ``self.ConstraintList`` mirror the
formulation's, with two differences: everything is namespaced by the block's
name, and every declared handle is also set as an attribute of the block, so
its own rows and its caller reach quantities by name rather than by dict key.

.. py:function:: self.Variable(name, guess, units, description, **kw)

    Declares a variable in this block's group. Also recorded in
    ``self.variables`` and set as ``self.<name>``.

.. py:function:: self.Constant(name, value, units, description, **kw)

    Declares a constant in this block's group. Also recorded in
    ``self.constants`` and set as ``self.<name>``.

.. py:function:: self.ConstraintList(rows)

    Posts this block's rows. ``None`` entries are dropped, so a row that exists
    only under some setting can be written inline as a conditional.

A block declares only what it **owns**. A quantity two blocks share -- a blade
count, a root cutout -- is declared once by the assembly and the handle is
assigned in, so there is one constant and one deck key for it rather than two
that can drift apart.


Two helpers a block will want
-----------------------------

.. py:function:: f.scalar_sum(parts)

    Add scalar quantities, refusing anything vector-valued.

    A rollup -- a weight statement, a power budget -- is a scalar sum by
    construction. Both Python's ``sum`` and ``f.sum`` accept a vector among the
    parts and quietly return an **array**, which downstream becomes one
    constraint per element instead of one row: a different model that still
    solves. This says the intent instead, and the seed is the first part rather
    than a dimensionless zero, so units are checked on every addition rather
    than against a bare ``0``.

.. py:function:: f.retype_to_float(x)

    The float behind a number or a declared ``Constant``.

    For the places that need a **value** at build time rather than a symbol in
    a row: a station guess grid, a branch on whether a term exists at all, and
    an exponent on a **dimensional** base.

    An exponent may otherwise stay symbolic -- a ``Constant`` raised over a
    dimensionless base is legal, keeps the GP structure, and is the only way to
    get a sensitivity to a fit exponent (:doc:`constants`). Pass it through
    here when you want the number instead: over a dimensional base a symbolic
    exponent is refused, because the units of the result would then depend on a
    value the deck can change.

.. literalinclude:: ../tests/test_docSnippets.py
    :language: python
    :dedent: 8
    :start-after: # BEGIN: SubModels_Snippet_04
    :end-before: # END: SubModels_Snippet_04

Both are also available on a group, as ``f.group(...).scalar_sum`` and
``.retype_to_float``.


Passing the formulation explicitly
----------------------------------

``Block(f, 'name', **settings)`` still works, and builds the block immediately
if it declares no inputs. It exists so a model library written before the
assignment syntax keeps working and can be migrated one block at a time; new
blocks should be attached by assignment.
