Constraints
===========


Overview
--------

Constraints are the mathematical representation of rules that are imposed on your decisions/variables when minimizing or maximizing.  In engineering design, constraints are often imposed by physics or operational limits.

The Constraint constructor is a very thin wrapper on pyomo ``Constraint``, and so experienced pyomo users will not see any significant differences from base pyomo.  


Construction
------------

Constraints are constructed by creating an instance of a new constraint in a LCsolver Formulation 

.. py:function:: f.Constraint(expr)

    Declares a constraint in a pyomo.lcsolver.formulation

   :param expr: The expression representing the constraint
   :type  expr: pyomo expression

   :return: None
   :rtype: None


However, the expected use case is the ``f.ConstraintList()`` function:

.. py:function:: f.ConstraintList(conList)

    Declares new constraints in a pyomo.lcsolver.formulation from a list of inputs

   :param conList: The list of constraints to be generated.  Entries will be pyomo expressions, or lists/tuples/dicts that are used to create RuntimeConstraints (see :doc:`here <./blackboxconstraints>`)
   :type  conList: list

   :return: None
   :rtype: None


.. py:function:: f.ConstraintDict(conDict)

    Declares new constraints from a dict of ``name: constraint`` pairs, so each row can be found by name afterwards

   :param conDict: The constraints to be generated, keyed by a user name.  Values are what ``ConstraintList`` accepts: a pyomo expression, a vector comparison (one constraint per element, the name mapping to all of them), or a list/dict that creates a RuntimeConstraint (see :doc:`here <./blackboxconstraints>`)
   :type  conDict: dict

   :return: None
   :rtype: None


Relation to Pyomo Constraint
----------------------------

The LCsolver constraint constructor is essentially a direct pass through to base pyomo.  Constraints will be added to the ``pyomo.ConcreteModel`` in increasing order with key ``constraint_###`` where the the index of the objective appears after the underscore.  First constraint is labeled as ``constraint_1``, and constraint names are never padded with zeros.  RuntimeConstraints also contribute to this counter.


Naming constraints
------------------

A constraint can carry a user name as well as its ``constraint_###`` key, through ``assignedName`` on ``Constraint``, ``HolographicConstraint`` and ``RuntimeConstraint``, or by declaring with ``ConstraintDict`` (``HolographicConstraintDict``) instead of a list::

    f.ConstraintDict({
        "lift": W == 0.5 * rho * S * C_L * V**2,
        "drag": D >= 0.5 * rho * S * C_D * V**2,
        "cap":  x <= x_max,                        # a vector row: one constraint per element
    })

    f.constraint("lift")          # the pyomo Constraint
    f.constraint("cap")           # the list of per-element Constraints
    f.constraint_names()          # {"lift": "constraint_1", "drag": "constraint_2", "cap": ["constraint_3", ...]}

The pyomo component keeps its ``constraint_###`` key; the user name is a map on the side.  That is deliberate: everything that walks the model (unit correction, structure detection, the solvers) is unchanged, and unnamed rows keep the same numbers they always had.  Names must be unique within a formulation, and a name declared through a group carries the group prefix, the way a group's Variable does (``wing.ConstraintDict({"lift": ...})`` registers ``wing_lift``).  A lookup of an unknown name raises ``KeyError`` listing the names that exist.


Examples
--------

A standard declaration statement
++++++++++++++++++++++++++++++++

.. literalinclude:: ../tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Constraints_Snippet_01
    :end-before: # END: Constraints_Snippet_01


With the core constructor
+++++++++++++++++++++++++

.. literalinclude:: ../tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Constraints_Snippet_02
    :end-before: # END: Constraints_Snippet_02

Using indexed variables and constants
+++++++++++++++++++++++++++++++++++++

.. literalinclude:: ../tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Constraints_Snippet_03
    :end-before: # END: Constraints_Snippet_03

Tips
----

* For the typical user, constraints should always be constructed using the ``f.ConstraintList()`` function to produce a cleaner input file that is easier to modify


Known Issues
------------

* Indexed variables must be broken up using either indices or a pyomo rule (see `this issue <https://github.com/codykarcher/pyomo/issues/3>`__)
* Units that are inconsistent, but not the same (ie, meters and feet) will flag as invalid when checking units (see `this issue <https://github.com/codykarcher/pyomo/issues/6>`__)
