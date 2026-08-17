Constants
=========

Overview
--------
Constants are a key mechanism used to capture the relationship between variables.  In engineering design, constants are often defined by physics or operational limits.

The Constant constructor is a very thin wrapper on pyomo ``Param``, and so experienced pyomo users will not see any significant differences from base pyomo.  


Construction
------------

Constants are constructed by 1) creating an instance of a new parameter in a LCsolver Formulation and 2) passing out this newly constructed parameter to be used in objective and constraint construction.  

.. py:function:: f.Constant(name, value, units, description='', size=None, within=None)

    Declares a constant in a pyomo.lcsolver.formulation

   :param name: The name of the constant for the purposes of tracking in the formulation.  Commonly, this will be the same as the constant name in local namespace.
   :type  name: str
   :param value: The value of the constant.  For scalar constants, this should be a valid float or int for the specified domain.  For vector, matrix and tensor constants, a single float or int is given to every element; a nested list or a numpy array of the declared shape is laid out onto the index set with the first index outermost, so ``value[i][j]`` is element ``(i, j)``; and a dictionary of index-value pairs is accepted as in accordance with base pyomo.  An array whose shape does not match ``size`` is refused rather than reshaped, because a transposed array is a different model and would solve without complaint.
   :type  value: float or int or dict
   :param units: The units of the constant.  Every entry in a vector constant must have the same units.  Entries of '', ' ', '-', 'None', and 'dimensionless' all become units.dimensionless
   :type  units: str or pyomo.core.base.units_container._PyomoUnit
   :param description: A description of the constant
   :type  description: str
   :param size: The size (or shape) of the constant.  Entries of 0, 1, and None all correspond to scalar constants.  Other integers correspond to vector constants.  Matrix and tensor constants are declared using lists of ints, ex: [10,10].  Matrix and tensor constants with a dimension of 1 (ie, [10,10,1]) will be rejected as the extra dimension holds no meaningful value.  
   :type  size: int or list
   :param within: The domain of the constant (ex: Reals, Integers, etc).  Default of None constructs a constant in Reals.  This option should rarely be used.
   :type  within: pyomo set

   :return: The constant that was declared in the formulation
   :rtype: pyomo.core.base.param.ScalarParam or pyomo.core.base.param.IndexedParam


Relation to Pyomo Param
-----------------------

The fields: name and within, and bounds are directly passed to the pyomo ``Param`` constructor, with some minor checking.  The value field is passed to the ``Param`` initialize field.  The description field is passed to the doc field in the pyomo ``Param``.  Units are passed directly with an additional check.  All Constants set the pyomo ``Param`` mutable field to True.

Non-scalar constants are constructed using pyomo ``Sets``.  Sets are constructed to be integer sets that fill the entire interval from lower bound to upper bound, ie a vector constant of length 5 would create a pyomo ``Set`` with valid indices [0,1,2,3,4] with no skips.  In this way, non-scalar constatants are slightly less flexible than general non-scalar pyomo ``Param``.


Setting constants after the build
---------------------------------

Every Constant is a **mutable** ``Param``, so a model is built once with its
defaults declared inline and an input deck is applied to the built model
afterwards::

    f = build_my_model()
    f.load_constants({'weight_payload': 600.0, 'v_cruise': 62.0})
    result = lcsolver.solve(f)

Nothing is reconstructed, so a sweep is a loop over loads and solves rather
than a configuration dictionary threaded through the constructor and a rebuild
per case. ``load_constants`` returns the formulation, so it chains onto a
build.

Names are the ones ``Constant`` was called with, group prefixes included --
``'wing_area'`` for a constant declared on ``f.group('wing')``. A name that is
not a declared constant raises ``KeyError`` with near-misses suggested, rather
than being ignored, because a deck key that silently does nothing is a model
that silently sizes the wrong thing. A Constant declared with ``size`` takes a
sequence of that length.

One consequence worth knowing if you pass ``structures=`` to ``solve`` to skip
re-detecting: a detected structure carries a *clone* of the model, frozen at
the constant values it was detected with. Loading a deck bumps the model's
revision and ``solve`` then refuses those structures rather than answering the
previous deck's question with the current deck's label. Re-detect after
loading.


A Constant as an exponent
-------------------------

A Constant may be used as an **exponent**, not only as a coefficient::

    e = f.Constant(name='e', value=0.49, units='-', description='fit exponent')
    f.ConstraintList([x * y**e >= 1.0 * units.dimensionless])

The model is still a geometric program, and the dual prices the exponent like
any other constant -- which is the point, since an exponent that came from a
curve fit is exactly the kind of assumption worth ranking (:doc:`sensitivities`).
A deck may then sweep it with ``load_constants`` without rebuilding.

The base must be **dimensionless**. Were it dimensional, the units of the
result would depend on a number the deck can change, so the model would not
have fixed units at all; that is refused with a message saying so rather than
left to surface as a units error deeper down. Divide by a reference quantity to
make the base dimensionless, or pass ``f.retype_to_float(e)`` if the exponent
really is a fixed number (:doc:`submodels`).


Examples
--------


A standard declaration statement
++++++++++++++++++++++++++++++++

.. literalinclude:: ../tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Constants_Snippet_01
    :end-before: # END: Constants_Snippet_01


Shortest possible declaration
+++++++++++++++++++++++++++++

.. literalinclude:: ../tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Constants_Snippet_02
    :end-before: # END: Constants_Snippet_02


An alternative units definition
+++++++++++++++++++++++++++++++

.. literalinclude:: ../tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Constants_Snippet_03
    :end-before: # END: Constants_Snippet_03


A vector constant
+++++++++++++++++

.. literalinclude:: ../tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Constants_Snippet_04
    :end-before: # END: Constants_Snippet_04


A matrix/tensor constant
++++++++++++++++++++++++

.. literalinclude:: ../tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Constants_Snippet_05
    :end-before: # END: Constants_Snippet_05


More complicated units definition
+++++++++++++++++++++++++++++++++

.. literalinclude:: ../tests/test_docSnippets.py
    :language: python 
    :dedent: 8
    :start-after: # BEGIN: Constants_Snippet_06
    :end-before: # END: Constants_Snippet_06


Tips
----

* Declare constants in alphabetical order.  Trust me.  It's a pain at first, but it saves a huge amount of time down the road, especially for large models.
* Designate a section in your file for constant declarations, as is done in the :doc:`introductory example <./quickstart>`
* Align all of your constant declarations in a pretty, grid like fashion.  Depending on preference, these may or may not line up with variable declarations (I usually do not bother with this)
* Use the keyword names during constant declarations.  Takes extra space, but is a massive boost to readability and intrepretability
* Declare one constant on one single line with no breaks, no matter what style guides tell you.  Again, this is a significant boost to readability
* Do not skimp out on the description field, it is extremely helpful
