API Reference
=============

The pages in the User's Guide describe how to *use* each part of LCsolver. This
page is the generated reference: every public class and function, with its
signature and docstring, taken directly from the source.

If you are looking for one thing in particular, the entries most people want are
:class:`~lcsolver.objects.formulation.Formulation` (declaring a model),
:class:`~lcsolver.objects.blackBoxFunctionModel.BlackBoxFunctionModel` (wrapping
an analysis code), and :func:`~lcsolver.solvers.solver.solve` (solving one).


The package namespace
---------------------

Everything listed in ``lcsolver.__all__`` is importable directly from the top
level, so ``from lcsolver import Formulation, solve`` is the expected form.

Members are documented below under the module that defines them, rather than
twice. Documenting them here as well would give every class two targets, and a
cross-reference to ``Formulation`` would then be ambiguous.

.. automodule:: lcsolver
   :no-members:


Building a model
----------------

.. automodule:: lcsolver.objects.formulation
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: EDIVar, EDIParam

.. automodule:: lcsolver.objects.vector
   :members:
   :undoc-members:
   :show-inheritance:


Black-box constraints
---------------------

.. automodule:: lcsolver.objects.blackBoxFunctionModel
   :members:
   :undoc-members:
   :show-inheritance:


Solving
-------

.. automodule:: lcsolver.solvers.solver
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: lcsolver.solvers.sequential.bridge
   :members:
   :undoc-members:
   :show-inheritance:


Checks run before the solve
---------------------------

.. automodule:: lcsolver.presolve.unitCorrector
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: lcsolver.presolve.structureDetector
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: lcsolver.presolve.feasibilityCheck
   :members:
   :undoc-members:
   :show-inheritance:


The solution, and the checks run after the solve
------------------------------------------------

.. automodule:: lcsolver.objects.solution
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: lcsolver.postsolve.sensitivity
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: lcsolver.postsolve.holographic
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: lcsolver.postsolve.writeback
   :members:
   :undoc-members:
   :show-inheritance:


The solver environment
----------------------

What IPOPT this installation will actually use, and how to get one.

.. automodule:: lcsolver.environment
   :members:
   :undoc-members:
   :show-inheritance:
