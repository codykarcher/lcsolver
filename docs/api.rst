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

.. automodule:: lcsolver.objects.submodel
   :members:
   :undoc-members:
   :show-inheritance:


Black-box constraints
---------------------

.. automodule:: lcsolver.objects.blackBoxFunctionModel
   :members:
   :undoc-members:
   :show-inheritance:

:class:`~lcsolver.objects.blackBoxFunctionModel.BlackBoxFunctionModel_Variable`
is exported under three shorter aliases, because declaring inputs and outputs is
the one place its full name would be typed repeatedly. They are the same class,
not subclasses:

.. py:class:: lcsolver.BlackBoxVariable
.. py:class:: lcsolver.BBVariable
.. py:class:: lcsolver.BBV

   Aliases of
   :class:`~lcsolver.objects.blackBoxFunctionModel.BlackBoxFunctionModel_Variable`.


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

The model checks themselves live in :mod:`lcsolver.presolve.reductions`
alongside the presolve machinery, which is internal. Only the entry points are
public, so they are listed individually rather than by pulling in the module.

.. autofunction:: lcsolver.presolve.reductions.optimization_check
.. autofunction:: lcsolver.presolve.reductions.presolve_check
.. autofunction:: lcsolver.presolve.reductions.postsolve_check
.. autofunction:: lcsolver.presolve.reductions.structure_report
.. autofunction:: lcsolver.presolve.reductions.rigidity_report
.. autofunction:: lcsolver.presolve.reductions.rigidity_text
.. autofunction:: lcsolver.presolve.reductions.annihilated_report
.. autofunction:: lcsolver.presolve.reductions.unbuilt_blocks
.. autofunction:: lcsolver.presolve.reductions.unbuilt_blocks_check


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
