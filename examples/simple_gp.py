"""A two-variable geometric program, SLCP paper Equation 16.

.. math::

    \\begin{aligned}
    \\underset{x,y}{\\text{minimize}} \\quad
      & x^{-0.1} + 15x^{0.01} + y^{-0.1} + 15y^{0.01} \\\\
    \\text{subject to} \\quad
      & 0.01x^{-1.1} + x^{0.1} + y \\le 1
    \\end{aligned}

Small enough to check by hand, but genuinely a GP: the optimum
f* = 31.8115934 sits at (0.0593208, 0.0224878).
"""

from pyomo.environ import units

from lcsolver.objects.formulation import Formulation
from lcsolver.solvers.solver import solve

# =====================
# Declare the Variables
# =====================
f = Formulation()
f.Variable(name='x', guess=0.3, units='', description='x')
f.Variable(name='y', guess=0.05, units='', description='y')

# =======================================
# Declare the Objective and Constraint
# =======================================
f.Objective(f.x ** -0.1 + 15 * f.x ** 0.01
            + f.y ** -0.1 + 15 * f.y ** 0.01)
f.ConstraintList([
    0.01 * f.x ** -1.1 + f.x ** 0.1 + f.y <= 1.0 * units.dimensionless,
])

# =====
# Solve
# =====
solve(f)
print(f.solution)
