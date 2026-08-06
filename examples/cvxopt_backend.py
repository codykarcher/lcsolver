"""Three small models solved directly on the cvxopt backend.

``solve()`` picks a backend automatically; these call ``cvxopt_solve``
explicitly, which is the thing to do when you want the classic GP solver on
a small, well-scaled model. Each model is a two-liner chosen to exercise a
distinct piece of machinery:

1. a vector variable (``size=2``) with a monomial equality pinning one entry,
2. a scalar model with a monomial equality,
3. an objective with a squared term.
"""

from lcsolver.objects.formulation import Formulation
from lcsolver.solvers.solver import cvxopt_solve

# ----------------------------------------------------------------------
# 1. Vector variable; x[1] pinned by a monomial equality
# ----------------------------------------------------------------------
f = Formulation()
x = f.Variable(name='x', guess=5.0, units='', size=2, description='x variable')
f.Objective(x[0] ** 2)
f.ConstraintList([
    x[0] >= 2,
    x[1] == 1.0,
])
res = cvxopt_solve(f)
print('vector variable:   x =', [round(float(v), 6) for v in res['x']])

# ----------------------------------------------------------------------
# 2. Scalar model with an equality
# ----------------------------------------------------------------------
f = Formulation()
x = f.Variable(name='x', guess=5.0, units='', description='x variable')
y = f.Variable(name='y', guess=5.0, units='', description='y variable')
f.Objective(x ** 2)
f.ConstraintList([
    x >= 2,
    y == 5.0,
])
res = cvxopt_solve(f)
print('scalar equality:   x =', [round(float(v), 6) for v in res['x']])

# ----------------------------------------------------------------------
# 3. Squared term in the objective
# ----------------------------------------------------------------------
f = Formulation()
x = f.Variable(name='x', guess=1.0, units='', description='x variable')
y = f.Variable(name='y', guess=1.0, units='', description='y variable')
f.Objective(y ** 2 + x)
f.ConstraintList([
    y >= 4,
    x >= 1,
])
res = cvxopt_solve(f)
print('squared objective: x =', [round(float(v), 6) for v in res['x']])
