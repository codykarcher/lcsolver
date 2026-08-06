"""Unit-aware modeling: mixed units reconciled by the unit corrector.

Two models. The first is a toy whose variables deliberately carry different
dimensions (m^2, m) so the constraints only balance with explicit
conversions. The second is the Hoburg aircraft sizing GP of
``aircraft_gp.py`` with one change: the wing area is declared in **ft^2**
while every constant stays SI. The answer must not change -- exercising the
unit corrector on a real model is the entire point of the file.
"""

import numpy as np
import pyomo.environ as pyo
from pyomo.environ import units

from lcsolver.objects.formulation import Formulation
from lcsolver.solvers.solver import cvxopt_solve

# ----------------------------------------------------------------------
# 1. A toy with mixed dimensions
# ----------------------------------------------------------------------
f = Formulation()
x = f.Variable(name='x', guess=1.0, units='m^2', description='an area')
y = f.Variable(name='y', guess=1.0, units='m', description='a length')
z = f.Variable(name='z', guess=1.0, units='m', description='a length')
f.Objective(z)
f.ConstraintList([
    z >= x / units.m + y,
    x >= 1 * units.m ** 2,
    y >= 0.5 * units.m,
])
res = cvxopt_solve(f)
print('toy mixed units:  [x, y, z] =', [round(float(v), 6) for v in res['x']])

# ----------------------------------------------------------------------
# 2. The aircraft GP with the wing area in ft^2
# ----------------------------------------------------------------------
f = Formulation()

A   = f.Variable(name='A',   guess=10.0,    units='-',    description='aspect ratio')
C_D = f.Variable(name='C_D', guess=0.025,   units='-',    description='Drag coefficient of wing')
C_f = f.Variable(name='C_f', guess=0.003,   units='-',    description='skin friction coefficient')
C_L = f.Variable(name='C_L', guess=0.5,     units='-',    description='Lift coefficient of wing')
D   = f.Variable(name='D',   guess=300,     units='N',    description='total drag force')
Re  = f.Variable(name='Re',  guess=3e6,     units='-',    description="Reynold's number")
S   = f.Variable(name='S',   guess=110.0,   units='ft^2', description='total wing area, in feet')
V   = f.Variable(name='V',   guess=30.0,    units='m/s',  description='cruising speed')
W   = f.Variable(name='W',   guess=10000.0, units='N',    description='total aircraft weight')
W_w = f.Variable(name='W_w', guess=2500,    units='N',    description='wing weight')

C_Lmax     = f.Constant(name='C_Lmax', value=2.0,     units='-',      description='max CL with flaps down')
CDA0       = f.Constant(name='CDA0',   value=0.0306,  units='m^2',    description='fuselage drag area')
e          = f.Constant(name='e',      value=0.96,    units='-',      description='Oswald efficiency factor')
k          = f.Constant(name='k',      value=1.2,     units='-',      description='form factor')
mu         = f.Constant(name='mu',     value=1.78e-5, units='kg/m/s', description='viscosity of air')
N_ult      = f.Constant(name='N_ult',  value=2.5,     units='-',      description='ultimate load factor')
rho        = f.Constant(name='rho',    value=1.23,    units='kg/m^3', description='density of air')
S_wetratio = f.Constant(name='Srat',   value=2.05,    units='-',      description='wetted area ratio')
tau        = f.Constant(name='tau',    value=0.12,    units='-',      description='airfoil thickness to chord ratio')
V_min      = f.Constant(name='V_min',  value=22,      units='m/s',    description='takeoff speed')
W_0        = f.Constant(name='W_0',    value=4940.0,  units='N',      description='aircraft weight excluding wing')
W_W_coeff1 = f.Constant(name='W_c1',   value=8.71e-5, units='1/m',    description='Wing Weight Coefficient 1')
W_W_coeff2 = f.Constant(name='W_c2',   value=45.24,   units='Pa',     description='Wing Weight Coefficient 2')

f.Objective(D)

pi = np.pi
C_D_fuse = CDA0 / S
C_D_wpar = k * C_f * S_wetratio
C_D_ind = C_L ** 2 / (pi * A * e)
W_w_strc = W_W_coeff1 * (N_ult * A ** 1.5 * (W_0 * W * S) ** 0.5) / tau
W_w_surf = W_W_coeff2 * S

f.ConstraintList([
    C_D >= C_D_fuse + C_D_wpar + C_D_ind,
    W_w >= W_w_surf + W_w_strc,
    D >= 0.5 * rho * S * C_D * V ** 2,
    Re == (rho / mu) * V * (S / A) ** 0.5,
    C_f == 0.074 / Re ** 0.2,
    W == 0.5 * rho * S * C_L * V ** 2,
    W == 0.5 * rho * S * C_Lmax * V_min ** 2,
    W >= W_0 + W_w,
])

res = cvxopt_solve(f)
print('aircraft, S in ft^2:  D = %.4f N,  S = %.4f ft^2'
      % (pyo.value(f.D), pyo.value(f.S)))
