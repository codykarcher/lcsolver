# import time
# t = time.time()
import numpy as np
pi = np.pi
# from corsair import units
# from corsair.optimization.variable import Variable
# from corsair.optimization import Formulation, Constant

import numpy as np
import pyomo
from pyomo.contrib.edi import Formulation
from pyomo.environ import units
f = Formulation()


# Free Variables
x = f.Variable(name = "x", guess = 0.0, units = "m", description = "x position")
y = f.Variable(name = "y", guess = 0.0, units = "m", description = "y position")

constraints = []

# and the rest of the models
constraints += [y <= -x**2,
                x>=1]



f.ConstraintList(constraints)
f.Objective(y)

var_list = f.get_variables()
    
# f.pprint()
from solver import cvxopt_solve
res = cvxopt_solve(f)
# print(res)

# import pyomo.environ as pyo
# opt = pyo.SolverFactory('ipopt')
# opt.solve(f)
#Uncomment only if solving with ipopt

var_list = f.get_variables()
ctr = 0
for var in var_list:
    #print(type(var))
    if isinstance(var,pyomo.core.base.var.IndexedVar):
        # print('%s:  %s'%(var.name,var._units))
        # print(var[0].value)
        # print(var.index_set())
        for ix in var.index_set():
            lbls = ["out","ret","sprint"]
            print('%s[%s]:  %.4f  %s'%(var.name,lbls[ix],res['x'][ctr],var._units))
            ctr += 1
    else:
        print('%s:  %.4f  %s'%(var.name,res['x'][ctr],var._units))
        ctr += 1

