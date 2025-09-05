# =================
# Import Statements
# =================
import numpy as np
import pyomo
from pyomo.contrib.edi import Formulation
from pyomo.environ import units


# ===================
# Declare Formulation
# ===================
f = Formulation()
N = f.Constant(name="N", value=5, units="-", description="# of panels")
N = N.value
# N formulated here for use in the following declarations

# =================
# Declare Variables
# =================
#D_i            = f.Variable(name="D_i",            guess=10.0,    units="N",              description="induced drag")
delta_phi      = f.Variable(name="delta_phi",      guess=1.0,    units="-",   size = N,     description="potential difference")
delta_s        = f.Variable(name="delta_s",        guess=1.0,    units="m",   size = N,     description="length along wake")
#denom          = f.Variable(name="denom",          guess=1.0,    units="m^2",               description="denominator of nabla_phi step")
#denom_store    = f.Variable(name="denom_store",    guess=1.0,    units="m^2",   size = N,     description="denominator of nabla_phi step, stored for each individual loop")
Gamma          = f.Variable(name="Gamma",          guess=1.0,    units="-",   size = N + 1, description="Lifting Line circulation")
L_dist         = f.Variable(name="L_dist",         guess=1.0,    units="N",   size = N,     description="lift distribution")
L_dist_sum     = f.Variable(name="L_dist_sum",     guess=5.0,    units="N",               description="lift distribution sum")
#n              = f.Variable(name="n",              guess=1.0,    units="-",  size = N,     description="normal vector")
#n_hat          = f.Variable(name="n_hat",          guess=1.0,    units="-",  size = N,     description="normal unit vector")
#sum_ij           = f.Variable(name="sum_ij",           guess=1.0,    units="m^2", size = N,     description="posynomial term of gradient representation sum")
W              = f.Variable(name="W",              guess=1.0,    units="N",               description="weight")
y_ij           = f.Variable(name="y_ij",           guess=1.0,    units="m^2", size = N,     description="posynomial term of y gradient representation")
y_mid          = f.Variable(name="y_mid",          guess=1.0,    units="m",   size = N,     description="spanwise coordinate at panel center")
#z_ij          = f.Variable(name="z_ij",          guess=1.0,    units="m^2",   size = N,     description="posynomial term of z gradient representation")
z_mid          = f.Variable(name="z_mid",          guess=1.0,    units="m",   size = N,     description="vertical coordinate at panel center")

# Panel coordinate setup  
y     = np.array([0, 3, 6, 9, 12, 15])  # array of y-coordinates, preselected, panel edgepoints
z     = np.array([1.5, 1, 1, 1, 1, 1.5])    # numpy array of z-coordinates, preselected for example

# =================
# Declare Constants
# =================
epsilon    = f.Constant(name="epsilon",    value=1e-10,   units="m",       description="small constant to avoid denom of 0")

# =====================
# Declare the Objective
# =====================
f.Objective(L_dist_sum)

# =======================
# Declare the Constraints
# =======================
Constraints = []

# ==============
# Parasitic Drag
# ==============

#delta_phi == [1, 1, 1, 1, 1]
for i in range(N):
    Constraints += [
                    L_dist[i] == delta_phi[i] * (y[i + 1] - y[i]), 
                    delta_phi[i] <= 11
                    ]
    
Constraints += [
                L_dist_sum == sum(L_dist),
                L_dist_sum == W
                ]

for i in range(N): 
    Constraints += [
                    delta_s[i] >= ((y[i + 1] - y[i])**2 + (z[i + 1] - z[i])**2)**0.5,
                    y_mid[i] >= (y[i + 1] + y[i]) / 2,
                    z_mid[i] >= (z[i + 1] + z[i]) / 2
                    ]

# ============
# Induced Drag
# ============

for i in range(N + 1):
   Constraints += [
                   Gamma[i] <= 10,
                   # n_hat[i] == n[i] / (n[i]**0.5)**2,  # Ensure no division by zero if n[i] could be 0
                   # this might not be needed at all? components are used later
                    ]

Constraints += [
                Gamma[0] <= -delta_phi[0]   # Boundary condition
                ]   

for i in range(1, N - 1):
   Constraints += [
                   Gamma[i + 1] <= delta_phi[i - 1] - delta_phi[i]
                   # Gamma is same size as y and z, N + 1; n_hat is size N
                   ]
# print(Gamma.values)
Constraints += [Gamma[N] <= delta_phi[N - 1]]    # Boundary condition

# works up to here though lift distribution values are sus


# Gradient Representation 
for i in range(N): 
    for j in range(N): 
        Constraints += [
                        y_ij[j] == y_mid[i]**2 - 2*y_mid[i]*y[j] + y[j]**2,  
                        ]



# ======================
# Optimizing with Solver
# ======================

f.ConstraintList(Constraints)

var_list = f.get_variables()
    
f.pprint()
from solver import cvxopt_solve
res = cvxopt_solve(f)
print(res)


# ================
# Printing Results
# ================

#Using pandas library to print res as a dataframe
import pandas as pd

# Initialize lists to store variable details
variable_names = []
variable_values = []
variable_units = []
variable_descriptions = []

ctr = 0  # Counter for results array index

for var in var_list:
    if isinstance(var, pyomo.core.base.var.IndexedVar):
        # Handle indexed variables with multiple entries
        for ix in var.index_set():
            lbls = ["1", "2", "3", "4", "5", "6"]
            variable_names.append(f"{var.name}[{lbls[ix]}]") # setting labels to index values
            variable_values.append(res['x'][ctr])            # listing values, units, descriptions
            variable_units.append(var._units)
            variable_descriptions.append(var.doc) 
            ctr += 1
    else:
        # Handle non-indexed variables
        variable_names.append(var.name)
        variable_values.append(res['x'][ctr]) # listing values, units, descriptions
        variable_units.append(var._units)
        variable_descriptions.append(var.doc)  
        ctr += 1

# Create dataframe with the details
pd.set_option('display.max_rows', None)           # display all variable rows
pd.options.display.float_format = '{:.4f}'.format # set output to 4 decimal places
df = pd.DataFrame({
    "Variable Name": variable_names,
    "Value": variable_values,
    "Units": variable_units,
    "Description": variable_descriptions
})

print(df)
