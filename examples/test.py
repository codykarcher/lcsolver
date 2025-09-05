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

# =================
# Declare Variables
# =================
A        = f.Variable(name="A",        guess=14.0,   units="-",           description="aspect ratio")
C_D      = f.Variable(name="C_D",      guess=0.005,  units="-",   size=3, description="Wing drag coefficient")
C_Dfus   = f.Variable(name="C_Dfus",   guess=0.0001, units="-",   size=3, description="fuselage drag coefficient")
C_Di     = f.Variable(name="C_Di",     guess=0.0001, units="-",   size=3, description="induced drag coefficient")
C_Dp     = f.Variable(name="C_Dp",     guess=0.0001, units="-",   size=3, description="Profile drag coefficient")
C_L      = f.Variable(name="C_L",      guess=0.5,    units="-",   size=3, description="Wing lift coefficient")
D        = f.Variable(name="D",        guess=600,    units="N",   size=3, description="total drag force")
eta_0    = f.Variable(name="eta_0",    guess=0.3,    units="-",   size=3, description="overall efficiency")
eta_i    = f.Variable(name="eta_i",    guess=1,      units="-",   size=3, description="inviscid propeller efficiency")
eta_prop = f.Variable(name="eta_prop", guess=0.8,    units="-",   size=3, description="propeller efficiency")
I_cap    = f.Variable(name="I_cap",    guess=1e-5,   units="-",           description="area moment of inertia per unit chord")
M_barr   = f.Variable(name="M_barr",   guess=3000,   units="N",           description="M_r/c_r, root moment per chord")
nu       = f.Variable(name="nu",       guess=0.5,    units="-",           description="(1 + lambda + lambda^2) / (1 + lambda)^2")
p        = f.Variable(name="p",        guess=1.9,    units="-",           description="1 + 2 * lambda")
P_max    = f.Variable(name="P_max",    guess=1.2e6,  units="W",           description="max engine output power")
q        = f.Variable(name="q",        guess=1.5,    units="-",           description="1 + lambda")
R        = f.Variable(name="R",        guess=5e6,    units="m",           description="range")
Re       = f.Variable(name="Re",       guess=500000, units="-",   size=3, description="Reynold's number")
S        = f.Variable(name="S",        guess=25.0,   units="m^2",         description="total wing area")
tau      = f.Variable(name="tau",      guess=0.12,   units="-",           description="wing thickness ratio")
t_cap    = f.Variable(name="t_cap",    guess=0.004,  units="-",           description="spar cap thickness per unit chord")
t_web    = f.Variable(name="t_web",    guess=0.0005, units="-",           description="shear web thickness per unit chord")
V        = f.Variable(name="V",        guess=40,     units="m/s", size=3, description="cruising speed")
V_stall  = f.Variable(name="V_stall",  guess=30.0,   units="m/s",         description="stall speed requirement")
W        = f.Variable(name="W",        guess=20000,  units="N",   size=3, description="total aircraft weight")
W_bar    = f.Variable(name="W_bar",    guess=5000,   units="N",           description="average aircraft operating weight")
W_cap    = f.Variable(name="W_cap",    guess=4000,   units="N",           description="spar cap weight")
W_eng    = f.Variable(name="W_eng",    guess=3000,   units="N",           description="engine weight")
W_f      = f.Variable(name="W_f",      guess=6000,   units="N",           description="total out-and-back fuel burn")
W_fout   = f.Variable(name="W_fout",   guess=2500.0, units="N",           description="outbound fuel burn")
W_fret   = f.Variable(name="W_fret",   guess=2500.0, units="N",           description="return trip fuel burn")
W_MTO    = f.Variable(name="W_MTO",    guess=2000.0, units="N",           description="total maximum takeoff weight")
W_pay    = f.Variable(name="W_pay",    guess=5000.0, units="N",           description="payload weight")
W_web    = f.Variable(name="W_web",    guess=100,    units="N",           description="shear web weight")
W_wing   = f.Variable(name="W_wing",   guess=7500,   units="N",           description="wing weight")
W_zfw    = f.Variable(name="W_zfw",    guess=3000,   units="N",           description="zero-fuel weight")
z_bre    = f.Variable(name="z_bre",    guess=0.1,    units="-",   size=2, description="Breguet parameter")


# =================
# Declare Constants
# =================
A_prop     = f.Constant(name="A_prop",     value=0.785,   units="m^2",    description="propeller disk area")
C_Lmax     = f.Constant(name="C_Lmax",     value=1.5,     units="-",      description="max CL with flaps down")
e          = f.Constant(name="e",          value=0.95,    units="-",      description="Oswald efficiency factor")
eta_eng    = f.Constant(name="eta_eng",    value=0.35,    units="-",      description="engine efficiency")
eta_v      = f.Constant(name="eta_v",      value=0.85,    units="-",      description="viscous propeller efficiency")
f_wadd     = f.Constant(name="f_wadd",     value=2,       units="-",      description="wing added weight fraction")
g          = f.Constant(name="g",          value=9.81,    units="m/s^2",  description="gravitational constant")
h_fuel     = f.Constant(name="h_fuel",     value=46e6,    units="J/kg",   description="fuel heating value")
mu         = f.Constant(name="mu",         value=1.69e-5, units="kg/m/s", description="viscosity of air")
N_lift     = f.Constant(name="N_lift",     value=6,       units="-",      description="wing loading multiplier")
omega_bar  = f.Constant(name="omega_bar",  value=0.5,     units="-",      description="wing-box width.chord")
r_h        = f.Constant(name="r_h",        value=0.75,    units="-",      description="wing box quadratically tapered fraction")
rho        = f.Constant(name="rho",        value=0.91,    units="kg/m^3", description="density of air at 3000m")
rho_cap    = f.Constant(name="rho_cap",    value=2700,    units="kg/m^3", description="cap density (aluminum)")
rho_web    = f.Constant(name="rho_web",    value=2700,    units="kg/m^3", description="web density (aluminum)")
rho_sl     = f.Constant(name="rho_sl",     value=1.23,    units="kg/m^3", description="density of air at sea level")
sigma_max  = f.Constant(name="sigma_max",  value=250e6,   units="Pa",     description="allowable stress, 6061-T6")
sigma_maxs = f.Constant(name="sigma_maxs", value=167e6,   units="Pa",     description="allowable shear stress")
W_fixed    = f.Constant(name="W_fixed",    value=14700,   units="N",      description="fixed aircraft weight ")
pi         = np.pi
#sigma_max and sigma_maxs constraints are slightly different in corsair

# =====================
# Declare the Objective
# =====================
f.Objective(W_f)

# =======================
# Declare the Constraints
# =======================
Constraints = []
Constraints += [
                W_f >= W_fout + W_fret,
                ]   

# Steady Level Flight Relations
for i in [0,1,2]:
    Constraints += [ 
                    W[i]  == 0.5 * rho * S * C_L[i] * V[i]**2,
                    D[i]  >= 0.5 * rho * S * C_D[i] * V[i]**2,
                    Re[i] == (rho / mu) * V[i] * (S / A) ** 0.5,
                    ]
    
# Landing flight condition
Constraints += [
                W_MTO   <= 0.5 * rho_sl * S * C_Lmax * V_stall**2,
                V_stall == 38,
                ]

# Sprint flight condition
Constraints += [ 
                P_max >= D[2] * V[2] / eta_0[2],
                V[2]  >= 150,  
                ]

# Drag Model
for i in [0,1,2]:
    Constraints += [C_Dfus[i] == 0.05 / S, 
                    C_Di[i]   == (C_L[i]**2) / (pi * A * e),
                    C_D[i]    >= C_Dfus[i] + C_Dp[i] + C_Di[i],
                    1         >= (2.56    * C_L[i]**( 5.88) * tau**(-3.32) * Re[i]**(-1.54) * C_Dp[i]**(-2.26) + 
                                  3.80e-9 * C_L[i]**(-0.92) * tau**( 6.23) * Re[i]**(-1.38) * C_Dp[i]**(-9.57) + 
                                  2.20e-3 * C_L[i]**(-0.01) * tau**( 0.03) * Re[i]**( 0.14) * C_Dp[i]**(-0.73) + 
                                  1.19e4  * C_L[i]**( 9.78) * tau**( 1.76) * Re[i]**(-1.00) * C_Dp[i]**(-0.91) + 
                                  6.14e-6 * C_L[i]**( 6.53) * tau**(-0.52) * Re[i]**(-0.99) * C_Dp[i]**(-5.19) ),
                    ]
    
# Propulsive Efficiency   
for i in [0,1,2]:
    Constraints += [
                    eta_0[i]    == eta_eng * eta_prop[i],
                    eta_prop[i] == eta_i[i] * eta_v,
                    4           >= 4*eta_i[i] + (D[i]*eta_i[i]**2) / (0.5 * rho * A_prop * V[i]**2),
                    ]
    
# Range Constraints
Constraints += [R >= 5000 * 10**3,
                ]
for i in [0,1]:
        Constraints += [
                    z_bre[i]      == g * R * D[i] / (h_fuel * eta_0[i] * W[i]),
                    W_fout / W[0] >= z_bre[0] + (z_bre[0]**2 / 2) + (z_bre[0]**3 / 6) + (z_bre[0]**4 / 24),
                    W_fret / W[1] >= z_bre[1] + (z_bre[1]**2 / 2) + (z_bre[1]**3 / 6) + (z_bre[1]**4 / 24),
                   ]
        
# Weight Relations        
Constraints += [
                W_pay           >= 500 * g,
                W_bar           >= W_fixed + W_pay + W_eng,
                W_zfw           >= W_bar + W_wing,
                W_eng           >= 0.0372 * P_max**0.803,
                W_wing / f_wadd >= W_web + W_cap,
                ] 
for i in [0,1,2]:
    Constraints += [
                W[0]        >= W_zfw + W_fret,   
                W_MTO       >= W[0] + W_fout,
                W[1]        >= W_zfw,
                W[2]        == W[0],            
                ]

# Wing Structural Models   
Constraints += [ 
                2*q        >= 1 + p,
                p          >= 1.9,
                tau        <= 0.15,
                M_barr     == W_bar * A * p / 24,
                0.92 * omega_bar * tau * t_cap**2 + I_cap <= (0.92**2 / 2) * omega_bar * tau**2 * t_cap,
                8          == N_lift * M_barr * A * q**2 * tau / (S * I_cap * sigma_max),
                12         == A * W_bar * N_lift * q**2 / (tau * S * t_web * sigma_maxs),
                # in the corsair code it is written as q**3, though the difference is marginal (~28 N of W_f)
                nu**3.94   >= 0.86 * p**-2.38 + 0.14 * p**0.56,
                W_cap      >= 8 * rho_cap * g * omega_bar * t_cap * S**1.5 * nu / (3 * A**0.5),
                W_web      >= 8 * rho_web * g * r_h * tau * t_web * S**1.5 * nu / (3 * A**0.5),
                ]

f.ConstraintList(Constraints)

var_list = f.get_variables()
con_list = f.get_constants()
    
#f.pprint()
from solver import cvxopt_solve
res = cvxopt_solve(f)
#print(res)

# Uncomment only if solving with ipopt
# import pyomo.environ as pyo
# opt = pyo.SolverFactory('ipopt')
# opt.solve(f)


# =======================
# Printing Results
# =======================

# Messy printing directly into terminal
# var_list = f.get_variables()
# ctr = 0
# for var in var_list:
#     #print(type(var))
#     if isinstance(var,pyomo.core.base.var.IndexedVar):
#         for ix in var.index_set():
#             lbls = ["out","ret","sprint"]
#             print('%s[%s]:  %.4f  %s'%(var.name,lbls[ix],res['x'][ctr],var._units))
#             ctr += 1
#     else:
#         print('%s:  %.4f  %s'%(var.name,res['x'][ctr],var._units))
#         ctr += 1


# Neater printing into table in terminal
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
            lbls = ["out", "ret", "sprint"]
            variable_names.append(f"{var.name}[{lbls[ix]}]")
            variable_values.append(res['x'][ctr])
            variable_units.append(var._units)
            variable_descriptions.append(var.doc)  # Description if available
            ctr += 1
    else:
        # Handle non-indexed variables
        variable_names.append(var.name)
        variable_values.append(res['x'][ctr])
        variable_units.append(var._units)
        variable_descriptions.append(var.doc)  # Description if available
        ctr += 1

# Create dataframe with the details
pd.set_option('display.max_rows', None) #display all variable rows
pd.options.display.float_format = '{:.4f}'.format #set output to 4 decimal places
vdf = pd.DataFrame({
    "Variable Name": variable_names,
    "Value": variable_values,
    "Units": variable_units,
    "Description": variable_descriptions
})

print(vdf)



#Initialize lists for constants 
constant_names = []
constant_values = []
constant_units = []
constant_descriptions = []

ctr = 0  # Counter for results array index

for con in con_list:
    constant_names.append(con.name)
    constant_values.append(res['x'][ctr])
    constant_units.append(con._units)
    constant_descriptions.append(con.doc)  # Description if available
    ctr += 1

# Create dataframe with the details
pd.set_option('display.max_rows', None) #display all variable rows
pd.options.display.float_format = '{:.4f}'.format #set output to 4 decimal places
cdf = pd.DataFrame({
    "Constant Name": constant_names,
    "Value": constant_values,
    "Units": constant_units,
    "Description": constant_descriptions
})

print(cdf)

# # Save to a csv or excel file (?)
# df.to_csv("variables_results_table.csv", index=False)
# df.to_excel("variables_results_table.xlsx", index=False)
