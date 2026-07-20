# ====================
# Import Statements
# ====================
import numpy as np
import pyomo 
from edi.objects.formulation import Formulation
from pyomo.environ import units

# ====================
# Declare Formulation
# ====================
f = Formulation()

# ====================
# Declare Free Variables
# ====================
a           = f.Variable(name='a',          guess=297,    units='m/s',    size=3, description='Speed of sound')
A0h         = f.Variable(name='A0h',        guess=50,     units='m^2',            description='Horizontal bending area constant')
A1hland     = f.Variable(name='A1hland',    guess=50,     units='m',              description='Horizontal bending area constant (landing case)')
A1hMLF      = f.Variable(name='A1hMLF',     guess=50,     units='m',              description='Horizontal bending area constant (maximum aero load case)')
A2hland     = f.Variable(name='A2hland',    guess=50,     units='-',              description='Horizontal bending area constant 2 (landing case)')
A2hMLF      = f.Variable(name='A2hMLF',     guess=50,     units='-',              description='Horizontal bending area constant 2 (maximum aero load case)')
A_floor     = f.Variable(name='A_floor',    guess=0.05549,     units='m^2',            description='Floor beam cross-sectional area')
A_fuse      = f.Variable(name='A_fuse',     guess=10.81,     units='m^2',            description='Fuselage cross-sectional area')
Ahbndbld    = f.Variable(name='Ahbndbld',   guess=50,     units='m^2',            description='Horizontal bending area at rear wing box (landing case)')
AhbndbMLF   = f.Variable(name='AhbndbMLF',  guess=50,     units='m^2',            description='Horizontal bending area at rear wing box (maximum aero load case)')
Ahbndfld    = f.Variable(name='Ahbndfld',   guess=50,     units='m^2',            description='Horizontal bending area of front wing box (landing case)')
AhbndfMLF   = f.Variable(name='AhbndfMLF',  guess=50,     units='m^2',            description='Horizontal bending area of front wing box (maximum aero load case)')
alpha_ht    = f.Variable(name='alpha_ht',   guess=0.06026,     units='-',      size=3, description='Horizontal tail angle of attack')
alpha_w     = f.Variable(name='alpha_w',    guess=0.1,     units='-',      size=3, description='Wing angle of attack')
AR_ht       = f.Variable(name='AR_ht',      guess=7.707,    units='-',              description='Horizontal tail aspect ratio')
AR_w        = f.Variable(name='AR_w',       guess=9.298,    units='-',              description='Wing Aspect Ratio')
AR_vt       = f.Variable(name='AR_vt',      guess=0.7093,   units='-',              description='Vertical tail aspect ratio')
A_skin      = f.Variable(name='A_skin',     guess=0.01087,     units='m^2',            description='Skin cross-sectional area')
Avbndb      = f.Variable(name='Avbndb',     guess=50,     units='m^2',    size=3, description='Vertical bending material area at rear wing box')
B           = f.Variable(name='B',          guess=15.75,   units='m',              description='Landing Gear base')
B0v         = f.Variable(name='B0v',        guess=50,     units='m^2',    size=3, description='Vertical bending area constant')
B1v         = f.Variable(name='B1v',        guess=50,     units='m',      size=3, description='Vertical bending area constant 2')
b_ht       = f.Variable(name='b_ht',       guess=10.68,   units='m',              description='Horizontal tail half-span')
b_w         = f.Variable(name='b_w',        guess=30.19,   units='m',              description='wingspan')
b_vt        = f.Variable(name='b_vt',       guess=3.707,   units='m',              description='Vertical tail half-span')
c0          = f.Variable(name='c0',         guess=10,     units='m',              description='Root chord of the wing')
cbar_ht     = f.Variable(name='cbar_ht',    guess=1.591,     units='m',              description='Horizontal tail mean aerodynamic chord')
cbar_w      = f.Variable(name='cbar_w',     guess=3.728,     units='m',              description='mean aerodynamic chord(wing)')
cbar_vt     = f.Variable(name='cbar_vt',    guess=6.519,     units='m',              description='Vertical tail mean aerochord')
C_D         = f.Variable(name='C_D',        guess=0.02911,   units='-',      size=3, description='Drag Coefficient')
C_D0ht      = f.Variable(name='C_D0ht',     guess=0.005348,   units='-',      size=3, description='Horizontal tail parasitic drag coefficient')
C_Dfuse     = f.Variable(name='C_Dfuse',    guess=0.005,  units='-',      size=3, description='Fuselage drag coefficient')
C_Dht       = f.Variable(name='C_Dht',       guess=0.005,  units='-',      size=3, description='Horizontal tail drag coefficient') #Not used in the model
C_Dw        = f.Variable(name='C_Dw',       guess=0.01677,  units='-',      size=3, description='Drag coefficient,wing')
C_Diht      = f.Variable(name='C_Diht',     guess=0.005,  units='-',      size=3, description='Horizontal tail induced drag coefficient')
C_Diw       = f.Variable(name='C_Diw',      guess=0.005,  units='-',      size=3, description='Wing Induced Drag Coefficient')
C_Dpw       = f.Variable(name='C_Dpw',      guess=0.005637,  units='-',      size=3, description='Wing Parasitic Drag Coefficient')
C_Dpvt      = f.Variable(name='C_Dpvt',     guess=0.004781,   units='-',      size=3, description='Viscous drag coefficient')
C_Lalphaht0 = f.Variable(name='C_Lalphaht0',guess=5.348,      units='-',      size=3, description='Horizontal tail isolated lift curve slope')
C_Lalphaht  = f.Variable(name='C_Lalphaht', guess=2.963,      units='-',      size=3, description='Horizontal tail lift curve slope')
C_Lht       = f.Variable(name='C_Lht',      guess=0.1785,      units='-',      size=3, description='Horizontal tail lift coefficient')
C_Lw        = f.Variable(name='C_Lw',       guess=0.5616,    units='-',      size=3, description='Lift Coefficient, wing')
C_Lwa       = f.Variable(name='C_Lwa',      guess=5.616,    units='-',      size=3, description='Lift-curve slope, wing')
C_LvtEO     = f.Variable(name='C_LvtEO',    guess=0.3905,      units='-',      size=3, description='Vertical tail lift coefficient during engine out')
C_Lvtland   = f.Variable(name='C_Lvtland',  guess=3,      units='-',      size=3, description='Vertical tail lift coefficient during landing')
c_rootht    = f.Variable(name='c_rootht',   guess=2.31,     units='m',              description='Horizontal tail root chord')
c_rootw     = f.Variable(name='c_rootw',    guess=5.411,     units='m',              description='Wing root chord')
c_rootvt    = f.Variable(name='c_rootvt',   guess=9.146,     units='m',              description='Vertical tail root chord')
c_tipht     = f.Variable(name='c_tipht',   guess=0.462,     units='m',              description='Horizontal tail tip chord')
c_tipw      = f.Variable(name='c_tipw',     guess=1.082,     units='m',              description='Wing tip chord')
c_tipvt     = f.Variable(name='c_tipvt',    guess=2.744,     units='m',              description='Vertical tail tip chord')
D           = f.Variable(name='D',          guess=34241,  units='N',      size=3, description='Total aircraft drag (cruise)')
D_fuse      = f.Variable(name='D_fuse',     guess=8844,   units='N',      size=3, description='Fuselage Drag')
D_ht        = f.Variable(name='D_ht',       guess=1505,   units='N',      size=3, description='Horizontal tail drag')
D_vt        = f.Variable(name='D_vt',       guess=6548,   units='N',      size=3, description='Vertical tail drag')
D_w         = f.Variable(name='D_w',        guess=17344,  units='N',      size=3, description='Wing Drag')
D_wm        = f.Variable(name='D_wm',       guess=3609,     units='N',      size=3, description='Engine out windmill drag')
d_nacelle   = f.Variable(name='d_nacelle',  guess=2.05,     units='m',              description='Nacelle diameter')
d_oleo      = f.Variable(name='d_oleo',     guess=0.3119,     units='m',              description='Diameter of oleo shock absorber')
d_tm        = f.Variable(name='d_tm',       guess=39.72,   units='m',           description='Diameter of main gear tires')
d_tn        = f.Variable(name='d_tn',       guess=31.78,     units='m',           description='Diameter of nose gear tires')
delt_Lo     = f.Variable(name='delt_Lo',    guess=50,     units='N',      size=3, description='Center wing lift loss')
delt_Lt     = f.Variable(name='delt_Lt',    guess=50,     units='N',      size=3, description='Wing-tip lift loss')
delt_xacw   = f.Variable(name='delt_xacw',  guess=50,     units='N',      size=3, description='Wing aerodynamic center shift')
delt_xlht  = f.Variable(name='delt_xlht',  guess=32.57,     units='m',      size=3, description='Distance from c.g. to HT leading edge')
delt_xtht   = f.Variable(name='delt_xtht',  guess=34.88,     units='m',      size=3, description='Distance from c.g. to HT trailing edge')
delt_xlvt   = f.Variable(name='delt_xlvt',  guess=25.73,     units='m',      size=3, description='Distance from c.g. to VT leading edge')
delt_xtvt   = f.Variable(name='delt_xtvt',  guess=34.88,     units='m',      size=3, description='Distance from c.g. to VT trailing edge')
delt_xm     = f.Variable(name='delt_xm',    guess=3.149,     units='m',      size=3, description='Distance between main gear and c.g.')
delt_xn     = f.Variable(name='delt_xn',    guess=12.6,     units='m',      size=3, description='Distance between nose gear and c.g.')
E_land      = f.Variable(name='E_land',     guess=2.657e5,     units='J',      size=3, description='Maximum kinetic energy to be absorbed in landing')
e           = f.Variable(name='e',          guess=0.9702,     units='-',      size=3, description='Oswald efficiency factor')
eta_o       = f.Variable(name='eta_o',      guess=1,     units='-',      size=3, description='center wingspan coefficient')
f_fuel      = f.Variable(name='f_fuel',     guess=1,      units='-',      size=3, description='Percent fuel remaining')
f_lambdaht = f.Variable(name='f_lambdaht', guess=0.0524,     units='-',      size=3, description='Horizontal tail taper ratio function')
f_lambdaw   = f.Variable(name='f_lambdaw',  guess=0.0033,      units='-',      size=3, description='Empirical efficiency function of taper')
F_wm        = f.Variable(name='F_wm',       guess=4458,   units='-',      size=3, description='Weight factor (main)')
F_wn        = f.Variable(name='F_wn',       guess=400.9,    units='-',      size=3, description='Weight Factor (nose)')
h_fuse      = f.Variable(name='h_fuse',     guess=100,    units='m',              description='Fuelage height')
I_hshell    = f.Variable(name='I_hshell',   guess=1000,   units='m^4',            description='Shell horizontal bending inertia')
I_m         = f.Variable(name='I_m',        guess=5.007e-6,   units='m^4',    size=3, description='Area moment of inertia (main strut)')
I_n         = f.Variable(name='I_n',        guess=7.204e-7,   units='m^4',    size=3, description='Area moment of inertia (nose strut)')
I_vshell    = f.Variable(name='I_vshell',   guess=1000,   units='m^4',            description='Shell vertical bending inertia')
I_z         = f.Variable(name='I_z',        guess=1000,   units='kg*m^2', size=3, description='Total aircraft moment of inertia')
I_zfuse     = f.Variable(name='I_zfuse',    guess=1000,   units='kg*m^2', size=3, description='Fuselage moment of inertia')
I_ztail     = f.Variable(name='I_ztail',    guess=1000,   units='kg*m^2', size=3, description='Tail moment of inertia')
I_zwing     = f.Variable(name='I_zwing',    guess=1000,   units='kg*m^2', size=3, description='Wing moment of inertia')
l_cone      = f.Variable(name='l_cone',     guess=22.86,     units='m',              description='Cone length')
l_floor     = f.Variable(name='l_floor',    guess=28.12,     units='m',              description='Floor length')
l_fuse      = f.Variable(name='l_fuse',     guess=52.47,   units='m',              description='Fuselage length')
l_ht       = f.Variable(name='l_ht',       guess=34.73,     units='m',              description='Horizontal tail moment arm')
l_mgear     = f.Variable(name='l_mgear',    guess=2.375,     units='m',              description='Length of main gear')
l_ngear     = f.Variable(name='l_ngear',    guess=1.627,     units='m',              description='Length of nose gear')
l_oleo      = f.Variable(name='l_oleo',     guess=0.7399,     units='m',              description='Length of oleo shock absorber')
l_shell     = f.Variable(name='l_shell',    guess=24.41,     units='m',              description='Shell length')
l_vt        = f.Variable(name='l_vt',       guess=28.2,      units='m',      size=3, description='Vertical tail moment arm')
lambda_ht  = f.Variable(name='lambda_ht',  guess=0.2,     units='-',              description='Horizontal tail taper ratio')
lambda_w    = f.Variable(name='lambda_w',   guess=0.2,     units='-',              description='wing taper ratio')
lambda_vt   = f.Variable(name='lambda_vt',  guess=0.3,     units='-',              description='Span and Taper ratio')
L_D         = f.Variable(name='L_D',        guess=19.29,   units='-',      size=3, description='Lift/drag ratio')
L_ht        = f.Variable(name='L_ht',       guess=2.695e4,     units='N',      size=3, description='Horizontal tail downforce')
L_htmax     = f.Variable(name='L_htmax',    guess=4.702e5,     units='N',      size=3, description='Maximum horizontal tail downforce')
L_m         = f.Variable(name='L_m',        guess=4.489e5,   units='N',      size=3, description='Maximum static load through main gear')
L_n         = f.Variable(name='L_n',        guess=1.122e5,   units='N',      size=3, description='Minimum static load through nose gear')
L_ndyn      = f.Variable(name='L_ndyn',     guess=4.834e4,   units='N',      size=3, description='Dynamic braking load, nose gear')
L_total     = f.Variable(name='L_total',    guess=6.399e5,   units='N',      size=3, description='Total Lift generated by aircraft')
L_w         = f.Variable(name='L_w',        guess=5.612e5,   units='N',      size=3, description='Wing Lift')
L_wmax      = f.Variable(name='L_wmax',     guess=3.112e6,   units='N',      size=3, description='Maximum lift generated by wing')
L_wm        = f.Variable(name='L_wm',       guess=1.122e5,   units='N',      size=3, description='Static load per wheel (main)')
L_wn        = f.Variable(name='L_wn',       guess=5.612e4,   units='N',      size=3, description='Static load per wheel (nose)')
L_vtEO      = f.Variable(name='L_vtEO',     guess=2.271e4,   units='N',      size=3, description='Vertical tail lift in engine out')
L_vtmax     = f.Variable(name='L_vtmax',    guess=1.28e6,   units='N',      size=3, description='Maximum load for structural sizing')
M           = f.Variable(name='M',          guess=0.8,    units='-',      size=3, description='Cruise mach number')
M_floor     = f.Variable(name='M_floor',    guess=4.442e5,    units='N*m',    size=3, description='Maximum bending moment in floor beams')
m_ratio     = f.Variable(name='m_ratio',    guess=10,     units='-',      size=3, description='Ratio of HT and wing lift-curve slopes')
mu          = f.Variable(name='mu',         guess=1.4e-5,    units='N*s/m^2',size=3, description='Dynamic Viscosity')
n_rows      = f.Variable(name='n_rows',     guess=31,     units='-',              description='Number of rows')
n_seats     = f.Variable(name='n_seats',    guess=186,      units='-',              description='Number of seats')
P_floor     = f.Variable(name='P_floor',    guess=1.137e6,    units='N',      size=3, description='Distributed floor load')
p_ht       = f.Variable(name='p_ht',       guess=1.4,     units='N/m',           description='Horizontal tail theoretical wing loading')
p_o         = f.Variable(name='p_o',        guess=10,     units='N/m',    size=3, description='Center section theoretical wing loading')
p_w         = f.Variable(name='p_w',        guess=1.4,     units='-',              description='Dummy variable (1+2lambda_w)')
p_vt        = f.Variable(name='p_vt',       guess=1.6,     units='-',              description='Dummy variable(1+2lambdavt)')
q_htd      = f.Variable(name='q_htd',      guess=1.2,     units='-',              description='Dummy Variable (1+lambdah)')
Q_vt        = f.Variable(name='Q_vt',       guess=100,    units='N*m',            description='Maximum Torsion Moment imparted by vertical tail')
q_vtd       = f.Variable(name='q_vtd',      guess=1.3,     units='-',              description='Dummy Variable (1+lambdavt)')
q_w         = f.Variable(name='q_w',        guess=1.2,     units='-',              description='Dummy Variable (1+lambdaw)')
R           = f.Variable(name='R',          guess=50000,   units='m',   size=3, description='Segment range')
Re_ht       = f.Variable(name='Re_ht',      guess=1.001e7,    units='-',      size=3, description='Horizontal tail Reynolds number')
Re_w        = f.Variable(name='Re_w',       guess=2.344e7,    units='-',      size=3, description='Cruise Reynolds number (wing)')
Re_vt       = f.Variable(name='Re_vt',      guess=4.099e7,    units='-',      size=3, description='Vertical tail Reynolds number')
R_fuse      = f.Variable(name='R_fuse',     guess=1.855,   units='m',              description='Fuselage radius')
r_m         = f.Variable(name='r_m',        guess=0.04181,     units='m',              description='Radius of main gear struts')
r_n         = f.Variable(name='r_n',        guess=0.04592,     units='m',              description='Radius of nose gear struts')
rho_cabin   = f.Variable(name='rho_cabin',  guess=1.225,  units='kg/m^3', size=3, description='Cabin air density')
rho_free    = f.Variable(name='rho_free',   guess=0.38,    units='kg/m^3', size=3, description='Freestream density')
S_bulk      = f.Variable(name='S_bulk',     guess=21.62,     units='m^2',            description='Bulkhead surface area')
S_floor     = f.Variable(name='S_floor',    guess=5.686e5,    units='N',      size=3, description='Maximum shear in floor beams')
S_ht       = f.Variable(name='S_ht',       guess=14.81,     units='m^2',            description='Horizontal tail cross-sectional area')
S_nose      = f.Variable(name='S_nose',     guess=49.82,    units='m^2',            description='Nose surface area')
S_sa        = f.Variable(name='S_sa',       guess=0.2959,     units='m',      size=3, description='Stroke of the shock absorber')
S_w         = f.Variable(name='S_w',        guess=98,  units='m^2',            description='Wing area')
S_vt        = f.Variable(name='S_vt',       guess=19.38,   units='m^2',            description='Vertical tail reference area(half)')
sigma_Mh    = f.Variable(name='sigma_Mh',   guess=10,     units='N/m^2',          description='Horizontal bending material stress')
sigma_Mv    = f.Variable(name='sigma_Mv',   guess=10,     units='N/m^2',  size=3, description='Vertical bending material stress')
sigma_thta  = f.Variable(name='sigma_thta', guess=1.034e8,     units='N/m^2',  size=3, description='Skin hoop stress')
sigma_x     = f.Variable(name='sigma_x',    guess=3.831e7,     units='N/m^2',  size=3, description='Axial stress in skin')
SM          = f.Variable(name='SM',         guess=0.05,   units='-',      size=3, description='Stability Margin')
T_lg        = f.Variable(name='T_lg',       guess=5.662,     units='m',      size=3, description='Mainlanding gear track')
t           = f.Variable(name='t',          guess=5000,   units='s',    size=3, description='Flight time')
t_cone      = f.Variable(name='t_cone',     guess=0.1,     units='m',              description='Thickness of the tail cone')
t_m         = f.Variable(name='t_m',        guess=0.02181,     units='m',              description='Thickness of main gear strut wall')
t_n         = f.Variable(name='t_n',        guess=0.002368,     units='m',              description='Thickness of nose gear strut wall')
t_shell     = f.Variable(name='t_shell',    guess=0.001259,     units='m',              description='Shell thickness')
t_skin      = f.Variable(name='t_skin',     guess=0.0009324,     units='m',              description='Skin thickness')
tau_cone    = f.Variable(name='tau_cone',   guess=1.034e8,     units='N/m^2',          description='Shear stress in tail cone')
tau_ht      = f.Variable(name='tau_ht',     guess=0.15,     units='-',      size=3, description='Horizontal tail thickness/chord ratio')
tau_w       = f.Variable(name='tau_w',      guess=0.15,     units='-',      size=3, description='wing thickness/chord ratio')
tau_vt      = f.Variable(name='tau_vt',     guess=0.09958,     units='-',      size=3, description='Vertical tail thickness/chord ratio')
V_bulk      = f.Variable(name='V_bulk',     guess=0.02016,    units='m^3',            description='Bulkhead skin volume')
V_cabin     = f.Variable(name='V_cabin',    guess=315,    units='m^3',            description='Cabin Volume')
V_cone      = f.Variable(name='V_cone',     guess=0.1657,    units='m^3',            description='Cone skin Volume')
V_cyl       = f.Variable(name='V_cyl',      guess=0.2653,    units='m^3',            description='Cylinder skin volume')
V_floor     = f.Variable(name='V_floor',    guess=0.1734,    units='m^3',            description='Floor Volume')
V_free      = f.Variable(name='V_free',     guess=231.7,    units='m/s',    size=3, description='Freestream Velocity')
V_fuelmax   = f.Variable(name='V_fuelmax',  guess=10,   units='m^3',    size=3, description='Available Fuel Volume')
V_hbend     = f.Variable(name='V_hbend',    guess=100,    units='m^3',    size=3, description='Horizontal bending material volume')
V_hbndb     = f.Variable(name='V_hbndb',    guess=100,    units='m^3',            description='Horizontal bending material volume b')
V_hbndc     = f.Variable(name='V_hbndc',    guess=100,    units='m^3',            description='Horizontal bending material volume c')
V_hbndf     = f.Variable(name='V_hbndf',    guess=100,    units='m^3',            description='Horizontal bending material volume f')
V_ht        = f.Variable(name='V_ht',       guess=1,    units='m^3',            description='Horizontal tail volume')
V_nose      = f.Variable(name='V_nose',     guess=0.04645,    units='m^3',            description='Nose skin volume')
V_TO        = f.Variable(name='V_TO',       guess=73.38,    units='m/s',    size=3, description='Takeoff Velocity')
V_vbend     = f.Variable(name='V_vbend',    guess=100,    units='m^3',    size=3, description='Vertical bending material volume')
V_vbndb     = f.Variable(name='V_vbndb',    guess=100,    units='m^3',    size=3, description='Vertical bending material volume b')
V_vbndc     = f.Variable(name='V_vbndc',    guess=100,    units='m^3',    size=3, description='Vertical bending material volume c')
wdt_floor   = f.Variable(name='wdt_floor',  guess=3.125,    units='m',              description='Floor half width')
wdt_fuse    = f.Variable(name='wdt_fuse',   guess=3.71,    units='m',              description='Fuselage width')
wdt_tm      = f.Variable(name='wdt_tm',     guess=0.3439,     units='m',              description='Width of main tires')
wdt_tn      = f.Variable(name='wdt_tn',     guess=0.2751,     units='m',              description='Width of nose tires')
W_avg       = f.Variable(name='W_avg',      guess=50000,    units='N',    size=3, description='Average aircraft weight during flight segment') # Changed from lbf
W_buoy      = f.Variable(name='W_buoy',     guess=1517,   units='N',    size=3, description='Buoyancy weight')
W_cone      = f.Variable(name='W_cone',     guess=7898,   units='N',            description='Cone weight')
W_dry       = f.Variable(name='W_dry',      guess=92822,  units='N',            description='Aircraft dry weight') # Changed from lbf
W_end       = f.Variable(name='W_end',      guess=1000,   units='N',    size=3, description='Aircraft weight at end-of-flight segment')
W_floor     = f.Variable(name='W_floor',    guess=9865,   units='N',            description='Floor weight')
W_fuel      = f.Variable(name='W_fuel',     guess=50000,   units='N',    size=3, description='Weight of fuel burned per flight segment') # Changed from lbf
W_fuelprim  = f.Variable(name='W_fuelprim', guess=5.164e4,   units='N',    size=3, description='Primary fuel weight (excludes reserves)')
W_fueltotal = f.Variable(name='W_fueltotal',guess=5.164e4,   units='N',    size=3, description='Total fuel weight')
W_fuelwing  = f.Variable(name='W_fuelwing', guess=5000,   units='N',    size=3, description='Maximum fuel weight carried in wing') # Changed from lbf
W_fuse      = f.Variable(name='W_fuse',     guess=1.616e5,  units='N',            description='Fuselage weight')
W_hbend     = f.Variable(name='W_hbend',    guess=5000,   units='N',    size=3, description='Horizontal bending material weight') # Changed from lbf
W_hpesys    = f.Variable(name='W_hpesys',   guess=5000,   units='N',            description='Power system weight') # Changed from lbf
W_ht        = f.Variable(name='W_ht',       guess=5196,    units='N',            description='Horizontal tail weight')
W_insul     = f.Variable(name='W_insul',    guess=4307,   units='N',            description='Insulation material weight')
W_lg        = f.Variable(name='W_lg',       guess=1.212e4,   units='N',            description='Landing gear weight')
W_max       = f.Variable(name='W_max',      guess=500000,   units='N',    size=3, description='Maximum aircraft weight') # Changed from lbf
W_mg        = f.Variable(name='W_mg',       guess=1.1e4,   units='N',            description='Main landing gear weight')
W_misc      = f.Variable(name='W_misc',     guess=5000,   units='N',            description='Miscellaneous system weight') # Changed from lbf
W_ms        = f.Variable(name='W_ms',       guess=1048,   units='N',            description='Weight of main struts')
W_mw        = f.Variable(name='W_mw',       guess=1781,   units='N',            description='Weight of main wheels (per strut)')
W_ng        = f.Variable(name='W_ng',       guess=1113,   units='N',            description='Nose landing gear weight')
W_ns        = f.Variable(name='W_ns',       guess=85.63,   units='N',            description='Weight of nose strut')
W_nw        = f.Variable(name='W_nw',       guess=410.8,   units='N',            description='Weight of nose wheels (total)')
W_padd      = f.Variable(name='W_padd',     guess=6.465e4,   units='N',            description='Miscellaneous weights (galley,toilets,doors,etc)')
W_pay       = f.Variable(name='W_pay',      guess=1.616e5,   units='N',            description='Payload weight')
W_s         = f.Variable(name='W_s',        guess=15,     units='N/m^2',  size=3, description='Wing Loading')
W_seats     = f.Variable(name='W_seats',    guess=2.79e4,   units='N',            description='Seating weight')
W_shell     = f.Variable(name='W_shell',    guess=1.582e4,   units='N',            description='Shell weight')
W_skin      = f.Variable(name='W_skin',     guess=8791,   units='N',            description='Skin weight')
W_start     = f.Variable(name='W_start',    guess=5.162e5,   units='N',    size=3, description='Aircraft weight at start of flight segment') # Changed from lbf
W_structht  = f.Variable(name='W_structht', guess=1.2e4,   units='N',    size=3, description='Horizontal tail box weight')
W_structvt  = f.Variable(name='W_structvt', guess=1.2e4,   units='N',    size=3, description='Vertical tail box weight')
W_structw   = f.Variable(name='W_structw',  guess=3927,   units='N',    size=3, description='Wing box weight')
W_vbend     = f.Variable(name='W_vbend',    guess=5000,   units='N',    size=3, description='Vertical bending material weight') # Changed from lbf
W_vt        = f.Variable(name='W_vt',       guess=1963,   units='N',            description='Vertical tail weight')
W_wam       = f.Variable(name='W_wam',      guess=200.2,   units='N',            description='Wheel assembly weight for single main gear wheel') # Changed from lbf
W_wan       = f.Variable(name='W_wan',      guess=46.18,   units='N',            description='Wheel assembly weight for single nose gear wheel') # Changed from lbf
W_window    = f.Variable(name='W_window',   guess=1.062e4,   units='N',            description='Window weight')
W_wing      = f.Variable(name='W_wing',     guess=1.204e5,  units='N',            description='Wing Weight')
x_b         = f.Variable(name='x_b',        guess=1000,   units='m',      size=3, description='Wing box forward bulkhead location')
x_bwb       = f.Variable(name='x_bwb',      guess=100,    units='m',              description='x location of back of wing box')
x_CG        = f.Variable(name='x_CG',       guess=17.6,   units='m',      size=3, description='x location of c.g.')
x_CGht      = f.Variable(name='x_CGht',     guess=51.32,   units='m',      size=3, description='x location of horizontal tail c.g.')
x_CGlg      = f.Variable(name='x_CGlg',     guess=19.3,   units='m',      size=3, description='x location of landing gear c.g.')
x_CGmisc    = f.Variable(name='x_CGmisc',   guess=10,   units='m',      size=3, description='x location of miscellaneous systems c.g.')
x_CGvt      = f.Variable(name='x_CGvt',     guess=47.9,   units='m',      size=3, description='x location of vertical tail c.g.')
x_fwb       = f.Variable(name='x_fwb',      guess=10,     units='m',              description='x location of front of wing box')
x_hbndld    = f.Variable(name='x_hbndld',   guess=100,    units='m',             description='Horizontal zero bending location (landing case)') # Changed from m
x_hbndMLF   = f.Variable(name='x_hbndMLF',  guess=100,    units='m',             description='Horizontal zero bending location (maximum aero load case)') # Changed from m
x_hpesys    = f.Variable(name='x_hpesys',   guess=1000,   units='m',      size=3, description='Power systems centroid')
x_m         = f.Variable(name='x_m',        guess=20.75,   units='m',      size=3, description='x location of main gear')
x_mg        = f.Variable(name='x_mg',       guess=20.75,   units='m',      size=3, description='Main landing gear centroid')
x_n         = f.Variable(name='x_n',        guess=5,   units='m',      size=3, description='x location of nose gear')
x_ng        = f.Variable(name='x_ng',       guess=5,   units='m',      size=3, description='Nose landing gear centroid')
x_shell1    = f.Variable(name='x_shell1',   guess=5.2,    units='m',              description='Start of cylinder section')
x_shell2    = f.Variable(name='x_shell2',   guess=29.61,    units='m',              description='End of cylinder section')
x_tail      = f.Variable(name='x_tail',     guess=100,    units='m',      size=3, description='x location of tail')
x_TO        = f.Variable(name='x_TO',       guess=1524,   units='m',      size=3, description='Takeoff distance')
x_up        = f.Variable(name='x_up',       guess=29.61,    units='m',      size=3, description='x location of fuselage upsweep point')
x_vbend     = f.Variable(name='x_vbend',    guess=10,    units='m',     size=3, description='Vertical zero bending location')
x_w         = f.Variable(name='x_w',        guess=19.6,    units='m',      size=3, description='Position of wing aerodynamic center')
x_wing      = f.Variable(name='x_wing',     guess=19.6,   units='m',      size=3, description='Wing centroid')
# x_wing4     = f.Variable(name='x_wing4',    guess=100,    units='m',      size=3, description='x location of wing c/4')
y           = f.Variable(name='y',          guess=0.1318,      units='-',      size=3, description='Takeoff parameter')
y_cbarht = f.Variable(name='y_cbarht',   guess=3.052,     units='m',               description='Spanwise location of horizontal tail mean aerodynamic chord')
y_cbarw     = f.Variable(name='y_cbarw',    guess=8.624,     units='m',              description='Spanwise location of mean aerodynamic chord')
y_cbarvt    = f.Variable(name='y_cbarvt',   guess=4.5,     units='m',               description='Spanwise location of vertical tail mean aerodynamic chord')
y_m         = f.Variable(name='y_m',        guess=2.831,    units='m',      size=3, description='y location of main gear (symmetric)')
z_bre       = f.Variable(name='z_bre',      guess=0.1036,      units='-',      size=3, description='Breguet parameter')
z_cbarvt    = f.Variable(name='z_cbarvt',   guess=1.004,     units='m',      size=3, description='Vertical location of mean aerodynamic chord')
zeta        = f.Variable(name='zeta',       guess=0.07295,      units='-',      size=3, description='Takeoff parameter')

# =======================
# Wing Box Free Variables
# =======================
I_cap       = f.Variable(name='I_cap',      guess=3.458e-5,    units='-',      size=3, description='Nondimensional spar cap area moment of interia')
M_r         = f.Variable(name='M_r',        guess=1.688e6,    units='N',      size=3, description='Root moment per root chord')
t_cap       = f.Variable(name='t_cap',      guess=0.008251,    units='-',      size=3, description='Nondimensional spar cap thickness')
t_web       = f.Variable(name='t_web',      guess=0.002828,    units='-',      size=3, description='Nondimensional shear web thickness')
W_cap       = f.Variable(name='W_cap',      guess=7.984e4,    units='N',            description='Weight of Spar caps')
# W_fuelw     = f.Variable(name='W_fuelw',    guess=100,    units='lbf',    size=3, description='Maximum fuel weight carried in wing')
W_web       = f.Variable(name='W_web',      guess=6158,    units='N',            description='Weight of shear web')
v           = f.Variable(name='v',          guess=0.8612,    units='-',              description='Dummy variable ((t^2+t+1)/(t+1)^2)')

# ========================
# Vertical Tail Free variables
# ========================
I_capvt    = f.Variable(name='I_capvt',   guess=3.458e-5,    units='-',      size=3, description='Nondimensional vertical tail spar cap area moment of interia')
M_rvt      = f.Variable(name='M_rvt',     guess=1.688e6,    units='N',      size=3, description='Vertical tail root moment per root chord')
W_webvt = f.Variable(name='W_webvt', guess=1000, units='N', description='Weight of vertical tail shear web')
W_capvt = f.Variable(name='W_capvt', guess=1000, units='N', description='Weight of vertical tail spar caps')
vvt     = f.Variable(name='v_vt',      guess=0.8612,    units='-',       description='Dummy variable ((t^2+t+1)/(t+1)^2)')

# =========================
# Horizontal Tail Free variables
# =========================
W_capht    = f.Variable(name='W_capht',   guess=1000,    units='N',            description='Weight of horizontal tail spar caps')
W_webht    = f.Variable(name='W_webht',   guess=1000,    units='N',            description='Weight of horizontal tail shear web')
I_capht    = f.Variable(name='I_capht',   guess=3.458e-5,    units='-',      size=3, description='Nondimensional horizontal tail spar cap area moment of interia')
M_rht      = f.Variable(name='M_rht',     guess=1.688e6,    units='N',      size=3, description='Horizontal tail root moment per root chord')
vht    = f.Variable(name='v_ht',      guess=0.8612,    units='-',       description='Dummy variable ((t^2+t+1)/(t+1)^2)')

# ===============
# Trig Variables?
# ===============
tan_phi     = f.Variable(name='tan_phi',    guess=1,      units='-',      size=3, description='Angle between main gear and c.g.')
tan_psi     = f.Variable(name='tan_psi',    guess=1,      units='-',      size=3, description='Tip over angle')

# =======================
# Declare Fixed Variables
# =======================
A_fan       = f.Constant(name='A_fan',      value=2.41,   units='m^2',    description='Engine reference area')
alpha_wmax  = f.Constant(name='alpha_wmax', value=15,     units='-',      description='Maximum angle of attack')
b_wmax      = f.Constant(name='b_wmax',     value=35.9,   units='m',      description='maximum allowed wingspan')
c_Dwm       = f.Constant(name='c_Dwm',      value=0.5,    units='-',      description='Windmill drag coefficient')
c_Lhtmax    = f.Constant(name='c_Lhtmax',   value=2.5,    units='-',      description='Maximum horizontal tail lift coefficient')
c_lvt_EO    = f.Constant(name='c_Lvt_EO',   value=1,      units='-',      description='Sectional lift force coefficient (engine out)')
c_Lwmax     = f.Constant(name='c_Lwmax',    value=2.5,      units='-',      description='Maximum lift coefficient, wing')
c_Lvtmax    = f.Constant(name='c_Lvtmax',   value=2.6,      units='-',      description='Maximum lift coefficient')
c_mac       = f.Constant(name='c_mac',      value=0.1,   units='-',      description='Moment coefficient about aerodynamic center(wing)')
c_T         = f.Constant(name='c_T',        value=0.00008333,    units='1/s',    description='thrust specific fuel consumption') # Changed from 0.3 1/h
d_fan       = f.Constant(name='d_fan',      value=1.75,   units='m',      description='Fan diameter')
d2W_floor   = f.Constant(name='d2W_floor',  value=60,    units='N/m^2',  description='Floor weight per unit area')
d2W_insul   = f.Constant(name='d2W_insul',  value=22,     units='N/m^2',  description='Insulation material weight per unit area')
delt_Pover  = f.Constant(name='delt_Pover', value=2.9,    units='psi',    description='Cabin overpressure')
delt_xCG    = f.Constant(name='delt_xCG',   value=2,      units='m',      description='Center of gravity travel range')
dW_window   = f.Constant(name='dW_window',  value=435,    units='N/m',    description='Window weight per unit length')
E           = f.Constant(name='E',          value=205,    units='GPa',    description='Modulus of elasticity, 4340 steel')
e_vt        = f.Constant(name='e_vt',       value=0.8,   units='-',      description='Span efficiency of vertical tail')
eta_ht      = f.Constant(name='eta_ht',     value=0.97,    units='-',      description='Tail efficiency')
eta_s       = f.Constant(name='eta_s',      value=0.8,    units='-',      description='Shock absorber efficiency')
eta_w       = f.Constant(name='eta_w',      value=0.97,    units='-',      description='Lift efficiency')
f_aileron   = f.Constant(name='f_aileron',  value=0.005,  units='-',      description='Aileron Added weight fraction')
f_apu       = f.Constant(name='f_apu',      value=0.035,   units='-',      description='APU weight as fraction of payload weight')
f_addm      = f.Constant(name='f_addm',     value=1.5,    units='-',      description='Proportional added weight, main')
f_addn      = f.Constant(name='f_addn',     value=1.5,    units='-',      description='Proportional added weight, nose')
f_fadd      = f.Constant(name='f_fadd',     value=0.2,    units='-',      description='Fractional add weight of local reinforcements')
f_flap      = f.Constant(name='f_flap',     value=0.03,   units='-',      description='Flap Added weight fraction')
f_frame     = f.Constant(name='f_frame',    value=0.25,   units='-',      description='Fractional frame weight')
f_fuelres   = f.Constant(name='f_fuelres',  value=0.1,    units='-',      description='Fuel reserve fraction')
f_fueluse   = f.Constant(name='f_fueluse',  value=0.9,    units='-',      description='Usability factor of maximum fuel volume')
f_fuelw     = f.Constant(name='f_fuelw',    value=0.5,    units='-',      description='Fraction of total fuel stored in wing')
f_Lo        = f.Constant(name='f_Lo',       value=0.8,    units='-',      description='Center wing lift reduction coefficient')
f_Lt        = f.Constant(name='f_Lt',       value=0.8,    units='-',      description='Wing-tip lift reduction coefficient')
f_Ltotw     = f.Constant(name='f_Ltotw',    value=1.2,    units='-',      description='Total lift divided by wing lift')
f_lete      = f.Constant(name='f_lete',     value=0.05,   units='-',      description='Lete added weight fraction')
f_padd      = f.Constant(name='f_padd',     value=0.4,    units='-',      description='Other misc weight as fraction of payload weight')
f_ribs      = f.Constant(name='f_ribs',     value=0.02,   units='-',      description='Wing rib added weight fraction')
f_slat      = f.Constant(name='f_slat',     value=0.02,   units='-',      description='Slat added weight fraction')
f_spoiler   = f.Constant(name='f_spoiler',  value=0.01,   units='-',      description='Spoiler added weight fraction')
f_string    = f.Constant(name='f_string',   value=0.35,   units='-',      description='Fractional stringer weight')
f_tip       = f.Constant(name='f_tip',      value=0.05,   units='-',      description='Induced drag reduction from wing-tip devices')
f_watt      = f.Constant(name='f_watt',     value=0.0001, units='-',      description='Wing attatchment hardware added weight fraction')
g           = f.Constant(name='g',          value=9.81,   units='m/s^2',  description='Gravitational acceleration')
h_floor     = f.Constant(name='h_floor',    value=0.08359,    units='m',      description='Floor beam height')
h_hold      = f.Constant(name='h_hold',     value=2,      units='m',      description='Hold height')
h_nacelle   = f.Constant(name='h_nacelle',  value=0.5,   units='m',      description='Minimum nacelle clearance')
K           = f.Constant(name='K',          value=2,      units='-',      description='Column effective length factor')
l_nose      = f.Constant(name='l_nose',     value=5.2,    units='m',      description='Noselength')
l_r         = f.Constant(name='l_r',        value=2134,   units='m',      description='Maximum runway length') # Changed from ft
lambda_cone = f.Constant(name='lambda_cone',value=0.4,    units='-',      description='Tailcone radius taper ratio')
lambda_LG   = f.Constant(name='lambda_LG',  value=2.5,      units='-',      description='Ratio of max to static load')
lambda_htmin = f.Constant(name='lambda_htmin', value=0.2,   units='-',      description='Minimum horizontal tail taper ratio')
lambda_wmin = f.Constant(name='lambda_wmin',value=0.2,   units='-',      description='Minimum wing taper ratio')
lambda_vtmin = f.Constant(name='lambda_vtmin',value=0.2,   units='-',      description='Minimum vertical tail taper ratio')
M_fuseD     = f.Constant(name='M_fuseD',    value=0.8,    units='-',      description='Fuselage drag reference Mach number')
M_min       = f.Constant(name='M_min',      value=0.6,    units='-',      description='Minimum mach number')
n_eng       = f.Constant(name='n_eng',      value=2,      units='-',      description='Number of engines')
N_land      = f.Constant(name='N_land',     value=6,      units='-',      description='Emergency lading load factor')
N_liftmax   = f.Constant(name='N_lftmax',   value=2.5,    units='-',      description='Wing maximum load factor')
N_s         = f.Constant(name='N_s',        value=2.0,    units='-',      description='Factor of safety')
n_mg        = f.Constant(name='n_mg',       value=2,      units='-',      description='Number of main gear struts')
n_pass      = f.Constant(name='n_pass',     value=167,    units='-',      description='Number of passengers')
n_spr       = f.Constant(name='n_spr',      value=6,      units='-',      description='Number of seats per row')
n_wps       = f.Constant(name='n_wps',      value=1,      units='-',      description='Number of wheels per strut')
pi          = np.pi
p_cabin     = f.Constant(name='p_cabin',    value=7.5e4,  units='N/m^2',  description='Cabin Air Pressure')
p_oleo      = f.Constant(name='p_oleo',     value=1800,   units='psi',    description='Oleo Pressure')
p_s         = f.Constant(name='p_s',        value=0.787,     units='m',     description='Seat pitch')
R_sheat     = f.Constant(name='R_sheat',    value=287,   units='J/kg*K', description='Air specific heat')
R_e         = f.Constant(name='R_e',        value=1,      units='-',      description='Ratio of stringer/skin moduli')
R_Mh        = f.Constant(name='R_Mh',       value=1,      units='-',      description='Horizontal inertial relief factor')
R_Mv        = f.Constant(name='R_Mv',       value=1,      units='-',      description='Vertical inertial relief factor')
R_req       = f.Constant(name='R_req',      value=50000,   units='m',  description='Required total range')
rdot_req    = f.Constant(name='rdot_req',   value=2,      units='s^-2',   description='Maximum required yaw rate acceleration at landing')
rho_bend    = f.Constant(name='rho_bend',   value=2700,   units='kg/m^3', description='Stringer density')
rho_cone    = f.Constant(name='rho_cone',   value=2700,   units='kg/m^3', description='Cone material density')
rho_floor   = f.Constant(name='rho_floor',  value=2700,   units='kg/m^3', description='Floor material density')
rho_fuel    = f.Constant(name='rho_fuel',   value=817,    units='kg/m^3', description='Density of fuel')
rho_skin    = f.Constant(name='rho_skin',   value=2700,   units='kg/m^3', description='Skin density')
rho_ST      = f.Constant(name='rho_ST',     value=7850,   units='kg/m^3', description='Density of 4340 steel')
rho_TO      = f.Constant(name='rho_TO',     value=1.225,  units='kg/m^3', description='Takeoff density')
sigma_bend  = f.Constant(name='sigma_bend', value=1.85e6, units='N/m^2',  description='Bending material stress')
sigma_floor = f.Constant(name='sigma_floor',value=2.069e8,  units='N/m^2',  description='Maximum allowable floor stress')
sigma_skin  = f.Constant(name='sigma_skin', value=1.034e8,  units='N/m^2',  description='Maximum allowable skin stress')
sigma_yc    = f.Constant(name='sigma_yc',   value=4.7e8,  units='Pa',     description='Compressive yield strength 4340 steel')
SM_min      = f.Constant(name='SM_min',     value=0.05,   units='-',      description='Minimum allowed stability margin')
T           = f.Constant(name='T',          value=1.29e5,   units='N',      description='Thrust per engine in cruise')
T_cabin     = f.Constant(name='T_cabin',    value=300, units='K',      description='Cabin Air Temperature')
T_TO        = f.Constant(name='T_TO',       value=1.29e5,  units='N',      description='Thrust per engine at takeoff')
t_nacelle   = f.Constant(name='t_nacelle',  value=0.15,    units='m',      description='Nacelle thickness')
tau_floor   = f.Constant(name='tau_floor',  value=2.069e8,  units='N/m^2',  description='Maximum allowable shear web stress')
W_apu       = f.Constant(name='W_apu',      value=5657,    units='N',      description='APU weight')
w_aisle     = f.Constant(name='w_aisle',    value=0.51,    units='m',      description='Aisle width')
W_avgpass   = f.Constant(name='W_avgpass',  value=800,    units='N',    description='Average passenger weight, including luggage') # Converted from 180 lbf
W_eng       = f.Constant(name='W_eng',      value=1e4,  units='N',      description='Engine weight')
W_fix       = f.Constant(name='W_fix',      value=3000,    units='lbf',    description='Fixed weights (pilots,cockpit seats,navcom)')
W_seat      = f.Constant(name='W_seat',     value=178,    units='N',      description='Weight per seat')
w_seatwdth  = f.Constant(name='w_seatwdth', value=0.5,   units='m',      description='Seat width')
W_smax      = f.Constant(name='W_smax',     value=3434,   units='N/m^2',  description='Maximum Wing loading')
w_sys       = f.Constant(name='w_sys',      value=0.1,    units='m',      description='Width between cabin and skin for systems')
w_ult       = f.Constant(name='w_ult',      value=20,     units='m/s',   description='Ultimate velocity of descent') # Converted from 65 ft/s
V1          = f.Constant(name='V1',         value=70,     units='m/s',    description='Minimum takeoff velocity')
V_land      = f.Constant(name='V_land',     value=65,     units='m/s',    description='Landing velocity')
V_ne        = f.Constant(name='V_ne',       value=144,    units='m/s',    description='Never exceed velocity')
y_eng       = f.Constant(name='y_eng',      value=4.83,    units='m',    description='Engine moment arm')
z_CG        = f.Constant(name='z_CG',       value=2,      units='m',      description='Center of gravity height relative to bottom of fuselage')
z_wing      = f.Constant(name='z_wing',     value=0.5,    units='m',      description='Height of wing relative to base of fuselage')

# ========================
# Wing box Fixed Variables
# ========================
N_lift      = f.Constant(name='N_lift',     value=2,    units='-',      description='Wing loading multiplier')
rho_cap     = f.Constant(name='rho_cap',    value=2700,   units='kg/m^3', description='Density of spar cap material')
rho_web     = f.Constant(name='rho_web',    value=2700,   units='kg/m^3', description='Density of shear web material')
sigma_max   = f.Constant(name='sigma_max',  value=2.5e8,  units='Pa',     description='Allowable tensile stress')
sigma_maxsh = f.Constant(name='sigma_maxsh',value=1.67e8,  units='Pa',     description='Allowable shear stress')
r_h         = f.Constant(name='r_h',        value=0.75,   units='-',      description='Fractional Wing thickness at spar web')
r_wc        = f.Constant(name='r_wc',       value=0.25,   units='-',      description='Wing box width-to-chord ratio')

# ========================
# Vertical Tail Box Fixed Variables
# ========================
r_vtwc      = f.Constant(name='r_vtwc',      value=0.32,   units='-',      description='Vertical tail box width-to-chord ratio')
r_vth       = f.Constant(name='r_vth',       value=0.75,   units='-',      description='Fractional vertical tail thickness at spar web')

# ===========================
# Horizontal Tail Box Fixed Variables
# ===========================
r_hht = f.Constant(name='r_hht',       value=0.75,   units='-',      description='Fractional horizontal tail thickness at spar web')
r_htc = f.Constant(name='r_hthwc',    value=0.40,   units='-',      description='Horizontal tail box width-to-chord ratio')

# ===============
# Trig Constants?
# ===============
cos_chord   = f.Constant(name='cos_chord',  value=0.866,      units='-',      description='Cosine of quarter-chord sweep angle')
tan_chord   = f.Constant(name='tan_chord',  value=0.5774,      units='-',      description='Tangent of quarter-chord sweep angle')
tan_chdvt   = f.Constant(name='tan_chdvt',  value=0.8391,      units='-',      description='Tangent of leading edge sweep (40 deg)')
tan_gamma   = f.Constant(name='tan_gamma',  value=0.08749,          units='-',      description='Dihedral Angle')
tan_phimin  = f.Constant(name='tan_phimin', value=0.2679,        units='-',      description='Lower bound on phi')
tan_psimax  = f.Constant(name='tan_psimax', value=1.963,         units='-',      description='Upper bound on psi')
tan_thtmax  = f.Constant(name='tan_thtmax', value=0.2679,         units='-',      description='Maximum rotation angle')

# =======================
# System Level Model
#========================
# f.Objective(W_max[0] + W_max[1] + W_max[2])  # Minimize Max weight at start
f.Objective(W_fuel[0] + W_fuel[1] + W_fuel[2]) # Minimize fuel burned

Constraints = [] 
# Constraints += [
#                 R[0] + R[1] + R[2] >= R_req,
#                 R[0] == R[1],
#                 R[1] == R[2]
#                 ]



# Flight Performance
for i in [0,1,2]:
    Constraints += [
                    # R[i] == R_req / 3,
                    R[0] + R[1] + R[2] >= R_req,
                    R[0] == R[1],
                    R[1] == R[2],
                    W_fuel[i] >= W_end[i] * (z_bre[i] + (z_bre[i]**2)/2 + (z_bre[i]**3)/6),
                    z_bre[i]  >= c_T * t[i] * (D[i] / W_avg[i]),
                    t[i]      == R[i] / V_free[i],
                    L_D[i]    == W_avg[i] / D[i],
                    D[i]      == n_eng * T,
                    W_dry       >= W_wing + W_fuse + W_vt + W_ht + W_lg + W_misc + n_eng * W_eng,
                    W_start[0]     == W_max[i],
                    W_start[i]     >= W_end[i] + W_fuel[i],
                    W_start[1]     == W_end[0],
                    W_start[2]     == W_end[1],
                    W_end[i]       >= W_dry + W_pay + f_fuelres * W_fuelprim[i],
                    W_max[i]       >= W_dry + W_pay + W_fuelprim[i] * (1 + f_fuelres),
                    W_fuelprim[i]  >= W_fuel[0] + W_fuel[1] + W_fuel[2],
                    W_avg[i]       >= (W_start[i] * W_end[i])**0.5 + W_buoy[i],
                    D[i]           >= D_w[i] + D_fuse[i] + D_vt[i] + D_ht[i],
                    M[i]   == V_free[i] / a[i],
                    M[i]   >= M_min,
                    x_TO[i]    <= l_r,
                    1 + y[i]   <= 2 * ((g * x_TO[i] * T_TO) / (W_max[i] * V_TO[i]**2)),
                    1       >= 0.0464 * (zeta[i]**2.7 / y[i]**2.9) + 1.044 * (zeta[i]**0.3 / y[i]**0.049),
                    zeta[i]    >= 0.5 * ((rho_TO * S_w * C_D[i]) / T_TO),
                    V_TO[i]    == 1.2 * ((2 * W_max[i]) / (c_Lwmax * S_w *rho_TO))**0.5,
                    ]


# System Level Properties
for i in [0,1,2]:
    Constraints += [
                W_end[i] * x_CG[i]        >= W_wing * (x_wing[i] + delt_xacw[i]) +
                                       W_fuelprim[i] * (f_fuel[i] + f_fuelres) * (x_wing[i] + delt_xacw[i] * f_fuel[i]) +
                                       0.5 * (W_fuse + W_pay) * l_fuse + W_ht * x_CGht[i] + W_vt * x_CGvt[i] +
                                       n_eng * W_eng * x_b[i] + W_lg * x_CGlg[i] + W_misc * x_CGmisc[i],
                # x_CG[i] >= 1,
                f_fuel[i]              >= (W_fuel[0] + W_fuel[1] + W_fuel[2]) / W_fuelprim[i],
                W_lg * x_CGlg[i]       >= W_mg * x_mg[i] + W_ng * x_ng[i],
                W_misc * x_CGmisc[i]   >= W_hpesys * x_hpesys[i],
                x_hpesys[i] >= 1,
                I_z[i]                 >= I_zwing[i] + I_zfuse[i] + I_ztail[i],
                I_zwing[i]             >= ((n_eng * W_eng * y_eng**2) / g) + 
                                       ((W_fuelwing[i] + W_wing) / g) * ((c_rootw * b_w**3) / (16*S_w)) * (lambda_w + (1/3)),
                I_zfuse[i]             >= ((W_fuse + W_pay) / g) * ((x_wing[i]**3 + l_vt[i]**3) / (3*l_fuse)),
                # I_zfuse[i] >= 1, l_vt[i] >= 1,
                I_ztail[i]             >= ((W_apu + W_vt + W_ht) / g) * l_vt[i]**2,
                # I_ztail[i] >= 1,
                ]

# ========================
# Wing Model
# ========================

# Geometry
Constraints += [
                S_w         == b_w * ((c_rootw + c_tipw) / 2),
                # b_w        >= 1,
                p_w         == 1 + 2*lambda_w,
                q_w         == 1 + lambda_w,
                cbar_w      >= (2/3) * ((1 + lambda_w + lambda_w**2) / q_w) * c_rootw,
                y_cbarw     == (b_w * q_w) /(3 * p_w),
                lambda_w    == c_tipw / c_rootw,
                lambda_w    >= lambda_wmin,
                b_w         <= b_wmax,
                ]

# Wing Lift
for i in [0,1,2]:
    Constraints += [
                L_total[i]             >= W_avg[i] + L_ht[i],
                # L_ht[i] >= 1,
                L_total[i]             == f_Ltotw * L_w[i],
                L_w[i]                 == 0.5 * rho_free[i] * S_w * C_Lw[i] * V_free[i]**2 - delt_Lo[i]  - delt_Lt[i]*2,
                # 0.5 * rho_free[i] * S_w * C_Lw[i] * V_free[i]**2 >= L_w[i] + delt_Lo[i] + 2 * delt_Lt[i], Ignore this for now
                delt_Lo[i]             == eta_o[i] * f_Lo * p_o[i] * b_w *0.5,
                # eta_o[i] >= 1,
                # p_o[i] >= 1,
                delt_Lt[i]             == f_Lt * p_o[i] * c_rootw * lambda_w**2,
                C_Lw[i]                == C_Lwa[i] * alpha_w[i],
                alpha_w[i]             <= alpha_wmax,
                # C_Lwa[i]               == (2*pi * AR_w) / (2 + ((AR_w / eta_w)**2 * (1 + tan_chord**2 - M[i]**2) + 4)**0.5), IGnore this for now
                4 * pi**2           == (C_Lwa[i]**2 / eta_w**2) * (1 + tan_chord**2 - M[i]**2) + ((8 * pi * C_Lwa[i]) / AR_w),
                f_Ltotw * L_wmax[i]    >= N_lift * W_max[i] + L_htmax[i], 
                W_s[i]                 == 0.5 * rho_free[i] * C_Lw[i] * V_free[i]**2,
                W_s[i]                 <= W_smax,
                ]

# Wing Weight
for i in [0,1,2]:
    Constraints += [
                W_wing        >=  W_structw[i] * (1 + f_flap + f_slat + f_aileron + f_lete + f_ribs + f_spoiler + f_watt),
                W_structw[i]     >= W_cap + W_web,
                W_cap         >= (8 * rho_cap * g * r_wc * t_cap[i] * v * S_w**1.5) / (3 * AR_w**0.5),
                W_web         >= (8 * rho_web * g * r_h * tau_w[i] * t_web[i] * v * S_w**1.5) / (3 * AR_w**0.5),
                v**3.94       >= (0.14 * p_w**0.54) + (0.84 / p_w**2.4),
                p_w           >= 1 + 2 * lambda_w,
                2 * q_w       >= 1 + p_w,
                8             >= (AR_w * M_r[i] * N_lift * tau_w[i] * q_w**2) / (I_cap[i] * S_w * sigma_max),
                12            >= (AR_w * L_wmax[i] * N_lift * q_w**2) / (S_w * sigma_maxsh * tau_w[i] * t_web[i]),
                AR_w         == (b_w**2) / S_w,
                tau_w[i] <= 0.14,
                M_r[i]           >= (AR_w * L_wmax[i] * p_w) / 24, # Does not run without this for some reason
                M_r[i] * c_rootw >= (L_wmax[i] - N_lift * (W_wing + f_fuelw * W_fueltotal[i])) * ((b_w / 12) * (c_rootw + 2 * c_tipw)) * (b_w / S_w) - N_lift * W_eng * y_eng, 
                ]

# Wing Drag
for i in [0,1,2]:
    Constraints += [
                D_w[i]         == 0.5 * rho_free[i] * S_w * C_Dw[i] * V_free[i]**2,
                C_Dw[i]        >= C_Dpw[i] + C_Diw[i],
                C_Dpw[i]**1.65 >= 1.61 * (Re_w[i] / 1000)**-0.55 * tau_w[i]**1.29 * (M[i] * cos_chord)**3.04 * C_Lw[i]**1.78 + 
                               0.0466 * (Re_w[i] / 1000)**0.389 * tau_w[i]**0.784 * (M[i] * cos_chord)**-0.34 * C_Lw[i]**0.951 +
                               191 * (Re_w[i] / 1000)**-0.219 * tau_w[i]**3.95 * (M[i] * cos_chord)**19.3 * C_Lw[i]**-1.44 + 
                               2.82 * e[i] - 12 * (Re_w[i] / 1000)**1.18 * tau_w[i]**-1.76 * (M[i] * cos_chord)**0.105 * C_Lw[i]**-1.44,
                Re_w[i]        == (rho_free[i] * V_free[i] * cbar_w) / mu[i],
                C_Diw[i]       >= f_tip * (C_Lw[i]**2 / (pi * e[i] * AR_w)),
                # e[i]           <= 1 / (1 + (AR_w * f_lambdaw[i])), ignore this for now
                e[i] *(1 + (AR_w * f_lambdaw[i]))           <= 1,
                f_lambdaw[i]   >= 0.0524 * lambda_w**4 * 0.15 * lambda_w**3 + 0.1659 * lambda_w**2 - 0.0706 * lambda_w + 0.0119,
                # f_lambdaw[i] >= 1,
                ]

# Wing Aerodynamic Center
for i in [0,1,2]:
    Constraints += [
                delt_xacw[i]   >= (1/4) * tan_chord * AR_w * c_rootw * ((1/3) + (2/3) * lambda_w),
                ]

# Fuel Volume
for i in [0,1,2]:
    Constraints += [
                V_fuelmax[i]   <= 0.303 * cbar_w**2 * b_w * tau_w[i],
                W_fuelwing[i]  <= rho_fuel * V_fuelmax[i] * g,
                W_fuelwing[i]  >= (f_fuelw * W_fueltotal[i]) / f_fueluse,
                ]

# ===================
# Vertical Tail Model
# ===================

# Vertical Tail Geometry and Structure
for i in [0,1,2]:
    Constraints += [
                l_vt[i]        <= delt_xlvt[i] + z_cbarvt[i] * tan_chdvt + 0.25 * cbar_vt,
                # cbar_vt        >= 1,
                # z_cbarvt[i] >= 1,
                delt_xtvt[i]   >= delt_xlvt[i] + c_rootvt,
                # delt_xlvt[i] >= 1,
                l_fuse      >= x_CG[i] + delt_xtvt[i],
                x_CGvt[i]      >= x_CG[i] + 0.5 * (delt_xlvt[i] + delt_xtvt[i]),
                # delt_xtvt[i] >=1,
                L_vtmax[i]     == 0.5 * rho_TO * S_vt * c_Lvtmax * V_ne**2,
                # Include geometry equations for vertical tail
                S_vt >= b_vt * ((c_rootvt + c_tipvt) / 2),
                cbar_vt == (2/3) * ((1 + lambda_vt + lambda_vt**2) / q_vtd) * c_rootvt,
                y_cbarvt == (b_vt * q_vtd) / (3 * p_vt),
                lambda_vt == c_tipvt / c_rootvt,
                lambda_vt >= lambda_vtmin,
                ]

# Vertical Tail Weight
for i in [0,1,2]:
    Constraints += [
                W_vt        >=  W_structvt[i] * (1 + f_flap + f_lete + f_ribs + f_spoiler + f_watt), # Need to update with correct terms, rudder, etc.
                W_structvt[i]     >= W_capvt + W_webvt,
                W_capvt         >= (8 * rho_cap * g * r_vtwc * t_cap[i] * v * S_vt**1.5) / (3 * AR_vt**0.5),
                # r_vtwc >= 1, # Check after to remove
                # t_cap[i] >= 1, # Check after to remove
                W_webvt         >= (8 * rho_web * g * r_vth * tau_vt[i] * t_web[i] * vvt * S_vt**1.5) / (3 * AR_vt**0.5),
                # r_vth >= 1, # Check after to remove
                vvt**3.94       >= (0.14 * p_vt**0.54) + (0.84 / p_vt**2.4), # Don't know if a different dummy variable is needed here or we can reuse from wing
                p_vt           >= 1 + 2 * lambda_vt,
                2 * q_vtd       >= 1 + p_vt,
                # 0.4232 * tau_rvtc[i]**2 * t_cap[i] * r_vtwc >= 0.92 * tau_rvtc[i] * t_cap[i]**2 * r_vtwc + I_capvt[i], # What is tau_rvtc?
                8             >= (AR_vt * M_rvt[i] * N_lift * tau_vt[i] * q_vtd**2) / (I_capvt[i] * S_vt * sigma_max),
                12            >= (AR_vt * L_vtmax[i] * N_lift * q_vtd**2) / (S_vt * sigma_maxsh * tau_vt[i] * t_web[i]),
                AR_vt          == (b_vt**2) / S_vt,
                tau_vt[i] <= 0.14, # Check after to remove
                M_rvt[i]           >= (AR_vt * L_vtmax[i] * p_vt) / 24, # I believe this constraint is needed since its not carrying engines
                # M_rvt[i] * c_rootvt >= (L_vtmax[i]) * ((b_vt / 12) * (c_rootvt + c_tipvt)) * (b_w / S_w), #Not this one
    ]

# Engine-Out Condition
for i in [0,1,2]:
    Constraints += [
                L_vtEO[i] * l_vt[i]   >= D_wm[i] * y_eng + T_TO * y_eng,
                L_vtEO[i]          == 0.5 * rho_TO * S_vt * C_LvtEO[i] * V1**2,
                c_lvt_EO        >= C_LvtEO[i] * (1 + (c_lvt_EO / (pi * e_vt * AR_vt))),
                # C_LvtEO[i] >= 1,
                D_wm[i]            >= 0.5 * rho_TO * A_fan * c_Dwm * V1**2,
                ]

# Crosswind Landing Condition
for i in [0,1,2]:
    Constraints += [
                rdot_req / I_z[i]  <= 0.5 * rho_TO * S_vt * l_vt[i] * C_Lvtland[i] * V_land**2,
                ]

# Vertical Tail Drag
for i in [0,1,2]:
    Constraints += [
                D_vt[i]            >= 0.5 * rho_free[i] * S_vt * C_Dpvt[i] * V_free[i]**2,
                C_Dpvt[i]**1.189   >= 2.44e-77 * Re_vt[i]**-0.528 * tau_vt[i]**133.8 * M[i]**1022.7 + 
                                   0.003 * Re_vt[i]**-0.41 * tau_vt[i]**1.22 * M[i]**1.55 + 
                                   1.967e-4 * Re_vt[i]**0.214 * tau_vt[i]**-0.04 * M[i]**-0.14 + 
                                   6.59e-50 * Re_vt[i]**-0.498 * tau_vt[i]**1.56 * M[i]**114.6,
                Re_vt[i]           == (rho_free[i] * V_free[i] * cbar_vt) / mu[i],
                # mu[i] >= 1,
                ]

# =====================
# Horizontal Tail Model
# =====================

# Horizontal Tail Geometry and Structure

Constraints += [
                S_ht         == b_ht * ((c_rootht + c_tipht) / 2),
                # S_ht >= 1,
                # b_ht        >= 1,
                # c_rootht      >= 1,
                # c_tipht       <= 1,
                cbar_ht     >= (2/3) * ((1 + lambda_ht + lambda_ht**2) / q_htd) * c_rootht,
                # cbar_ht >= 1,
                y_cbarht    == (b_ht * q_htd) / (3 * p_ht),
                # y_cbarht >= 1,
                lambda_ht    == c_tipht / c_rootht,
                lambda_ht >= lambda_htmin,
    ]

for i in [0,1,2]:
    Constraints += [
        x_CGht[i]      >= x_CG[i] + 0.5 * (delt_xlht[i] + delt_xtht[i]),
        L_htmax[i] == 0.5 * rho_TO * S_ht * c_Lhtmax * V_ne**2,
        L_ht[i] * l_ht >= D_wm[i] * y_eng + T_TO * y_eng,
        L_ht[i]          == 0.5 * rho_TO * S_ht * C_Lht[i] * V1**2,
    ]

# Horizontal Tail Weight
for i in [0,1,2]:
    Constraints += [
                W_ht        >=  W_structht[i] * (1 + f_flap + f_ribs + f_watt), # Need to update with correct terms, elevator, etc.
                W_structht[i]     >= W_capht + W_webht,
                W_capht       >= (8 * rho_cap * g * r_htc* t_cap[i] * v * S_ht**1.5) / (3 * AR_ht**0.5),
                r_htc >= 1, # Check after to remove
                # t_cap[i] >= 1, # Check after to remove
                W_webht       >= (8 * rho_web * g * r_hht * tau_ht[i] * t_web[i] * vht * S_ht**1.5) / (3 * AR_ht**0.5),
                # r_hht >= 1, # Check after to remove
                vht**3.94       >= (0.14 * p_ht**0.54) + (0.84 / p_ht**2.4),
                p_ht           >= 1 + 2 * lambda_ht,
                2 * q_htd       >= 1 + p_ht,
                # # 0.4232 * tau_rhtc[i]**2 * t_cap[i] * r_htc >= 0.92 * tau_rhtc[i] * t_cap[i]**2 * r_htc + I_capht[i], # What is tau_rwc?
                8             >= (AR_ht * M_rht[i] * N_lift * tau_ht[i] * q_htd**2) / (I_capht[i] * S_ht * sigma_max),
                12            >= (AR_ht * L_htmax[i] * N_lift * q_htd**2) / (S_ht * sigma_maxsh * tau_ht[i] * t_web[i]),
                AR_ht          == (b_ht**2) / S_ht,
                tau_ht[i] <= 0.14, # Check after to remove
                M_rht[i]           >= (AR_ht * L_htmax[i] * p_ht) / 24, # I believe this constraint is not needed since its replaced by the following one
                M_rht[i] * c_rootht >= (L_htmax[i]) * ((b_ht / 12) * (c_rootht + 2 * c_tipht)) * (b_ht / S_ht), # No engines on HT. No fuel storage on HT 
    ]

# Trim Condition
for i in [0,1,2]:
    Constraints += [
                x_w[i] / cbar_w    <= (x_CG[i] / cbar_w) + (c_mac / C_Lw[i]) + ((V_ht * C_Lht[i]) / C_Lw[i]),
                C_Lht[i]           == C_Lalphaht[i] * alpha_ht[i],
                C_Lalphaht[i] >= 1,
                C_Lalphaht[i]      == C_Lalphaht0[i] * eta_ht * (1 - ((2 * C_Lwa[i]) / (pi * AR_w))),
                C_Lalphaht0[i] * eta_ht    >= C_Lalphaht[i] + eta_ht * C_Lalphaht0[i] * ((2 * C_Lwa[i]) / (pi * AR_w)),
                ]

# Minimum Stability Margin
for i in [0,1,2]:
    Constraints += [
                SM_min + (delt_xCG / cbar_w) + (c_mac / c_Lwmax)    <= V_ht * m_ratio[i] + ((V_ht * c_Lhtmax) / c_Lwmax),
                1 + (2 / AR_ht)  == m_ratio[i] * (1 + (2 / AR_w)),
                ]

# Stability Margin
for i in [0,1,2]:
    Constraints += [
                SM[i]  <= (x_w[i] - x_CG[i]) / cbar_w,
                SM[i]  >= SM_min,
                ]

# Horizontal Tail Drag
for i in [0,1,2]:
    Constraints += [
                D_ht[i]            >= 0.5 * rho_free[i] * S_ht * C_Dht[i] * V_free[i]**2,
                C_Dht[i]        >= C_D0ht[i] + C_Diht[i],
                C_D0ht[i]**6.49    >= 5.288e-20 * Re_ht[i]**0.901 * tau_ht[i]**0.912 * M[i]**8.645 + 
                                   1.676e-28 * Re_ht[i]**0.351 * tau_ht[i]**6.292 * M[i]**10.256 +
                                   7.098e-25 * Re_ht[i]**1.395 * tau_ht[i]**1.962 * M[i]**0.567 +
                                   3.731e-14 * Re_ht[i]**-2.574 * tau_ht[i]**3.128 * M[i]**0.448 +
                                   1.443e-12 * Re_ht[i]**-3.91 * tau_ht[i]**4.663 * M[i]**7.689,
                Re_ht[i]           == (rho_free[i] * V_free[i] * cbar_ht) / mu[i],
                C_Diht[i]       >= f_tip * (C_Lht[i]**2 / (pi * e[i] * AR_ht)),
                e[i] *(1 + (AR_ht * f_lambdaht[i]))           <= 1,
                f_lambdaht[i]   >= 0.0524 * lambda_ht**4 * 0.15 * lambda_ht**3 + 0.1659 * lambda_ht**2 - 0.0706 * lambda_ht + 0.0119,

                ]

# ==============
# Fuselage Model
# ==============
# Cross-Sectional Geometry
Constraints += [
                wdt_fuse    >= n_spr * w_seatwdth + w_aisle + 2 * w_sys,
                A_skin      >= 2 * pi * R_fuse * t_skin,
                A_fuse      >= pi * R_fuse**2,
                # R_fuse >= 1,
                ]

# Pressure Loading
for i in [0,1,2]:
    Constraints += [
                sigma_x[i]     == 0.5 * delt_Pover * (R_fuse / t_shell),
                sigma_thta[i]  == delt_Pover * R_fuse / t_skin,
                sigma_skin  >= sigma_x[i],
                sigma_skin  >= sigma_thta[i],
                ]

# Floor Loading
for i in [0,1,2]:
    Constraints += [
                P_floor[i]     >= N_land * (W_pay + W_seats),
                S_floor[i]     == P_floor[i] / 2,
                M_floor[i]     == (1/8) * P_floor[i] * wdt_floor,
                A_floor     >= 1.5 * (S_floor[i] / tau_floor) + 2 * (M_floor[i] / (sigma_floor * h_floor)),
                ]

# Shell Geometry
Constraints += [
                x_shell1    == l_nose,
                x_shell2    >= l_nose + l_shell,
                n_seats     == n_spr * n_rows,
                n_pass      == n_seats,
                l_shell     >= n_rows * p_s,
                l_floor     >= 2 * R_fuse + l_shell,
                l_fuse      == l_nose + l_shell + l_cone,
                x_fwb       <= x_wing[i] + 0.5 * c0 * r_wc,
                x_wing[i]      <= x_b[i] + 0.5 * c0 * r_wc,
                S_nose**(8/5)   >= (2 * R_fuse**2)**(8/5) * ((1/3) + (2/3) * (l_nose / R_fuse)**(8/5)),
                S_bulk          == 2 * pi * R_fuse**2,
                V_cyl           == A_skin * l_shell,
                V_nose          == S_nose * t_skin,
                V_bulk          == S_bulk * t_skin,
                V_cabin         >= A_fuse * ((2/3) * l_nose + l_shell + (2/3) * R_fuse),
                # R_fuse >= 1,
                ]

# Tail Cone
Constraints += [
                # Q_vt        == ((L_vtmax[i] * b_vt) / 3) * ((1 + 2 * lambda_vt) / (1 + lambda_vt)),
                Q_vt *3 * (1 + lambda_vt) == L_vtmax[i] * b_vt * (1 + 2 * lambda_vt),
                t_cone      == Q_vt / (2 * A_fuse * tau_cone),
                # Q_vt >= 1,
                R_fuse * tau_cone * (1 + p_vt) * V_cone * ((1 + lambda_cone) / (4 * l_cone)) >= L_vtmax[i] * b_vt * p_vt * (1/3),
                # p_vt >= 1,
                tau_cone    == sigma_skin,
                l_cone  == R_fuse / lambda_cone,
                ]
# Review this section. Especially Q_vt, lambda_vt, t_cone. All variables

# Fuselage Area Moment of Inertia
Constraints += [
                t_shell     >= t_skin * (1 + f_string * (rho_skin / rho_bend)),
                sigma_bend  >= sigma_Mh + ((delt_Pover * R_fuse) / (2 * t_shell)),
                sigma_bend  >= sigma_Mv[i] + ((delt_Pover * R_fuse) / (2 * t_shell)),
                I_hshell    <= pi * t_shell * R_fuse**3,
                I_vshell    <= pi * t_shell * R_fuse**3,
                ]
# Redo: figure out what the r_E is. Check the TASOPT doc.

# Horizontal Bending Model
for i in [0,1,2]:
    Constraints += [
                A0h     == I_hshell / (h_fuse**2),
                A1hland >= N_lift * ((W_vt + W_ht + W_apu) / (h_fuse * sigma_Mh)),
                A1hMLF  >= N_lift * ((W_vt + W_ht + W_apu + R_Mh * L_htmax[i]) / (h_fuse * sigma_Mh)),
                # L_htmax[i] >= 1,
                A2hland >= (N_land / (2 * l_shell * h_fuse * sigma_bend)) * (W_pay + W_padd + W_shell + W_window + W_insul + W_floor + W_seats),
                A2hMLF  >= (N_lift / (2 * l_shell * h_fuse * sigma_Mh)) * (W_pay + W_padd + W_shell + W_window + W_insul + W_floor + W_seats),
                # A2hMLF  >= 1,
#                 # Review COnstraints 173-180 here
                A0h     == A2hMLF * (x_shell2 - x_hbndMLF)**2 + A1hMLF * (x_tail[i] - x_hbndMLF),
                x_hbndMLF >= x_wing[i],
                x_hbndMLF <= l_fuse,
                A0h     == A2hland * (x_shell2 - x_hbndld)**2 + A1hland * (x_tail[i] - x_hbndld),
                x_hbndld >= x_wing[i],
                x_hbndld <= l_fuse,
                AhbndfMLF >= A2hMLF * (x_shell2 - x_fwb)**2 + A1hMLF * (x_tail[i] - x_fwb) - A0h,
                # AhbndfMLF >= 1,
                AhbndbMLF >= A2hMLF * (x_shell2 - x_bwb)**2 + A1hMLF * (x_tail[i] - x_bwb) - A0h,
                # AhbndbMLF >= 1,
                Ahbndfld >= A2hland * (x_shell2 - x_fwb)**2 + A1hland * (x_tail[i] - x_fwb) - A0h,
                # Ahbndfld >= 1,
                Ahbndbld >= A2hland * (x_shell2 - x_bwb)**2 + A1hland * (x_tail[i] - x_bwb) - A0h,
                # Ahbndbld >= 1,
                V_hbndf >= (A2hMLF / 3) * ((x_shell2 - x_fwb)**3 - (x_shell2 - x_hbndMLF)**3) + (A1hMLF / 2) * ((x_tail[i] - x_fwb)**2 - (x_tail[i] - x_hbndMLF)**2) - A0h * (x_hbndMLF - x_bwb),
                # V_hbndf >= 1,
                V_hbndb >= (A2hMLF / 3) * ((x_shell2 - x_bwb)**3 - (x_shell2 - x_hbndMLF)**3) + (A1hMLF / 2) * ((x_tail[i] - x_bwb)**2 - (x_tail[i] - x_hbndMLF)**2) - A0h * (x_hbndMLF - x_bwb),
                # V_hbndb >= 1,
                V_hbndc >= 0.5 * (AhbndfMLF + AhbndbMLF) * c0 * r_wc,

                V_hbndf >= (A2hland / 3) * ((x_shell2 - x_fwb)**3 - (x_shell2 - x_hbndld)**3) + (A1hland / 2) * ((x_tail[i] - x_fwb)**2 - (x_tail[i] - x_hbndld)**2) - A0h * (x_hbndld - x_bwb),
                V_hbndb >= (A2hland / 3) * ((x_shell2 - x_bwb)**3 - (x_shell2 - x_hbndld)**3) + (A1hland / 2) * ((x_tail[i] - x_bwb)**2 - (x_tail[i] - x_hbndld)**2) - A0h * (x_hbndld - x_bwb),
                V_hbndc >= 0.5 * (Ahbndfld + Ahbndbld) * c0 * r_wc,

                V_hbend[i] >= V_hbndc + V_hbndf + V_hbndb,
]

# Vertical Bending Model
for i in [0,1,2]:
    Constraints += [
                B1v[i]     == (R_Mv * L_vtmax[i]) / (R_fuse * sigma_Mv[i]),
                B0v[i]     == I_vshell / R_fuse**2,
                B0v[i]     == B1v[i] * (x_tail[i] - x_vbend[i]),
                x_vbend[i] >= x_wing[i],
                x_vbend[i] <= l_fuse,
                Avbndb[i]  >= B1v[i] * (x_tail[i] - x_b[i]) - B0v[i],
                V_vbndb[i] >= 0.5 * B1v[i] * ((x_tail[i] - x_b[i])**2 - (x_tail[i] - x_vbend[i])**2) - B0v[i] * (x_vbend[i] - x_b[i]),
                V_vbndc[i] >= 0.5 * Avbndb[i] * c0 * r_wc,
                # Avbndb[i] >= 1,
                # c0 >= 1,
                V_vbend[i] >= V_vbndb[i] + V_vbndc[i],
                ]

# Fuselage Weight
for i in [0,1,2]:
    Constraints += [
                W_fuse  >= W_apu + W_cone + W_floor + W_hbend[i] + W_vbend[i] + W_insul + W_padd + W_seats + W_shell + W_window + W_fix,
                W_skin  >= rho_skin * g * (V_bulk + V_cyl + V_nose),
                W_shell >= W_skin * (1 + f_fadd + f_frame + f_string),
                V_floor >= A_floor * wdt_floor,
                # A_floor >= 1, 
                W_floor >= V_floor * rho_floor * g + d2W_floor * l_floor * wdt_floor,
                W_cone  >= rho_cone * g * V_cone * (1 + f_fadd + f_frame + f_string),
                W_hbend[i] >= rho_bend * g * V_hbend[i],
                # V_hbend[i] >= 1,
                W_vbend[i] >= rho_bend * g * V_vbend[i],
                # V_vbend[i] >= 1,
                W_window == dW_window * l_shell,
                # l_shell >= 1,
                W_insul >= d2W_insul * (0.55 * (S_bulk + S_nose) + 1.1 * pi * R_fuse * l_shell),
                W_apu   == W_pay * f_apu,
                W_padd  == W_pay * f_padd,
                W_seats == W_seat * n_seats,
                # n_seats >= 1,
                rho_cabin[i]  == p_cabin / (R_sheat * T_cabin),
                W_buoy[i]  == rho_cabin[i] * g * V_cabin,
                W_pay   >= W_avgpass * n_pass,
                ]

# Fuselage Drag
for i in [0,1,2]:
    Constraints += [
                D_fuse[i]  == 0.5 * rho_free[i] * C_Dfuse[i] * V_free[i]**2 * (l_fuse * R_fuse * (M[i]**2 / M_fuseD**2)),
                ]

# ==================
# Landing Gear Model
# ==================

# Landing Gear Position
for i in [0,1,2]:
    Constraints += [
                T_lg[i]    == 2 * y_m[i],
                x_m[i]     >= x_n[i] + B,
                x_n[i] + delt_xn[i] == x_CG[i],
                x_CG[i] + delt_xm[i] == x_m[i],
                # x_n[i] + delt_xn[i] >= x_CG[i], Ignore this for now 
                # x_CG[i] + delt_xm[i] >= x_m[i], Ignore this for now
                y_m[i]     >= l_mgear,
                y_m[i]     <= y_eng,
                ]

# Wing Vertical Position and Engine Clearance
for i in [0,1,2]:
    Constraints += [
                l_ngear + z_wing + y_m[i] * tan_gamma >= l_mgear,
                l_mgear + (y_eng - y_m[i]) * tan_gamma    >= d_nacelle + h_nacelle,
                d_nacelle   >= d_fan + 2 * t_nacelle,
                # d_fan       >= 1,
                ]

# Takeoff Rotation
for i in [0,1,2]:
    Constraints += [
                l_mgear / tan_thtmax   >= x_up[i] - x_m[i],
                ]

# Tip-Over Criteria
for i in [0,1,2]:
    Constraints += [
                x_m[i]     >= (l_mgear + z_CG)*tan_phi[i] + x_CG[i],
                tan_phi[i] >= tan_phimin,
                1       >= ((z_CG + l_mgear)**2 * (y_m[i]**2 + B**2)) / ((delt_xn[i] * y_m[i] * tan_psi[i])**2),
                tan_psi[i] <= tan_psimax,
                ]

# Landing Gear Weight
for i in [0,1,2]:
    Constraints += [
                W_lg    >= W_mg + W_ng,
                W_mg    >= n_mg * (W_ms + W_mw * (1 + f_addm)),
                W_ng    >= W_ns + W_nw * (1 + f_addn),
                W_ms    >= 2 * pi * r_m * t_m * l_mgear * rho_ST * g,
                W_ns    >= 2 * pi * r_n * t_n * l_ngear * rho_ST * g,
                2 * pi * r_m * t_m * sigma_yc >= (lambda_LG * L_m[i] * N_s) / n_mg,
                2 * pi * r_n * t_n * sigma_yc >= (L_n[i] + L_ndyn[i]) * N_s,
                L_m[i]     <= (pi**2 * E * I_m[i]) / (K**2 * l_mgear**2),
                I_m[i]     == pi * r_m**3 * t_m,
                L_n[i]     <= (pi**2 * E * I_n[i]) / (K**2 * l_ngear**2),
                I_n[i]     == pi * r_n**3 * t_n,
                40      >= (2 * r_m) / t_m,
                40      >= (2 * r_n) / t_n,
                W_mw    == n_wps * W_wam,
                W_nw    == n_wps * W_wan,
                W_wam   == 1.2 * F_wm[i]**0.609,
                F_wm[i]    == L_wm[i] * d_tm,
                L_wm[i]    == L_m[i] / (n_mg * n_wps),
                W_wan   == 1.2 * F_wn[i]**0.609,
                F_wn[i]    == L_wn[i] * d_tn,
                L_wn[i]    == L_n[i] / n_wps,
                # L_n[i] >= 1,
                d_tm    == 1.63 * L_wm[i]**0.315,
                wdt_tm  == 0.104 * L_wm[i]**0.48,
                d_tn    == 0.8 * d_tm,
                wdt_tn  == 0.8 * wdt_tm,
                h_hold  >= 2 * wdt_tm + 2 * r_m,
                0.8     >= 2 * wdt_tn + 2 * r_n,
                ]

# Landing Gear Loads
for i in [0,1,2]:
    Constraints += [
                L_n[i]     == (W_max[i] * delt_xm[i]) / B,
                L_m[i]     == (W_max[i] * delt_xn[i]) / B,
                L_ndyn[i]  >= 0.31 * W_max[i] * ((l_mgear + z_CG) / B),
                0.05    <= L_n[i] / W_max[i],
                0.2     >= L_n[i] / W_max[i],
                ]

# Shock Absorbtion
for i in [0,1,2]:
    Constraints += [
                E_land[i]      >= (W_max[i] / (2 * g)) * w_ult**2,
                S_sa[i]        == (1 / eta_s) * (E_land[i] / (L_m[i] * lambda_LG)),
                l_oleo      == 2.5 * S_sa[i],
                # S_sa[i] >=1,
                d_oleo      == 1.3 * ((4 * lambda_LG * L_m[i] / n_mg) / (p_oleo * pi))**0.5,
                l_mgear     >= l_oleo + (d_tm / 2),
                ]

# ===========================
# Michael Code for displaying
# ===========================

f.ConstraintList(Constraints)
var_list = f.get_variables()
con_list = f.get_constants()

variable_names = []
variable_values = []
variable_units = []
variable_description = []

constant_names = []
constant_values = []
constant_units = []
constant_description = []

ctr = 0

from edi.solvers.solver import cvxopt_solve
res = cvxopt_solve(f)
# print(res)

import pandas as pd
for var in var_list:
    if isinstance(var, pyomo.core.base.var.IndexedVar):
        for ix in var.index_set():
            lbls = ['out', 'ret', 'sprint']
            variable_names.append(f'{var.name}[{lbls[ix]}]')
            variable_values.append(res['x'][ctr])
            variable_units.append(var._units)
            variable_description.append(var.doc)
            ctr += 1
    else:
        variable_names.append(var.name)
        variable_values.append(res['x'][ctr])
        variable_units.append(var._units)
        variable_description.append(var.doc)
        ctr += 1

pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.4f}'.format
vdf = pd.DataFrame({
    'Variable Name': variable_names,
    'Value': variable_values,
    'Units': variable_units,
    'Description': variable_description
})

print(vdf)

ctr2 = 0

for con in con_list:
    constant_names.append(con.name)
    constant_values.append(con.value)
    constant_units.append(con._units)
    constant_description.append(con.doc)
    ctr += 1

pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.4}'.format
cdf = pd.DataFrame({
    'Constant Name': constant_names,
    'Value': constant_values,
    'Units': constant_units,
    'Description': constant_description
})

print(cdf)