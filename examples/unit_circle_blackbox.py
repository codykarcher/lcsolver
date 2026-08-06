# =================
# Import Statements
# =================
import lcsolver
from lcsolver import Formulation, BlackBoxFunctionModel, units

# ===================
# Declare Formulation
# ===================
f = Formulation()

# =================
# Declare Variables
# =================
x = f.Variable(name='x', guess=1.0, units='m'  , description='The x variable')
y = f.Variable(name='y', guess=1.0, units='m'  , description='The y variable')
z = f.Variable(name='z', guess=1.0, units='m^2', description='Model output')

# =================
# Declare Constants
# =================
c = f.Constant(name='c', value=[1.0, 2.0], units='', size=2, description='A constant c')

# =====================
# Declare the Objective
# =====================
f.Objective(c[0] / x + c[1] / y)

# ===================
# Declare a Black Box
# ===================
class UnitCircle(BlackBoxFunctionModel):
    def __init__(self):  # The initialization function
        # Initialize the black box model
        super().__init__()

        # A brief description of the model
        self.description = 'This model evaluates the function: z = x**2 + y**2'

        # Declare the black box model inputs
        self.inputs.append(name='x', units='ft', description='The x variable')
        self.inputs.append(name='y', units='ft', description='The y variable')

        # Declare the black box model outputs
        self.outputs.append(
            name='z', units='ft**2', description='Resultant of the unit circle'
        )

        # Declare the maximum available derivative
        self.availableDerivative = 1

    def BlackBox(self, x, y):  # The actual function that does things
        # Convert to the declared input units (ft) and strip to plain floats
        x, y = self.sanitizeInputs(x, y, strip_units=True)

        z    = x**2 + y**2  # Compute z
        dzdx = 2 * x        # Compute dz/dx
        dzdy = 2 * y        # Compute dz/dy

        # Attach the declared units: z in ft**2, the gradient in ft**2/ft
        res = self.packOutputs(z, [dzdx, dzdy])

        return res

# =======================
# Declare the Constraints
# =======================
f.ConstraintList([
    [ z, '==', [x, y], UnitCircle() ], 
    x + y <= 1.0 * units.m
    ])

# =============================================
# Black Box can be run as a function!
# =============================================
uc = UnitCircle()
bbo = uc.BlackBox(0.5 * units.m, 0.5 * units.m)

# =======================
# Solve Model
# =======================
sol = lcsolver.solve(f)
print(sol.summary())

