# pycycle — a third engine reference

[pycycle](https://github.com/OpenMDAO/pycycle) is NASA/OpenMDAO's
thermodynamic cycle library: gradient-enabled OpenMDAO components for
compressor, turbine, combustor, nozzle, inlet, splitter, mixer, shaft and
duct, on top of CEA-based equilibrium thermodynamics.

It is useful here for two separate reasons.

## 1. As an engine reference

The SP engine formulation needs something to be checked against, and there
are now three independent sources that model the same physics at different
fidelities:

| source | form | fidelity |
|---|---|---|
| TASOPT `tfsize.f` / `tfoper.f` | Fortran, Newton-solved | 1D flowpath, variable-cp gas |
| York/Hoburg/Drela `turbofan` | gpkit SP | posynomial-fitted, 13 GP solves |
| **pycycle** | OpenMDAO, CEA thermo | equilibrium chemistry, highest |

Where the SP model and TASOPT agree but pycycle disagrees, the difference is
most likely in the thermodynamics (frozen vs equilibrium, cp tabulation).
Where the SP disagrees with *both*, it is the posynomial fit.

## 2. As a MAIDAS case

pycycle is a much better structure-detection target than a hand-written GP,
because it is *not* written with any convex form in mind. An OpenMDAO model
is a graph of explicit and implicit components with declared partials — the
question MAIDAS can ask is which parts of that graph are secretly monomial /
posynomial / signomial, and where the genuine obstructions are.

That is exactly the York result restated: TASOPT turned out to be almost
entirely SP-representable despite never having been written that way.
Repeating the analysis on pycycle would be independent evidence, and pycycle
is far easier to introspect than Fortran — the component graph, variable
metadata and partial-derivative structure are all available at runtime.

## Status

Installed and running. The `simple_turbojet` example converges:

```
Fn (lbf)      11799.99836
TSFC              0.79852
OPR              13.50000
FAR               0.01776
W (lbm/s)       147.33343
turb PR           3.85914
T4 (degR)      2370.00000
```

Nothing is ported or analyzed yet — this directory currently records only
that the reference works and how to drive it.

### Getting it running

```bash
pip install openmdao
git clone https://github.com/OpenMDAO/pycycle.git
cd pycycle && pip install -e .
```

The checkout used here lives at `~/Dropbox/research/_reference/pycycle` and
is installed editable from there.

**The example scripts' viewers are broken on modern numpy.** Running
`python simple_turbojet.py` gets all the way through the solve and then dies
in `viewer()` with

```
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

That is a formatting bug in the example's print helper, not the model — the
cycle has already converged by then. Drive the model directly and read values
with `prob.get_val(...)` rather than running the scripts as-is. The
`RuntimeWarning: invalid value encountered in sqrt` during setup is likewise
benign; it comes from evaluating the sonic-velocity residual before the flow
state is initialized.

### Driving it directly

```python
import numpy as np, openmdao.api as om
from simple_turbojet import MPTurbojet

prob = om.Problem(); mp = prob.model = MPTurbojet(); prob.setup(check=False)
prob.set_val('DESIGN.fc.alt', 0, units='ft')
prob.set_val('DESIGN.fc.MN', 0.000001)
prob.set_val('DESIGN.balance.Fn_target', 11800.0, units='lbf')
prob.set_val('DESIGN.balance.T4_target', 2370.0, units='degR')
prob.set_val('DESIGN.comp.PR', 13.5)
prob.set_val('DESIGN.comp.eff', 0.83)
prob.set_val('DESIGN.turb.eff', 0.86)
# balances need initial guesses -- see the example's __main__ block
prob.set_solver_print(level=-1)
prob.run_model()

fn = float(np.asarray(prob.get_val('DESIGN.perf.Fn', units='lbf')).ravel()[0])
```

Note the balance variables are named `balance.Fn_target` / `balance.T4_target`
in this version, not the `balance.rhs:FAR` form some older docs use.

## First MAIDAS result

`maidas_case.py` transcribes six pycycle `ExplicitComponent.compute()` methods
as plain functions (arithmetic copied verbatim, source line cited for each)
and runs MAIDAS structure detection over them:

| component | tightest form | why |
|---|---|---|
| `CorrectedInputsCalc` | monomial | ratios and half-powers only |
| `PressureRise` | monomial | `PR * Pt_in` |
| `ShaftPower` | monomial | `trq * Nmech` |
| `PressureLoss` | signomial | `Pt_in*(1 - dPqP)` — a subtraction |
| `EnthalpyRise` | signomial | `(ideal_ht - inlet_ht)/eff + inlet_ht` |
| `eff_poly_calc` | **not GP/SP** | `log(PR)` of a design variable |

Half the sampled components are monomial — trivially GP-compatible, usable as
equality constraints. Two more are signomial, needing an SP treatment or a
reformulation. Only one has a genuine obstruction: a logarithm of a design
variable, which no monomial form can express and which would have to be
fitted. That is precisely what the York turbofan model does.

This is a small sample and not yet a claim about pycycle as a whole, but it
is consistent with the York result: an engine model written with no convex
form in mind is mostly SP-representable, with the obstructions confined to a
few identifiable atoms.

### A MAIDAS gotcha worth knowing

`analyze()`'s **return shape is not fixed**. It gives back:

* a bare IR node for a single scalar output,
* a tuple for a multiple-return,
* an `IntermediateRepresentation` (dict) once tapping fires — whose `'_'`
  key holds the original return, *itself possibly a tuple*, alongside one
  entry per tapped intermediate.

Whether tapping fires depends on whether the traced function's module has
other user-defined functions in its globals to AST-rewrite. So the same
function analyzed from a script and from inside a module comes back with
different shapes — which is exactly how the first version of this script
broke. Consume the result by flattening recursively (see `_flatten`).
