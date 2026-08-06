import sys, warnings, time; warnings.filterwarnings("ignore")
sys.path.insert(0,'/Users/codykarcher/Desktop/spcomparisons')
sys.path.insert(0,'/Users/codykarcher/Desktop/spcomparisons/components')
sys.path.insert(0,'/Users/codykarcher/Dropbox/research/lcsolver')
import classes, architectures, aircraft
from lcsolver.presolve.structureDetector import structure_detector
from lcsolver.solvers.ipopt.slcp_bridge import solve_sia
from lcsolver.solvers.ipopt.sia import SIAOptions
from lcsolver.presolve.unitCorrector import unit_corrector
for ck, ak in (("b737","conventional"), ("b737","h2burn"), ("b787","conventional")):
    st = structure_detector(unit_corrector(
        aircraft.build(classes.CLASSES[ck], architectures.ARCHS[ak], seed="reference")))
    t0 = time.time()
    r = solve_sia(st, options=SIAOptions(max_iterations=800), presolve=False)
    print(f"RESULT {ck}/{ak} converged={r.converged} it={r.iterations} "
          f"{time.time()-t0:.0f}s {str(r.status)[:55]}", flush=True)
