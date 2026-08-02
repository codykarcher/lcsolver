# HANDOFF: SP-Native Rubber Engine (PyCycle physics as signomial constraints)

## Mission
Replace the deck-pinned turbofan in `components/turbofan/model.py` with a proper
rubber engine: the user sets a TECHNOLOGY LEVEL (component polytropic
efficiencies, T_metal, OPR bound) and key temperatures (Tt4 ratings), and the
cycle variables (OPR, FPR, BPR, corrected flows) are FREE design variables that
settle at physical optima inside the aircraft optimization. The approach is NOT
to wrap pycycle as a black box or surrogate: it is to REWRITE pycycle-fidelity
cycle physics directly as SP rows, with pycycle-the-code as the offline TRUTH
HARNESS (the same role TASOPT 2.16 played for the airframe, verified
station-by-station over the preceding sessions).

## Why this is feasible (established in this project)
- Variable-cp thermo: NASA 7-coefficient cp(T)/h(T) polynomials have mixed
  signs -> signomial equalities (move negative terms across). Real fidelity
  gain over the current constant-cp cycle (% level in TSFC).
- Combustion equilibrium is GP-NATIVE: the Gibbs-minimization optimality
  conditions are mass-action laws, products of x_i^nu_i == K(T) -- pure
  MONOMIALS -- plus linear element balances, with K(T) a monomial fit in T.
  (Chemical equilibrium is a founding GP application, Duffin/Peterson/Zener.)
  A jet-A/air burner with ~8-10 species is ~15-20 signomial rows. Dissociation
  at Tt4 ~ 1900 K is the point of doing this.
- Balances (shaft power, mass, nozzle areas): already-SP algebra.
- Off-design maps: pycycle uses scaled empirical TABLES -- data, not math. Fit
  them as posynomials (the existing m_tild/pi_tild rows are this pattern with
  worse provenance). Fitting pycycle's scaled maps is NOT a fidelity
  concession -- pycycle interpolates the same tables.
- Conditionals (choked/unchoked nozzles, mixer regimes): fix the branch per
  mission segment a priori (TASOPT hard-codes this via ichoke5/7; transports
  are known: fan nozzle near-choked at cruise, core unchoked), or use the
  one-sided-row trick (see tfcool below).

## Milestones
1. Stand up pycycle locally; reproduce a CFM56-class and a GEnx-class engine
   at design + 4 off-design points. These are the anchor points. This
   milestone is pure validation infrastructure -- zero risk to baselines --
   and should be delivered alone before anything else.
2. SP on-design cycle: variable-cp station chain + mass-action burner +
   cooling + balances, as a NEW module (do not mutate the existing model in
   place; the user prefers separate modules over conditionals, one master
   switch at the top). Validate each station's (Tt, Pt, ht) against pycycle
   at the anchors.
3. Off-design: posynomial fits of pycycle's scaled maps over the flight
   envelope, multi-rating structure (takeoff/climb/cruise per mission
   segment; the mission has 5 segments).
4. Integration: swap in behind an engine-model switch. The Fitzgerald weight
   model ALREADY keys on the right interface -- design corrected core flow
   (mbar_fan_D / BPR_D; wsize.f:1306 provenance is documented in model.py) --
   keep it. Keep tfcool (metal-temperature cooling, search "tfcool" in
   model.py) or absorb its physics into the new burner/turbine chain.
5. Acceptance: with cycle variables FREE, the 737-class engine must settle
   near the CFM56 deck point and the 787-class near GEnx WITHOUT deck pins.
   History: when FPR was freed without the fan-efficiency-lapse coupling, it
   ran to 1.889 vs 1.605 -- missing couplings get exploited. PyCycle's physics
   closes them; verify the solved free-cycle optimum against a pycycle
   optimization at the same tech level.

## Repo geography
- Repo: ~/Dropbox/research/edi, branch `convexengineering-rebuild`, remote
  `origin` (github.com/codykarcher/edi). NEVER add Co-Authored-By trailers.
  Commit style: long-form messages explaining WHY with measured evidence
  (read `git log` for the house voice). Push after committing.
- Aircraft model: `examples/spcomparisons/aircraft.py` (~3,000 lines).
- Engine: `examples/spcomparisons/components/turbofan/model.py` -- SUBS decks
  (CFM56, TASOPT_737800, GE90, D82_SPaircraft...), ETAS efficiency table,
  station naming 0/1.8/2/2.1/2.5/3/4/4a/4.1/4.5/4.9/5/6/7/8, per-segment
  arrays size N=5.
- Solver: SIA via `edi.solvers.ipopt.slcp_bridge.solve_sia`; options MUST
  include ipopt_options tol=1e-9 and constr_viol_tol=1e-9 (subproblems at
  1e-12 falsely report infeasible), stationarity_tolerance=1e-5,
  condense_numerator=True, 200-iteration cap (past 200 = not converging; do
  not raise the cap).
- TASOPT 2.16 source (cross-reference): /Users/codykarcher/Desktop/Tasopt2.16
  (src/tfsize.f, tfweight.f, tfcool.f; decks runs/737/737.tas, runs/D8/d82.tas).
- Write your own thin runner. Beware: existing scratch runners export env vars
  (TECH, V_VT_FLOOR, V_HT_FLOOR, H_FIELD_FT) that OVERRIDE class values.

## Regression gates (non-negotiable, run after every change)
Conventional 737 (default build = cranked wing, TASOPT_737800 deck,
TECH=cfm56_era, sea-level basis):
  MTOW 166,407 lbf / fuel 47,162 lbf -- bit-identical when your work is
  off-path.
D8 (arch d8, D82_SPaircraft deck, TECH=d8_era, sea-level basis,
V_VT_FLOOR=0.001, span 44.20 m / field 4,960 ft / M 0.72, 8-abreast 2-aisle):
  MTOW 137,025 / fuel 26,908 -- bit-identical off-path.
Against TASOPT with the current engine: D8 bare engine 0.92, 737 ~1.03-1.07 of
TASOPT's Webare; D8 fuel 0.998, 737 fuel 0.993 of TASOPT. The new engine, at
deck-equivalent tech levels, must not degrade these.

## House rules -- defect classes met repeatedly; you WILL meet them
1. FREE LEVER: any quantity the optimizer can move for free gets exploited.
   Defined quantities are EQUALITIES (mark `# [SP] SigEq`), not one-sided
   bounds. A one-sided row is only safe when its pressure direction is argued
   in a comment.
2. DUPLICATE EQUALITIES = LICQ death: never write the same identity in two
   modules (AR == b^2/S existed 3x and pinned complementarity at 1.0 forever;
   the solve looks converged-but-isn't). One owner per identity. Diagnostic:
   SVD the equality Jacobian at a feasible point; it must be full rank.
3. DETECTOR-HOSTILE SHAPES: differences of variables inside products/powers
   inside ratios with posynomial denominators get MISTRANSLATED by the
   structure detector (measured: a pyomo row 43% slack while the solver's
   copy read tight). Write pure posynomials via intermediate variables
   (deta-style in wingbox_tasopt.py) and blend variables. Verify translation
   by loading the solution into pyomo and checking every constraint.
4. One-sided rows implement max(0, .) for free: a row whose requirement goes
   negative leaves its variable on the positivity floor (how tfcool's per-row
   cooling and the noise buzzsaw onset work). Use it for regime cutoffs.
5. NO zero monomials (0*x is not a posynomial); branch at the Python list
   level instead.
6. `a >= b if cond else c` parses as `(a >= b) if cond else c` -- parenthesize.
7. Initial guesses matter in log space: a variable ten decades off its
   magnitude is what non-convergence looks like.
8. Units: weights are N internally; scratch-runner JSONs divide unit-"N"
   values by 4.448222 (lbf). FS_V is in KNOTS. Check units on every constant.

## Current engine model: keep / replace
KEEP: Fitzgerald weight (a(BPR)*mdotc^b*(OPR/40)^c on SLS-corrected design
core flow, validated to 0.1% of TASOPT), tfcool cooling chain (rating-point,
one-sided), the K_epf fan-efficiency-lapse concept, nacelle/pylon fractions,
the noise module interface (needs u_6, u_8, A_5, A_7, m_fan exports).
REPLACE: constant-cp station chain, deck-pinned pi_f_D/pi_lc_D/pi_hc_D/BPR_D
(these become free under the new physics), ETAS constants (become tech-level
parameters), m_tild/pi_tild fits (become pycycle-map fits).

## First action
Build and solve the conventional 737 with a thin runner and confirm the
regression numbers above BEFORE touching anything.
