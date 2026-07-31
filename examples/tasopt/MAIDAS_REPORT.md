# MAIDAS on TASOPT

Running MAIDAS structure detection over the TASOPT Python port, using
[SPaircraft](../convexengineering/spaircraft/) — a hand-written signomial
program for roughly the same aircraft — as a check wherever the two model the
same physics.

**The goal is MAIDAS capability.** TASOPT is the test case: ~4000 lines of
straight-line engineering Python that nobody wrote with a solver in mind, so
every place MAIDAS trips is a real gap rather than an artefact of the input.
The structure findings are a by-product, and a nice one.

Reproduce with `python maidas_report.py [target]`.

---

## Why TASOPT is a good test case

The pycycle case (`../pycycle/maidas_case.py`) had to be *transcribed* by
hand: OpenMDAO's `compute(self, inputs, outputs)` reads and writes dict-likes,
so the arithmetic was retyped into plain functions. That transcription is a
place for errors to enter and for the answer to be quietly steered.

The TASOPT port needs none of it. It is already plain scalar Python, verified
against the compiled Fortran to machine precision, so MAIDAS traces the same
code that reproduces the reference aircraft. Nothing is retyped and nothing is
chosen by hand except which discrete selectors to hold fixed.

---

## Capability added

Eight fixes, in the order the work surfaced them. All have regression tests;
the MAIDAS suite went 729 → 753.

### 1. Keyword-only parameters crashed the tracer — sometimes fatally

`analyze` wrapped its target with `functools.wraps` to strip pint units. That
sets `__wrapped__`, so `inspect.signature` reported the *wrapped* function's
parameters while the wrapper's own code object still said `(*args, **kwargs)`:

```
signature says:    (*, a, b, c)
code object says:  co_argcount=0 co_kwonlyargcount=0 co_varnames=('args','kwargs')
```

torch.fx sizes its placeholder list from the signature and then rebuilds the
code object, so the two disagreed. Two or more keyword-only parameters raised
`ValueError: code: co_varnames is too small`, and tracing several such
functions in one process **segfaulted the interpreter**.

torch.fx handles keyword-only functions fine on its own — the wrapper
introduced this. Fixed by generating a wrapper whose real parameter list
matches, defaults transplanted via `__defaults__`/`__kwdefaults__`, and
deliberately no `__wrapped__`.

### 2. `functools.partial` was not accepted

`analyze(partial(gasfun, 40))` died on `AttributeError: 'functools.partial'
object has no attribute '__globals__'`.

This matters more than it sounds. Engineering code is full of **selectors** —
`igas`, `ifuel`, `nfweb`, `iwplan` — integers that choose a configuration
rather than vary continuously. Traced symbolically they become Proxies and
then die inside a dict lookup or an `int()`, far from anywhere the user can
act. MAIDAS has no `static_argnums` equivalent, so `partial` is the natural
escape hatch and it did not work.

One subtlety: `inspect.signature(partial(f, b=2))` reports `(a, *, b=2, c)` —
it *keeps* keyword-bound parameters visible so callers can override them.
Forwarding `b` handed it straight back to the tracer as a symbol, defeating
the point. Bound keywords are now dropped from the generated signature.

### 3. `autobranch` silently dropped the implicit `else` — the worst bug found

An `if` with no `else` still has two paths. Only the taken one was emitted.

Alone that made `autobranch` bail out (one leaf, so it returned the function
unchanged), which failed loudly later. Combined with any *other* branch it was
far worse:

```python
def mixed(x, y):
    if x < 1.0:  a = 2.0*y
    else:        a = 3.0*y
    if y > 5.0:  a = a * 10.0
    return a
```

Four real cases. `autobranch` emitted **two**, both requiring `y > 5.0`. The
entire `y <= 5.0` half of the domain vanished with no error, and the Piecewise
built from it would have been *wrong* there rather than absent — the failure
mode you least want from a structure detector.

Fixed by yielding the fall-through with an empty body. The regression test
checks that the guards **partition** the domain and that each leaf reproduces
the original function, not merely that the branch count is right.

### 4. `max` / `min` could not be traced at all

Builtin `max` compares its arguments, and comparing two Proxies forces a
`bool()` the tracer cannot answer. But the IR already carried `MaxAffine`,
`MaxMonomial`, `MinAffine`, `MinMonomial`, and `rewrites.min_posy_epigraph`
already *expected* an `Apply("min", ...)`. Only the tracer could not make one.

There are **114 `max`/`min` call sites across 35 modules** of the port, and
they are concentrated where the physics is: `tfoper` (15), `blclosure` (5),
`thermal` (5), `takeoff` (5). In TASOPT they are explicit modelling
statements, e.g. `surfw`:

```python
# A heavy outboard engine could otherwise drive So/Mo below the
# strut-attach values, implying a negatively-tapered structure.
So = max(So, Ss)
```

Now a shim shadows `max`/`min` in the traced globals and emits graph nodes;
lift's existing fallback turns them into `Apply("max", args)`.

### 5. `max_posy_epigraph`

`t >= p_i` for each piece, replacing the expression with `t`.

The asymmetry with the existing `min` rule is the point. `t >= p` rearranges
to `p/t <= 1` — a posynomial over a monomial, still a posynomial, still GP.
`t <= p` puts a posynomial in the **denominator** and leaves the cone.

> A maximum is the tight side of a convex function; a minimum is not.

So `max` of posynomials costs one variable and *n* constraints and stays in
GP. `min` does not, and no auxiliary variable recovers it — the existing rule
is flagged rather than blessed.

A new `gp_representable` label distinguishes this from `posynomial`: a `max`
of posynomials is not a posynomial, it is *reachable* from one. That is
exactly the distinction that tells you whether a model needs a new variable.

### 6. Branch leaves lost every intermediate

`analyze` traced branch bodies with `tapped=False`, so any function with an
`if` reported one opaque result and everything inside the branch was invisible.

Two causes. The leaves are synthesised by `exec`, so `inspect.getsource` could
not read them back and the tap rewriter had nothing to rewrite; they are now
registered with `linecache` under a unique filename. And `analyze` now traces
each leaf tapped, attaching the results as `.branch_taps` — a tuple of IRDicts
parallel to `.branches`. The Piecewise value is unchanged, so callers that
only want the result are unaffected.

`surfcd` went from 0 intermediates to 29 per branch.

### 7. `autobranch` rejected every loop

Any `for` or `while` anywhere in a function raised `NotImplementedError`, even
when the loop had nothing to do with branching.

A `for` with a concrete iterable is simply unrolled by the tracer — that is
the common case (`for k in range(nk)` with `nk` bound as a selector), so
refusing it rejected functions that trace perfectly well. A `for` is now left
alone unless it *contains* an `if`, which cannot be split soundly because the
guard may differ between iterations.

`while` is still refused, and for a real reason: its trip count is re-evaluated
against live values, so it can neither be unrolled nor split.

This is what unblocked the fuselage.

### 8. Selectors and `autobranch` did not compose

A partial does not forward attributes, so `partial(autobranch(fn), nfweb=1)`
lost `.branches` and was traced as a single path. Leaves are now rebound the
same way and carried across, and the materialisation was hoisted to the top of
`analyze` so it happens *before* the branch check.

Also needed for this to work at all: a materialised partial's globals are its
own tiny `exec` namespace, so the real module was never patched and the
underlying function was never collected for rewriting; and a partial captures
its function **by reference**, so the tap-rewritten copy was bypassed and every
intermediate vanished. The wrapper now resolves its target through module
globals at call time.

### 9. Ternaries were invisible to `autobranch`

`A if C else B` is the same branch as an `if`, written differently, but
`_decompose` only walked `ast.If` *statements*. A function whose only
branching is a ternary reported "no branches found" and came back
untraceable — which covered `atmos`, most of the boundary-layer closure, and
anything else written in the compact style.

Each conditional expression is now hoisted into a fresh temporary assigned by
a real if/else placed before the statement that used it, innermost-first so
nested ternaries work and an outer guard may itself contain one. Loop bodies
are recursed into first, so a ternary inside a loop is hoisted within that
loop rather than lifted out of it.

### 11. Raising branches are guards, not pieces

`autobranch` generated a leaf for **every** path, including bodies that only
`raise`. Those leaves then failed to trace, usually on
`TypeError: unsupported format string passed to Proxy.__format__` — formatting
a traced value into the error message.

A validation guard is not a model branch. It states which inputs the model
rejects, not a region where the model behaves differently. Raising leaves are
now pruned, and the guard comes back as `.domain_constraints` — the model
requires its *negation*, so an emitter can impose it as a real constraint
rather than losing it with the leaf.

```
tube_geometry:  branches=1  domain_constraints=['(C >= 0.5)']
```

### 12. Branching inside a callee

`dilw` has no `if` of its own — it calls `hsl`, which branches at
`hk = 4.35`. Decomposing only the entry function found nothing, and the trace
then died on the callee.

Branching callees are now **inlined**, so the branch is split where it is
visible. Non-branching callees are left as calls (the graph should not grow
for nothing), recursion is skipped, and depth is capped.

The objection I had raised against composing instead — guards written in the
callee's locals — is solved by inlining: a parameter bound to a bare name or
literal is substituted outright, so `dilw` reports `(hk < 4.35)` in the
caller's own variable rather than `(_mi0_hk < 4.35)`.

**Two bugs here were invisible to "does it trace".** Converting `return` to
an assignment loses its short-circuit, so `hsl`'s fall-through overwrote the
taken branch and every input below the threshold silently got the *wrong
formula*. And substituting a parameter clobbered the caller's variable when
the callee reassigned it. Both were caught only by evaluating every leaf
against the original function over sampled inputs — which is now the
regression test, and should be the standing bar for this feature.

### 13. Recognizable opaque types

Some callables are deliberately not algebra: a Gaussian-process surrogate, a
tabulated map, an external solver. MAIDAS previously died on whatever
exception the callee happened to raise, losing the one fact that mattered —
*what kind of thing* was in the way.

`maidas/core/atoms.py` adds an open registry:

| kind | what it implies |
|---|---|
| `data_fit` | fit an SMA/ISMA surrogate and substitute — becomes GP-compatible |
| `table` | same, but sample points already exist |
| `external_solver` | express as `ImplicitSolve` over its residual |
| `black_box` | declared unanalysable, no plan |

`@data_fit("engine_weight_gpr")` marks a callable: it still computes normally
on concrete numbers, and emits a single named node when traced. Classification
gives `not_gp` **and** `atom:data_fit` — honest that it is not GP as it
stands, but named rather than fatal, so a later pass knows what to do.

Marking something opaque is a statement about this analysis, not about the
mathematics: a GPR posterior mean is a weighted sum of kernel evaluations and
may itself be GP-representable. `__wrapped_atom__` keeps the original so a
later pass can descend after all.

### 14. The "I got stuck here" report

`maidas/core/stuck.py`. `explain(fn)` returns a `StuckReport`; `format_stuck`
renders it:

```
MAIDAS could not analyse gasfun().

  what stopped it : a discrete index or key used to look something up
  category        : selector
  where           : tasopt_py/gas/properties.py:124
  underlying      : ValueError: GASFUN: undefined gas index: Proxy(...)

       123 |     try:
  >>   124 |         key = IGAS[igas]
       125 |     except KeyError:

  what usually fixes it:
      bind it with functools.partial(fn, <name>=<value>) so it stays
      concrete; it chooses a configuration rather than varying
```

Eight categories, each with a reason and an action. The report is falsey when
analysis succeeded, so `if explain(fn):` reads as "if it got stuck".

One thing worth knowing: **the traceback's line number cannot be trusted.**
The tapped trace runs an AST-rewritten copy, so the frame's `lineno` indexes
regenerated source and, read against the real file, usually lands in a
docstring. The frame supplies only the file and function name; the offending
line is then located by scanning the real source for the construct the
category implies.

### 10. `sqrt` classified as opaque

`(a*b)**0.5` lifted to a monomial. `math.sqrt(a*b)` lifted to
`Apply("sqrt", ...)` and classified `not_gp`. The same expression got two
different answers depending on how it was spelled.

There are **205 `math.sqrt` call sites** in the port, so this was a false
negative across the whole model. `sqrt` and `cbrt` are now lifted as the fixed
powers they are.

### 15. Suggesting the known tricks

`TAYLOR_DESCRIPTIONS` already carried a GP-usable series and a validity range
for every transcendental MAIDAS recognises, but nothing offered them at the
point of failure. A user told only "contains log()" has to already know the
trick; `optimization_check` now shows it:

```
  GP-compatible approximations available:
    - log(x)     ~  (x-1) - (x-1)^2/2 + (x-1)^3/3
                    valid: < 1 % error for 0.8 <= x <= 1.2
```

**The validity range is the part that matters.** For `cfturb` the argument is
`0.06 Re` with Re around 1e7 — nowhere near 0.8 to 1.2 — so the suggestion is
visibly useless there, which is exactly what a user needs to see. A
suggestion without a range would have been worse than none.

`optimization_check` also now reports what any declared atom (section 13) implies, so
naming a kind and then saying nothing about it no longer happens.

### Which obstructions actually have a trick

Counting every not_gp intermediate across all targets and branch leaves:

| obstruction carries | count |
|---|---|
| `cos` | 8 |
| `exp` | 4 |
| `sin` | 2 |
| `asin` | 2 |
| `tan` | 1 |
| **no transcendental — purely algebraic** | **70** |

So the series tricks address a *minority* of the problem. And it is smaller
still than it looks, because of where the transcendentals actually are:

| quantity | function | what it is |
|---|---|---|
| `cosL`, `sinL`, `tanL` | `cos`, `sin`, `tan` | **wing sweep angle** |
| `thetafb` | `asin` | **fuselage bubble angle** |
| `p`, `softplus` | `exp` | atmospheric pressure lapse |

Sweep and bubble angle are fixed geometry in every signomial formulation of
this aircraft — held constant, not optimised — so `cos(Lambda)` is a number
and no expansion is needed. **The only transcendental obstruction that is
genuinely a function of a design variable is the atmosphere's exponential.**

Which leaves the real work where SPaircraft put it: the 70 algebraic
obstructions — posynomials in denominators, roots of signomials, fractional
powers — none of which a series expansion touches.

### 16. Declaring atoms for code you do not own

`atom(...)` is a decorator, which assumes you can edit the source. Often you
cannot: the code is third-party, or a verified port whose text must not change
to suit the analysis. `declared_atoms({...})` installs the wrappers for the
duration of a `with` block.

Two things it had to get right:

**Patch every importer, not the defining module.** `from x import f` binds `f`
into the *caller's* namespace, and that is the binding the tracer resolves —
so patching only `x` silently misses every direct importer, which is most of
them. It now patches every module in `sys.modules` holding a reference to the
same object.

**Atoms need an output arity.** An atom emits one graph node, so a function
returning a 6-tuple hands back a single value the caller then tries to unpack,
and the trace dies on "Proxy object cannot be iterated" far from the cause.
`outputs=6` indexes the node into six proxies and unpacking works as written.
Getting this wrong is not silent — it fails as `too many values to unpack` —
but it must be declared, which argues for MAIDAS inferring it later.

### 17. The loop refusal was too strict

`autobranch` refused any `for` containing an `if`, on the grounds that the
guard can differ between iterations. True, but the conclusion was wrong: a
guard on the **loop variable** (`if ipass == 1:`) is concrete once the tracer
unrolls, and needs no splitting at all. Refusing turned functions that trace
perfectly well into hard failures.

Loops are now left intact and never split into. A genuinely data-dependent
guard still fails — but in the tracer, where the failure is honest and the
stuck report can explain it, rather than preemptively.

---

## The engine cycle

`tfsize` is the hardest target and the one with no SPaircraft counterpart —
where York had to fit hardest. It is **partly unblocked, not yet through.**

What it needed, in order:

1. **The gas property tables as atoms.** `gassum` / `gas_prat` / `gasfun`
   bottom out in a bracket search over a tabulated temperature grid. That is
   not algebra and never will be; declaring them `table` atoms is both what
   makes the trace possible and the honest modelling statement — TASOPT's gas
   properties *are* tabulated data, and a signomial formulation fits them.
2. **Correct output arities** — 6 for the `gas_*` state functions, 2 for
   `gassumd` and `gas_burn`, 1 for the dataclass-returning ones. I guessed two
   of these wrong and the failure was immediate and clear.
3. **List arguments bound** — `epsrow` and `Tmrow` are per-blade-row vectors,
   traced as single Proxies and then iterated.
4. **The loop refusal relaxed** (section 17) — the 60-pass offtake
   convergence loop branches on its own counter.

After all four the `TraceError` moves but does not disappear. The stuck report
points at line 142, but **that is the first matching construct, not
necessarily the culprit** — the locator finds a plausible line of the right
kind, which is honest for a heuristic and should not be read as a diagnosis.
Finding the real one needs another pass.

The structural question underneath is the one already flagged: `tfsize`
carries its own relaxation loop, and an iterative solver is not something to
trace. The right treatment is `ImplicitSolve` over its residual — the
machinery exists, the *recognition* does not.

### 18. Opening a convergence loop -- and what York settles

The engine cycle forced the question I had put to the user: how should a
relaxation loop be treated? The answer came from reading York, Hoburg and
Drela's turbofan SP rather than from taste.

York does **not** fit an engine deck, and does **not** use an implicit solve.
He writes the full 1-D cycle as a signomial program with, in his words, "no
on-design/off-design distinction: the same engine geometry is shared by every
flight segment, while the cycle state is per-segment". Nothing in his model
iterates.

So TASOPT's loop has no counterpart in the SP at all. TASOPT iterates because
it is a *simulation* -- given a design, converge the cycle. York writes the
converged state as constraints and lets the optimizer satisfy them
simultaneously. **The loop is the constraint set.**

`maidas/core/implicit.py` implements that reading. `open_loops(fn)` keeps one
pass of the loop body and returns the convergence test as a residual
constraint. On `tfsize`:

```
residual: 1.0 - mcold / mcore == 0
```

read straight out of `if abs(dmfrac) < TOLER: break`, with the bare name
resolved back through its assignment. Nothing is guessed. What is *not*
inferred is which variable the residual closes on -- in `1 - mcold/mcore`
both names appear and only the modeller knows -- so an `ImplicitSolve` needs
that one fact supplied; the residual constraint does not.

The iteration index is a choice, and a stated one: pass 1 of a relaxation
loop is initialisation, not the general equations, so the opener pins the
first *general* pass rather than silently keeping the initialisation branch.

### 19. Guard folding, and a cap

Opening `tfsize`'s loop exposed every guard at top level and `autobranch`
produced **36,864 leaves**. That is a bug-level outcome, not a decomposition.

Two fixes. `autobranch(fn, bindings={...})` folds any guard decidable from
bound arguments: `iBLIc == 0` with `iBLIc` supplied is a configuration
already chosen, not a branch of the model, and splitting on it doubles the
leaf count for nothing. And a `MAX_LEAVES` cap now refuses past 256, naming
the guards it found so the user can see which to bind.

Folding took `tfsize` from 36,864 to 8,192. Still capped -- which is the
finding.

---

## Where the engine cycle actually stands

**Not through, and the remaining obstacle is a strategy problem rather than a
tuning one.**

Everything upstream now works: the gas tables are `table` atoms, the
convergence loop opens into `1 - mcold/mcore == 0`, the selectors bind, the
decidable guards fold. What is left is roughly a dozen *genuine* guards --
nozzle choking at stations 6 and 8, `Trat < 1`, `fc >= 0.99` -- and
`autobranch` enumerates their cross product, which is 2^13 paths that mostly
do not exist.

Global path enumeration is the wrong representation for this. Nozzle choking
is not a path through the function, it is `M6 = min(M6_unchoked, 1)` -- a
**local** piecewise at the point it occurs. MAIDAS now has the IR for exactly
that (`MaxAffine`, `MinAffine`, `max_posy_epigraph`) and the tracer can reach
it, but nothing lifts an `if/else` that assigns one variable into a local
`min`/`max` node instead of splitting the whole function.

That is the next capability, and it is the one that would make `tfsize`
tractable: keep the graph flat, put the branch where the branch is. It is also
what York's formulation does -- his cycle has no paths, only constraints.

**The falsifiable check is ready and waiting.** York names exactly three
places his cycle leaves GP, all of the form "a sum on the greater side":
`fp1 == f + 1`, `alpha_p1 == alpha + 1`, and `F <= F_6 + F_8`. If MAIDAS gets
through `tfsize` and the signomial obstructions do not concentrate there,
either the port or the detection is wrong. That is a prediction from an
independent published derivation, which is a much stronger test than any
self-consistency check used so far.

---

## Coverage

Scalar-signature functions in the port that MAIDAS can analyse:

| | at session start | now |
|---|---|---|
| direct trace | 34 | 39 |
| via `autobranch` | 0 | 28 |
| **total of 82** | **34** | **67** |

The 25 that remain fall into four groups, in rough order of how much they
would unlock:

**Branching inside a callee (9).** `autobranch` decomposes only the entry
function. `dilw` has no branch of its own — it calls `hsl`, which does — so
the decomposition finds nothing and the trace dies on the callee's `if`.
Fixing it means either inlining branching callees or letting a callee return a
Piecewise that the caller composes with, and both are design decisions rather
than bug fixes.

**`if` inside a `for`, and `while` loops (8).** Refused deliberately: a guard
inside a loop can differ between iterations, so no single guard string
describes the path, and a `while` has no static trip count.

**Structured arguments (3).** `size_landing_gear` takes a `LandingGear`
dataclass whose `.model` field is a selector but whose other fields are design
variables. MAIDAS has no way to say "trace this object's numeric fields as
symbols and that one as a constant" — `partial` binds whole arguments, not
fields.

**Assorted (5).** A Gaussian-process engine-weight surrogate that is opaque by
construction, and a handful of `int()`/indexing calls on traced values.

---

## Structure findings

### Wing — `surfw` vs `wingbox.py` + `wing.py`

72 distinct intermediates: 26 monomial, 15 posynomial, 17 signomial, 14 not_gp.

| TASOPT quantity | MAIDAS | what SPaircraft did |
|---|---|---|
| `tbwebs`, `tbwebo` shear web thickness | **monomial** | monomial constraint — *no reformulation* |
| `Wscen`, `Wsinn`, `Wsout` weights | **posynomial** | `W_struct >= W_web + W_cap` — *no reformulation* |
| `hrmso`, `hrmss` rms box height | **not_gp** | **fixes `r_h = 0.75`**, bakes in the 0.92 factors |
| `tbcaps`, `tbcapo` spar cap thickness | **not_gp** | **changes variables** — constrains `I_cap`, never cap thickness |
| `So`, `Mo`, `Ss`, `Ms` shear/moment | **signomial** | **simplifies** to `M_r >= Lmax·AR·p/24` |
| `GJs`, `GJo` torsional stiffness | **not_gp** | **dropped entirely** |

Every not-GP intermediate lands where SPaircraft did something specific, and
MAIDAS says which *kind* of something: fix a parameter, change variables,
simplify, or drop.

`hrms` is the cleanest. SPaircraft's comment reads *"Assumes r_h = 0.75, so
the rms box height is ~0.92 t_max"* — and MAIDAS independently shows why:
`sqrt` of a signomial in `r_h`, removable only by making `r_h` constant.

**`GJ` is the find worth acting on.** That is not a reformulation — SPaircraft
has no torsional stiffness constraint anywhere. A GP wing box sized without
torsion can be wrong in a way no amount of solver accuracy fixes, and it is
invisible unless you diff against the model the SP was derived from. This is
the case for doing the comparison at all.

### Fuselage — `fusew` vs `fuselage.py` (83 constraints)

109 distinct intermediates: 3 constant, 37 monomial, 46 posynomial, 17
signomial, **6 not_gp**.

The most GP-friendly module in the aircraft — 78% monomial or posynomial
before any reformulation. That is consistent with SPaircraft's fuselage being
its largest and most faithful section: there was little to fight.


### Everything measured so far

15 routines, 291 classified intermediates. "GP-clean" means monomial,
posynomial or constant *as written*, before any reformulation.

| target | n | mono | posy | signomial | not_gp | SPaircraft counterpart |
|---|---|---|---|---|---|---|
| `fusew` fuselage | 109 | 37 | 46 | 17 | 6 | `fuselage.py` (83 cons) |
| `surfw` wing | 72 | 26 | 15 | 17 | 14 | `wingbox.py` + `wing.py` |
| `surfcd` surface drag | 29 | 18 | 0 | 3 | 7 | `model.py` drag build-up |
| `surfcm` pitching moment | 13 | 4 | 0 | 6 | 3 | `model.py` |
| `atmos` | 10 | 4 | 1 | 1 | 4 | `flight_state.py` |
| `wingsc` wing scaling | 9 | 4 | 0 | 3 | 1 | `wing.py` |
| `wingcl` lift | 8 | 3 | 0 | 3 | 2 | `model.py` |
| `chord_integrals` | 6 | 3 | 0 | 3 | 0 | `wing.py` |
| `wingpo` load case | 6 | 3 | 0 | 3 | 0 | `wing.py` |
| `planform_integrals` | 6 | 4 | 0 | 2 | 0 | `wing.py` |
| `dil` BL dissipation | 6 | 3 | 1 | 1 | 1 | none — York fits |
| `hsl` BL shape | 6 | 4 | 0 | 1 | 1 | none |
| `cfl` BL friction | 6 | 4 | 0 | 1 | 1 | none |
| `tailpo` tail load | 4 | 1 | 0 | 0 | 3 | horizontal/vertical tail |
| `cfturb` turbulent Cf | 1 | 0 | 0 | 0 | 1 | `model.py` |
| **total** | **291** | **118** | **63** | **61** | **44** | |

**64% is GP-clean before anyone reformulates anything.** That is the York
result reproduced from the arithmetic rather than asserted.

Three things stand out.

**`cfturb` is 1-for-1 not_gp**, and it is a one-line function:
`0.523 / log(0.06 Re)^2`. A logarithm of a design variable cannot be a
monomial in any spelling, which is exactly why a signomial formulation has to
*fit* skin friction rather than derive it. The smallest routine measured is
the one that most clearly needs a surrogate.

**The boundary-layer closure is friendlier than expected** — `dil`, `hsl` and
`cfl` are 3-4 monomial out of 6 each. The received wisdom is that BL closure
is hopeless for GP; measured, most of it is monomial and the damage is
concentrated in one or two terms per function.

**The atmosphere is 40% not_gp.** `atmos` is upstream of everything, so
whatever a signomial formulation does about it propagates into every flight
condition.

### Others

| target | distinct | mono | posy | signomial | not_gp |
|---|---|---|---|---|---|
| `tailpo` tail load | 4 | 1 | – | – | 3 |
| `wingsc` wing scaling | 9 | 4 | – | 3 | 1 (+1 gp_representable) |
| `chord_integrals` | 6 | 3 | – | 3 | – |
| `surfcd` surface drag | 29 | 18 | – | 3 | 7 |

`tailpo` shows the `1/(1 + lambda)` taper term as not_gp — a posynomial in a
denominator, the single most common way a clean aerodynamic expression leaves
GP.

---

## Traps worth knowing

**Classify constraints, not solved expressions.** An early pass classified
SPaircraft's *own* fitted `nu` correlation as not-GP, which is nonsense for a
shipped GP model. GP-ness is a property of a **constraint** — which side,
which direction. Writing `nu^3.94 >= posy` as `nu = posy^(1/3.94)` turns a
valid GP constraint into a fractional power of a posynomial.

**A verdict is only as clean as its worst branch.** When merging across branch
leaves the driver takes the *loosest* label, not the first or the most common.

**`classify_flat` maps over containers.** A routine returning a tuple gives
back a tuple of label-lists, not a flat list. Flatten before using.
