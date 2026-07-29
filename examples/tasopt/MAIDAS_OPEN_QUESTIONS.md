# MAIDAS: the remaining gaps, with the actual code

25 of 82 scalar-signature TASOPT functions still fail. Below is every one,
bucketed by **verified** cause — I re-derived these by capturing the real
failing frame rather than inferring from the source, and two of my earlier
claims were wrong. Corrections first.

## Corrections to what I told you earlier

**"9 blocked by branching inside a callee"** — actually **7**, and 4 of the 9
I counted were something else entirely (a selector in the function's own body,
which `partial` already fixes today).

**"3 blocked by structured arguments"** — that bucket does not exist. I
inferred it from `size_landing_gear` taking a `LandingGear` dataclass, but that
function is excluded from the sweep by its signature filter and was never in
the 25. The real third bucket is error-guard paths (below), which I had
mis-filed as generic `TypeError`.

---

## Bucket 1 — error-guard branches (6 functions, cheapest fix)

`autobranch` generates a leaf for **every** path, including the ones whose body
is `raise`. Those leaves then fail to trace, because formatting a Proxy into
the error message blows up:

```python
# tasopt_py/engine_v3/hx_size.py:148  tube_geometry
    C = SAFETY_FACTOR * dp / (2.0 * YTS)
    if C >= 0.5:
        raise ValueError(
            f"design pressure difference {dp:.4g} Pa against a yield stress "   # <-- Proxy.__format__
            f"of {YTS:.4g} Pa gives C = {C:.4f}; ...")
```

→ `TypeError: unsupported format string passed to Proxy.__format__`

Same shape in `heat_exchanger.tube_thickness` and `fuelcell.cell_voltage_simple`.

The same thing blocks three *more* functions indirectly, because the guard sits
in a **callee**:

```python
# tasopt_py/engine_v3/heat_exchanger.py:147  _capacity_rates
    C_min, C_max = min(C_c, C_p), max(C_c, C_p)
    if C_max <= 0.0:
        raise ValueError("both heat capacity rates are zero or negative")
    return C_min, C_max, C_min / C_max, C_c <= C_p
```

That one helper is the sole reason `heat_transfer`, `NTU_from_effectiveness`
and `effectiveness_from_NTU` cannot be traced.

**These are not model branches.** A validation guard has no place in a
Piecewise — it describes inputs the model rejects, not regions the model
behaves differently in.

> **Question 1:** should `autobranch` prune leaves whose body unconditionally
> raises? I think obviously yes, but it is a semantic choice: it means the
> decomposition no longer covers the whole input domain, and the guards stop
> partitioning it. I would keep the guard string on the Piecewise as a
> recorded *domain restriction* rather than discard it silently.

Unblocks 6 of 25 for what looks like ~20 lines.

---

## Bucket 2 — real branching inside a callee (7 functions)

Here the branch is genuine physics, just not in the function you called.

```python
# tasopt_py/aero/blclosure.py:89  dilw   -- no branch of its own
def dilw(hk: float, rt: float) -> float:
    hs = hsl(hk, rt, 0.0)          # <-- hsl branches at hk = 4.35
    rcd = 1.10 * (1.0 - 1.0 / hk) ** 2 / hk
    return 2.0 * rcd / (hs * rt)
```

| entry | branching callee | branch |
|---|---|---|
| `aero.blclosure.dilw` | `hsl` | `if hk < 4.35` — laminar closure regions |
| `engine_v3.fuelcell.power_density` | `cell_voltage_simple` | `if T > T_VAP` — vapour vs liquid product |
| `engine_v3.fuelcell_1d.dlambda_dz` | `nafion_diffusion` | `if lam < 16.8` — cubic vs extrapolation |
| `cryo.thermal.freestream_heat_coeff` | `atmos` | the softplus ternary |
| `engine_v3.heat_exchanger.heat_transfer` | `_capacity_rates` | *guard — see bucket 1* |
| `engine_v3.hx_size.NTU_from_effectiveness` | `_capacity_rates` | *guard* |
| `engine_v3.hx_size.effectiveness_from_NTU` | `_capacity_rates` | *guard* |

So bucket 1 takes 3 of these; **4 need real callee decomposition.**

> **Question 2:** how should a branching callee be handled?
>
> **(a) Inline it.** AST-substitute the callee body at the call site before
> decomposing. Exact, keeps one flat Piecewise, and the guard strings stay
> readable in terms of the caller's variables. But it needs argument renaming,
> breaks on recursion, and the graph grows multiplicatively — a caller with
> two 2-way callees becomes 4 leaves, and `tfoper` calls things that branch a
> dozen times.
>
> **(b) Let a callee return a Piecewise and compose.** The caller's IR holds a
> nested Piecewise; no inlining, no growth in source size, recursion is fine.
> But the guards are expressed in the *callee's* locals (`hk < 4.35` where
> `hk` is `hsl`'s parameter), so the caller's Piecewise has guards it cannot
> evaluate. You would need to carry a binding from callee parameter to caller
> expression, and then the conditions are no longer plain strings.
>
> **(c) Do nothing** and require the user to pass a per-branch selector, the
> way `nfweb` is passed now.
>
> The tradeoff is *flat and exact but explosive* against *compositional but
> with guards that need substitution*. I lean (b) because the explosion in (a)
> is real for the engine, but it changes what a guard string is, and that is
> your call.

---

## Bucket 3 — deliberate refusals (8 functions)

Not gaps; documenting so you can disagree.

```python
# tasopt_py/structures/cabin.py:81  seats_abreast
    while cabin_width > Dmin or math.isclose(...):     # trip count is data
        n += 1
```

`while` refused: no static trip count, cannot unroll or split.
Also `find_K1_head`.

```python
# tasopt_py/cryo/stiffeners.py:73  stiffeners_bending_moment
    for i in range(1, n + 1):
        if ...:                       # guard differs per iteration
```

`if` inside `for` refused: no single guard string describes the path.
Also `which_third_octave`, `gas_tset_single`, `pralt`, `place_cabin_seats`,
`stiffeners_bending_moment_outer`.

Both are Newton/bisection loops or seat-counting loops — **iterative solvers,
not algebra**. A signomial program would replace them with a residual
constraint rather than trace them, so I do not think either is worth
supporting.

---

## Bucket 4 — genuinely opaque (1) and selector-solvable (3)

`engine.weight.tfweight` selects a **Gaussian-process surrogate** when
`iengwgt = 3`. Opaque by construction; nothing to detect.

`gasfun`, `gasfuel` and friends need a gas index bound. `partial` handles most
of them today — `gaschem(igas=40)`, `tank_heat_coeff(ifuel=40)` and
`arrange_seats(6)` all trace now. `gasfun` itself still fails after binding,
because it *also* does a table bracket search (a real loop), and `gasfuel`
assigns into a list, which the tracer cannot do.

> **Question 3 (minor):** should tracing support item assignment into a list
> that was created inside the traced function? `gasfuel` builds a
> stoichiometry vector element by element. It is a common pattern and the list
> is local, so it is sound — but it is more tracer surface area.

---

## Summary

| bucket | count | status |
|---|---|---|
| error-guard leaves | 6 | **Q1** — cheap, I recommend yes |
| real callee branching | 4 | **Q2** — the real design decision |
| loops (`while`, `if`-in-`for`) | 8 | refused deliberately |
| selector-solvable | 3 | 3 of 5 already work via `partial`; 2 need Q3 |
| opaque surrogate | 1 | nothing to do |
| **total** | **25** | (buckets overlap by 3: the guard-in-callee cases) |
