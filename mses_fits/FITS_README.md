# MSES family CD fits (SMA / GP-compatible)

Fits of `CD = f(Re/1000, tau, Mach, CL)` per airfoil family (NC, NE, N3-T),
from the MSES dataset in `~/data/training_mses_families` (forced transition
3%/5%, Ncrit 9, Re 5e6-4e7, Mach 0.10-0.86). SMA = softmax-affine (GP-compatible
posynomial). **Re is in thousands** (Re/1000), tau = t/c, exponents documented.

Each fit ships as:
- `*.yaml`  — coefficients: `alpha`, `A` (K x n exponent matrix), `b`, var order, RMS.
- `*.posy.txt` — the ready-to-use posynomial inequality `CD**alpha >= sum_k exp(alpha*b_k) * prod_i x_i**(alpha*A_ki)`.

Model: `log(CD) = (1/alpha) * logsumexp_k( alpha*(b_k + A_k . log(x)) )`, i.e.
`CD^alpha = sum_k exp(alpha*b_k) * prod_i x_i^(alpha*A_ki)` (a posynomial; monomial
exponents on x_i are `alpha*A_ki`).

## Files & quality (RMS_log, median %, p95 %)

| File | model | RMS_log | med% | p95% | notes |
|---|---|---|---|---|---|
| `NC_SMA_final` | full range CL 0.1-1.2 | 0.184 | 1.7 | 45 | usable but high-CL/transonic tail is rough |
| `NE_SMA_final` | full range | 0.172 | ~2 | ~39 | " |
| `N3-T_SMA_final` | full range | 0.289 | ~2 | ~68 | weak (see CL=0.1 fit below) |
| `NC_SMA_CL0p7_final` | **CL<=0.7**, all Mach | **0.046** | 1.0 | **10** | clean; captures drag rise to CL 0.7 |
| `NE_SMA_CL0p7_final` | **CL<=0.7**, all Mach | **0.046** | 0.8 | **11** | clean |
| `N3-T_SMA_CL0p7_final` | CL<=0.7 | 0.278 | 1.9 | 72 | still poor (T family) |
| `N3-T_SMA_CL0p1_final` | **CL=0.1, NO CL var**  CD=f(Re/1000,tau,Mach) | **0.043** | 0.5 | **7.4** | the T-airfoil fit; each polar interp'd to CL=0.1 |
| `N3-T_SMA_CL0p1_boundedL12.json` | CL=0.1, exponents capped `|alpha*A|<=12` | 0.195 | ~10 | ~46 | solver-friendly but less accurate (see below) |

## Recommendations
- **NC / NE**: use the `*_CL0p7_final` fits (RMS ~0.046, p95 ~10%). They capture
  the Mach/drag-rise dependence cleanly up to CL 0.7. Above CL~0.7 the transonic
  buffet corner is non-convex (signomial) and breaks the posynomial.
- **N3-T (T airfoils)**: use `N3-T_SMA_CL0p1_final` (CD at CL=0.1, no CL
  dependence) — clean (RMS 0.043). The CL-dependent T fits are poor at all CL.
- **Full-range fits**: included as-is; median is good but the high-CL transonic
  tail is large.

## Exponent bounds
The CL=0.1 T fit's steep Mach drag-rise needs large exponents (max ~540). Capping
`|exponent| <= L` degrades RMS: L=60->~0.05, L=20->0.124, L=12->0.195, L=4->0.30.
`*_boundedL12.json` is the L=12 version if your GP solver dislikes large exponents.
