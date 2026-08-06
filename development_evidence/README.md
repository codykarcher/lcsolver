# Evidence of Continuous Development

Although LCsolver has received a recent burst of updates, the package is the
product of several years of sustained development. This folder collects the
supporting evidence: the commit history from its previous life inside Pyomo,
the peer-reviewed publications its algorithms come from, and two master's
theses built on the software while actively shaping its direction.

## History as the Engineering Design Interface (EDI) for Pyomo

Much of the core of this package was developed beginning in 2023 as a
contributed sub-package of [Pyomo](https://github.com/Pyomo/pyomo), the
*Engineering Design Interface* (EDI), over a period of more than three
months, and has since been used successfully on a variety of projects. The
complete commit log of the original work is visible in the
[`edi_dev` branch comparison](https://github.com/Pyomo/pyomo/compare/main...codykarcher:pyomo:edi_dev)
against Pyomo `main`.

## Peer-Reviewed Foundations

The algorithms at the center of the package were developed and published
across three peer-reviewed papers:

- Karcher, C. J., "[Logspace Sequential Quadratic Programming for Design
  Optimization](https://doi.org/10.2514/1.J060950)," *AIAA Journal*, Vol. 60,
  2022. ([arXiv:2105.14441](https://arxiv.org/abs/2105.14441))
- Karcher, C. J., "[A Method of Sequential Log-Convex Programming for
  Engineering Design](https://doi.org/10.1007/s11081-022-09750-3),"
  *Optimization and Engineering*, 2022.
  ([arXiv:2201.08436](https://arxiv.org/abs/2201.08436))
- Karcher, C. J., "[Data Fitting with Signomial Programming Compatible
  Difference of Convex Functions](https://doi.org/10.1007/s11081-022-09717-4),"
  *Optimization and Engineering*, 2022.

The SLCP and SIA machinery in `lcsolver/solvers/sequential/` implements and
extends the first two; the companion fitting methods are distributed as the
[lcfit](https://github.com/codykarcher/lcfit) package.

## Recent Master's Theses at CSULB

Two recent master's theses in the Department of Mechanical and Aerospace
Engineering at California State University, Long Beach used the precursor to
this package (EDI) as the foundation of their work, and in doing so actively
shaped the direction of the software. Both are included in this folder:

- Avila, D. B., "[Aircraft Design Optimization Using Signomial
  Programming](Thesis_DavidAvila.pdf)," M.S. Thesis, California State
  University, Long Beach, December 2025. *Develops a signomial-programming
  framework for the conceptual design of commercial transport aircraft,
  minimizing takeoff weight across integrated aerodynamic, structural,
  weight, and mission-performance models.*
- Shoda, M. K., "[Signomial Programming for Lifting-Line Analysis and
  Optimization](Thesis_MichaelShoda.pdf)," M.S. Thesis, California State
  University, Long Beach, May 2026. *Embeds lifting-line aerodynamic
  analysis directly in signomial-programming form, so wing analysis and
  design optimization share a single formulation.*
