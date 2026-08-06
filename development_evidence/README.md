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
against Pyomo `main`.  The decision to spin off at this time has been 
taken to allow for this package to grow organically to meet upcoming
needs.  The recent flurry of updates have been primarily due to the 
development of a new solver, general clean up, and quality of life
improvements on a package that has been stable for almost three years.
Generative AI has helped to clear the backlog and is disclosed in the
disclosure statement in the paper.  

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
