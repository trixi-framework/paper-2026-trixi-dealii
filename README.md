# Multi-Solver Coupling for Parallel Adaptive Multi-Physics Simulations with Trixi.jl and deal.II

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](LICENSE)

<!--[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21620203.svg)](https://zenodo.org/doi/10.5281/zenodo.21620203)-->

Reproducibility repository for the paper

> *Multi-Solver Coupling for Parallel Adaptive Multi-Physics Simulations with
Trixi.jl and deal.II*

If you find these results useful, please cite the article

```bibtex
@online{ehlert2026multisolver,
  title={{Multi}-{Solver} {Coupling} for {Parallel} {Adaptive} {Multi}-{Physics} {Simulations} with {Trixi.jl} and {deal.II}},
  author={Ehlert, Vivienne and Gassner, Gregor and Kronbichler, Martin and Ranocha, Hendrik and Schlottke-Lakemper, Michael},
  year={2026},
  month={8},
  eprint={TODO},
  eprinttype={arxiv},
  eprintclass={math.NA}
}
```

If you use the implementation provided here, please also cite this repository as

```bibtex
@misc{ehlert2026multisolverRepro,
  title={Reproducibility repository for
         "{Multi}-{Solver} {Coupling} for {Parallel} {Adaptive} {Multi}-{Physics} {Simulations} with {Trixi.jl} and {deal.II}"},
  author={Ehlert, Vivienne and Gassner, Gregor and Kronbichler, Martin and Ranocha, Hendrik and Schlottke-Lakemper, Michael},
  year={2026},
  howpublished={\url{https://github.com/trixi-framework/paper-2026-trixi-dealii}},
}
```
<!--  doi={10.5281/zenodo.21620203}-->


## Abstract

Many standalone frameworks for numerical solvers have been developed to tackle the simulation of specific single- or multi-physics problems. For solving coupled problems, common approaches are to extend existing solvers, to develop an entirely new solver or to couple two existing solvers by using the interface provided by a coupling library or framework. However, to the best of our knowledge, there is not yet a framework that allows the easy and direct development of coupled solvers. In this work, we prototype a portable reproducible cross-language framework for the development of coupled parallel adaptive solvers using Trixi.jl and deal.II for the numerical simulation of coupled multi-physics problems. Currently, this is tightly entangled with an example coupled solver. We show its usability by developing a partitioned strongly-coupled multi-physics solver for the dynamics of Newtonian self-gravitational gases. For the coupled solver, we validate the expected order of convergence, physical sensibility of the results and mesh adaptivity. Finally, we investigate its parallel scaling. A publicly accessible reproducibility repository for the numerical results and code is available.

## Numerical experiments

This repository contains Julia and C++ code to reproduce paper results. The code and associated environments can be found in the `code/` subdirectory, see the `README.md` therein for how to reproduce figures and data in the paper.

## Requirements

- Julia v1.12.4 was used to generate the figures for the paper and the committed `Manifest.toml` files.

## Authors

 + Vivienne Ehlert
 + Gregor J. Gassner
 + Martin Kronbichler
 + Hendrik Ranocha
 + Michael Schlottke-Lakemper

## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!

