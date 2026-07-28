# deal.II solver package reproduction

This directory contains the source code for the multi-grid solver for the self-gravitational Poisson equation implemented in C++ using deal.II and its wrapping C library. 

The Julia package built from this source code can be found in [libdealii_trixi_paper2026_jll](https://github.com/trixi-framework/libdealii_trixi_paper2026_jll.jl), or built locally from the available sources by running the following command.
```bash
julia --project build_tarballs.jl --deploy=local
```
For reproducing the results in the accompanying paper this is not required. The `Manifest.toml` in `../solver-julia/` already uses the public repository mentioned above.
