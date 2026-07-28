# Reproducibility subdirectories

The subdirectory `solver-cpp/` contains the C++ code of the gravity solver and the
script used to build and package the solver, see the `README.md` therein to find
out, how to rebuild and repackage the solver we developed. This is not needed for
reproducing the results in the paper, since we provide precompiled binary packages
that are automatically installed through the Julia package manager.

The subdirectory `solver-julia/` contains the Julia code and scripts for running
the simulations from which the tables and figures in the paper were created. For
some of the figures and tables there is a lot of manual work involved for reproducing
them. See the `README.md` therein to find out, how to reproduce some of the figures
and the data presented in the tables in the paper.
