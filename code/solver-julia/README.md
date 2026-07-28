# Result reproduction

Scripts in this folder reproduce the figure 6, the profile plots in figure 7 and the
data shown in tables 1 and 2 and the pressure-density plot in figure 7.

Run all commands in this directory.

## Environment setup

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
Julia v1.12.4 was used to resolve the `Manifest.toml`.

## Commands for reproduction

### Section 4.1

The values in table 1 can be reproduced by running the following command
```bash
julia --project=. -e 'include("examples.jl"); reproduce_section41()'
```
The tables will be directly output to the command line.

### Section 4.2

The plots in figure 6 can be reproduced by running the following command
```bash
julia --project=. -e 'include("examples.jl"); reproduce_section42()'
```
There will be two files output `jeans_dealii_st.pdf` and `jeans_dealii_lt.pdf`.

### Section 4.3

The profile plots in figure 7 can be reproduced by running the following command
```bash
julia --project=. -e 'include("examples.jl"); reproduce_section43_profiles()'
```
For reproducing the pressure and density plots in figure 7, first clean the `out/`
directory, run the simulation afterwards and convert the output to vtk files:
```bash
rm -r ./out
julia --project=. -e 'include("examples.jl"); reproduce_section43_plotdata()'
julia --project=. -e 'trixi2vtk("out/solution*.h5", output_directory="section43")'
```
The plots were manually created in Paraview v6.1.1 from the vtk output, edited in
GIMP and there were added overlays in LaTeX to the plots.

### Section 4.4

For reproducing a single line of table 2 for *k* MPI ranks, execute the following
command three separate times for taking timings of numerical simulations on uniformly
refined meshes, where *k* is one of 1, 2, 4, 8, 16 or 32.
```bash
mpiexecjl --project=. -n k julia -e 'include("examples.jl"); rebenchmark_section44_fixed()'
```
Execute the following command for taking timings of numerical simulations on
 adaptively refined meshes, where *k* is one of 1, 2, 4 or 8.
```bash
mpiexecjl --project=. -n k julia -e 'include("examples.jl"); rebenchmark_section44_amr()'
```
The exact entries were manually extracted from the output of the simulation with the
minimum runtime of three separate runs.
