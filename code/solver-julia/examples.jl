include("euler_gravity.jl")
include("examples/elixir_convergence.jl")
include("examples/elixir_jeansinstability.jl")
include("examples/elixir_sedovblastwave_2d.jl")
include("examples/elixir_sedovblastwave_3d.jl")
include("examples/elixir_smallsedov_3d.jl")

using Plots
using MPI

function reproduce_section41(; dealii_initialised = false)
   if !dealii_initialised
      dealii_ret = dealii_init_libs(ARGS)
   end
   errors_l2_p3 = main_convergence(; dealii_initialised=true,
                                   use_dealii_solver=true,
                                   polydeg=3)
   errors_l2_p4 = main_convergence(; dealii_initialised=true,
                                   use_dealii_solver=true,
                                   polydeg=4)
   if mpi_isroot()
      convergence_table(errors_l2_p3; title = "deal.II coupling, polydeg = 3")
      convergence_table(errors_l2_p4; title = "deal.II coupling, polydeg = 4")
   end
end

function reproduce_section42(; dealii_initialised = false)
   if !dealii_initialised
      dealii_ret = dealii_init_libs(ARGS)
   end
   tend = 20.0
   sol,error,vals = elixir_jeans(; dealii_initialised=true, tend)
   (; val_t, val_energy_kinetic, val_energy_kin_ana, val_energy_internal, val_energy_int_ana, val_energy_potential, val_energy_pot_ana) = vals


   if mpi_isroot()
      t2 = 1
      t18 = 1
      for i = 1:size(val_t,1)
         if val_t[t2] <= 2.0
            t2 = i
         end
         if val_t[t18] <= 18.0
            t18 = i
         end
      end
      @show size(val_t) t2 t18
      plot(yguide="energy [erg]", xguide = "time [s]"; leg=:topright)
      ylims!(-10.0,10.0)
      xlims!(0.0,val_t[t2])
      plot!(val_t[1:t2], val_energy_kin_ana[1:t2], label = "kinetic energy", linecolor="orange", linewidth=3)
      scatter!(val_t[1:2:t2], val_energy_kinetic[1:2:t2], label = :none, markercolor="orange", markershape=:xcross, markerstrokewidth=2)
      plot!(val_t[1:t2], val_energy_int_ana[1:t2], label = "internal energy (rel)", linecolor="violet", linewidth=3)
      scatter!(val_t[1:2:t2], val_energy_internal[1:2:t2] .- val_energy_internal[1], label = :none, markercolor="violet", markershape=:xcross, markerstrokewidth=2)
      plot!(val_t[1:t2], val_energy_pot_ana[1:t2], label = "potential energy", linecolor="olive", linewidth=3)
      scatter!(val_t[1:2:t2], val_energy_potential[1:2:t2], label = :none, markercolor="olive", markershape=:xcross, markerstrokewidth=2)
      savefig("jeans_dealii_st.pdf")
      print("\n\n  Energy plot saved to file `./jeans_dealii_st.pdf`\n")

      plot(yguide="energy [erg]", xguide = "time [s]"; leg=:topright)
      ylims!(-10.0,10.0)
      xlims!(val_t[t18],tend)
      plot!(val_t[t18:end], val_energy_kin_ana[t18:end], label = "kinetic energy", linecolor="orange", linewidth=3)
      scatter!(val_t[t18:2:end], val_energy_kinetic[t18:2:end], label = :none, markercolor="orange", markershape=:xcross, markerstrokewidth=2)
      plot!(val_t[t18:end], val_energy_int_ana[t18:end], label = "internal energy (rel)", linecolor="violet", linewidth=3)
      scatter!(val_t[t18:2:end], val_energy_internal[t18:2:end] .- val_energy_internal[1], label = :none, markercolor="violet", markershape=:xcross, markerstrokewidth=2)
      plot!(val_t[t18:end], val_energy_pot_ana[t18:end], label = "potential energy", linecolor="olive", linewidth=3)
      scatter!(val_t[t18:2:end], val_energy_potential[t18:2:end], label = :none, markercolor="olive", markershape=:xcross, markerstrokewidth=2)
      savefig("jeans_dealii_lt.pdf")
      print("\n\n  Energy plot saved to file `./jeans_dealii_lt.pdf`\n")
   end
end

function reproduce_section43_profiles(; dealii_initialised = false)
   if !dealii_initialised
      dealii_ret = dealii_init_libs(ARGS)
   end
   ref1, error = elixir_sedovblastwave_2d(; dealii_initialised=true,
                                          initial_refinement_level = 8,
                                          amr_interval = 0, tend=0.5)
   ref2, error = elixir_sedovblastwave_2d(; dealii_initialised=true,
                                          initial_refinement_level = 8,
                                          amr_interval = 0, tend=1.0)
   sol1, error = elixir_sedovblastwave_2d(; dealii_initialised=true,
                                          amr_interval = 1, tend=0.5)
   sol2, error = elixir_sedovblastwave_2d(; dealii_initialised=true,
                                          amr_interval = 1, tend=1.0)

   function linesegment(a, b, n_points)
      coordinates = zeros(2, n_points)
      for i in 1:n_points
         coordinates[:,i] = (i-1.0)/n_points * (b - a) + a
      end
      coordinates
   end
   
   pdref1 = PlotData1D(ref1, curve=linesegment([0.0,0.0], [4.0,0.0], 200))
   pdsol1 = PlotData1D(sol1, curve=linesegment([0.0,0.0], [4.0,0.0], 200))
   pdref2 = PlotData1D(ref2, curve=linesegment([0.0,0.0], [4.0,0.0], 200))
   pdsol2 = PlotData1D(sol2, curve=linesegment([0.0,0.0], [4.0,0.0], 200))
   if mpi_isroot()
      plot(yguide="density [g/cm³]", xguide = "distance [cm]"; leg=:topright)
      ylims!(0.0, 4.0)
      xlims!(0.0, 4.0)
      plot!(pdref1["rho"], label="ρ", linecolor="violet", title="", linewidth=4)
      plot!(pdsol1["rho"], label=:none, markercolor="red", title="", markershape=:xcross, markerstrokewidth=3)
      plot!(yguide="density [g/cm³]", xguide = "distance [cm]"; leg=:topright)
      savefig("sedovblast_density_profile_0.5.pdf")
      print("\n\n  Density profile plot saved to file `./sedovblast_density_profile_0.5.pdf`\n")
      plot(yguide="density [g/cm³]", xguide = "distance [cm]"; leg=:topright)
      ylims!(0.0, 4.0)
      xlims!(0.0, 4.0)
      plot!(pdref2["rho"], label="ρ", linecolor="violet", title="", linewidth=4)
      scatter!(pdsol2["rho"], label=:none, markercolor="red", title="", markershape=:xcross, markerstrokewidth=3)
      plot!(yguide="density [g/cm³]", xguide = "distance [cm]"; leg=:topright)
      savefig("sedovblast_density_profile_1.0.pdf")
      print("\n\n  Density profile plot saved to file `./sedovblast_density_profile_1.0.pdf`\n")
   end
end

function reproduce_section43_plotdata(; dealii_initialised = false)
   if !dealii_initialised
      dealii_ret = dealii_init_libs(ARGS)
   end
   elixir_sedovblastwave_2d(; dealii_initialised=true,
                            amr_interval = 1, tend=2.2)
end

function rebenchmark_section44_amr(; dealii_initialised = false, save_solution = false, amr_interval = 1)
   if !dealii_initialised
      dealii_ret = dealii_init_libs(ARGS)
   end
   elixir_sedovblastwave_3d(; dealii_initialised=true, save_solution, tend=0.5)
end

function rebenchmark_section44_fixed(; dealii_initialised = false, save_solution = false)
   if !dealii_initialised
      dealii_ret = dealii_init_libs(ARGS)
   end
   initial_refinement_level = 6
   elixir_small_sedovblastwave_3d(; dealii_initialised=true, initial_refinement_level, save_solution, amr_interval=0, tend=0.5)
end
