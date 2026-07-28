include("../euler_gravity.jl")

# Original file: <trixi>/examples/paper_self_gravitating_gas_dynamics/elixir_eulergravity_convergence.jl
function elixir_convergence(; initial_refinement_level = 2,
                              use_dealii_solver = true,
                              polydeg = 3,
                              kwargs...)
  initial_condition = initial_condition_eoc_test_coupled_euler_gravity

  ###############################################################################
  # semidiscretization of the compressible Euler equations
  gamma = 2.0
  equations_euler = CompressibleEulerEquations2D(gamma)

  solver_euler = DGSEM(polydeg, flux_hll)

  coordinates_min = (0.0, 0.0)
  coordinates_max = (2.0, 2.0)
  trees_per_dimension = (1, 1)
  mesh = P4estMesh(trees_per_dimension;
                   polydeg=1, coordinates_min, coordinates_max,
                   initial_refinement_level)
  meshsettings = MeshSettings2D(false, trees_per_dimension; coordinates_min,
                                coordinates_max, initial_refinement_level)
  if use_dealii_solver
    dealii = dealii_init(polydeg, meshsettings)
  end

  semi_euler = SemidiscretizationHyperbolic(mesh, equations_euler, initial_condition, solver_euler,
                                            source_terms=source_terms_eoc_test_coupled_euler_gravity)

  ###############################################################################
  # (dummy) semidiscretization of the hyperbolic diffusion equations
  equations_gravity = HyperbolicDiffusionEquations2D()

  solver_gravity = DGSEM(polydeg, flux_lax_friedrichs)

  semi_gravity = SemidiscretizationHyperbolic(mesh, equations_gravity, initial_condition, solver_gravity,
                                              source_terms=source_terms_harmonic)

  ###############################################################################
  # combining both semidiscretizations for Euler + self-gravity
  if use_dealii_solver
    gravity_solver = GravitySolverDealII(dealii)
  else
    gravity_solver = GravitySolverRK(1.1, timestep_gravity_erk52_3Sstar!)
  end
  parameters = ParametersEulerGravity(background_density=2.0, # aka rho0
                                      # rho0 is (ab)used to add a "+8π" term to the
                                      # source terms for the manufactured solution
                                      gravitational_constant=1.0, # aka G
                                      resid_tol=1.0e-10,
                                      n_iterations_max=1000,
                                      gravity_solver=gravity_solver)

  semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters)

  ###############################################################################
  # ODE solvers, callbacks etc.
  tspan = (0.0, 1.0)
  ode = semidiscretize(semi, tspan)

  summary_callback = SummaryCallback()

  stepsize_callback = StepsizeCallback(cfl=0.8)

  save_solution = SaveSolutionCallback(interval=10,
                                       save_initial_solution=true,
                                       save_final_solution=true,
                                       solution_variables=cons2prim)

  analysis_interval = 100
  alive_callback = AliveCallback(analysis_interval=analysis_interval)

  analysis_callback = AnalysisCallback(semi_euler, interval=analysis_interval,
                                       save_analysis=false)

  callbacks = CallbackSet(summary_callback, stepsize_callback,
                          save_solution, analysis_callback,
                          alive_callback)


  ###############################################################################
  # run the simulation
  sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false);
              dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
              save_everystep=false, callback=callbacks, kwargs...);
  summary_callback() # print the timer summary
  mpi_println("Number of gravity subcycles: ", semi.gravity_counter.ncalls_since_readout)

  return (; sol, error = analysis_callback(sol))
end


# Refine top right quarter and coarsen lower left quarter of domain for domain of size [0, 2] × [0, 2]
mutable struct IndicatorCoarsenRefine{Cache <: NamedTuple} <: Trixi.AbstractIndicator
   cache::Cache
end

function IndicatorCoarsenRefine(semi)
   basis = semi.solver.basis
   alpha = Vector{real(basis)}()
   cache = (; semi.mesh, alpha)
   return IndicatorCoarsenRefine{typeof(cache)}(cache)
end

function (indicator::IndicatorCoarsenRefine)(u::AbstractArray{<:Any,4},
                                             mesh, equations, dg, cache;
                                             t, kwargs...)
   mesh = indicator.cache.mesh
   alpha = indicator.cache.alpha
   resize!(alpha, nelements(dg, cache))
   @info "IndicatorCoarsenRefine" indicator

   for element in eachelement(dg, cache)
      for j in eachnode(dg), i in eachnode(dg)
         x = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
         if x[1] > 1 && x[2] > 1
            alpha[element] = 1
         elseif x[1] < 1 && x[2] < 1
            alpha[element] = -1
         end
      end
   end

   return alpha
end

# Refine right half of domain for domain of size [0, 2] × [0, 2]
mutable struct IndicatorRefineRight{Cache <: NamedTuple} <: Trixi.AbstractIndicator
  cache::Cache
end

function IndicatorRefineRight(semi)
   basis = semi.solver.basis
   alpha = Vector{real(basis)}()
   cache = (; semi.mesh, alpha)
   return IndicatorRefineRight{typeof(cache)}(cache)
end

function (indicator::IndicatorRefineRight)(u::AbstractArray{<:Any,4},
                                           mesh, equations, dg, cache;
                                           t, kwargs...)
  alpha = indicator.cache.alpha #zeros(Int, nelements(dg, cache))
  resize!(alpha, nelements(dg, cache))
  @info "IndicatorRefineRight" indicator

  for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      x = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
      if x[1] > 1
        alpha[element] = 1
      end
    end
  end

  return alpha
end


# Coarsen right half of domain for domain of size [0, 2] × [0, 2]
mutable struct IndicatorCoarsenRight{Cache <: NamedTuple} <: Trixi.AbstractIndicator
  cache::Cache
end

function IndicatorCoarsenRight(semi)
    basis = semi.solver.basis
    alpha = Vector{real(basis)}()
    cache = (; semi.mesh, alpha)
    return IndicatorCoarsenRight{typeof(cache)}(cache)
end

function (indicator::IndicatorCoarsenRight)(u::AbstractArray{<:Any,4},
                                            mesh, equations, dg, cache;
                                            t, kwargs...)
  alpha = cache.alpha
  resize!(alpha, nelements(dg, cache))
  @info "IndicatorCoarsenRight" indicator

  for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      x = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
      if x[1] > 1
        alpha[element] = -1
      end
    end
  end

  return alpha
end


# Original file: <trixi>/examples/paper_self_gravitating_gas_dynamics/elixir_eulergravity_convergence.jl
function elixir_convergence_amr(; initial_refinement_level = 2,
                                  indicatorType = IndicatorRefineRight :: Type,
                                  polydeg = 3,
                                  kwargs...)
  initial_condition = initial_condition_eoc_test_coupled_euler_gravity

  ###############################################################################
  # semidiscretization of the compressible Euler equations
  gamma = 2.0
  equations_euler = CompressibleEulerEquations2D(gamma)

  solver_euler = DGSEM(polydeg, flux_hll)

  coordinates_min = (0.0, 0.0)
  coordinates_max = (2.0, 2.0)
  trees_per_dimension = (1, 1)
  mesh = P4estMesh(trees_per_dimension;
                   polydeg=1, coordinates_min, coordinates_max,
                   initial_refinement_level)
  meshsettings = MeshSettings2D(false, trees_per_dimension; coordinates_min,
                                coordinates_max, initial_refinement_level)
  if use_dealii_solver
    dealii = dealii_init(polydeg, meshsettings)
  end

  semi_euler = SemidiscretizationHyperbolic(mesh, equations_euler, initial_condition, solver_euler,
                                            source_terms=source_terms_eoc_test_coupled_euler_gravity)

  ###############################################################################
  # (dummy) semidiscretization of the hyperbolic diffusion equations
  equations_gravity = HyperbolicDiffusionEquations2D()

  solver_gravity = DGSEM(polydeg, flux_lax_friedrichs)

  semi_gravity = SemidiscretizationHyperbolic(mesh, equations_gravity, initial_condition, solver_gravity,
                                              source_terms=source_terms_harmonic)

  ###############################################################################
  # combining both semidiscretizations for Euler + self-gravity
  if use_dealii_solver
    gravity_solver = GravitySolverDealII(dealii)
  else
    gravity_solver = GravitySolverRK(1.1, timestep_gravity_erk52_3Sstar!)
  end
  parameters = ParametersEulerGravity(background_density=2.0, # aka rho0
                                      # rho0 is (ab)used to add a "+8π" term to the
                                      # source terms for the manufactured solution
                                      gravitational_constant=1.0, # aka G
                                      resid_tol=1.0e-10,
                                      n_iterations_max=1000,
                                      gravity_solver=gravity_solver)

  semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters)

  ###############################################################################
  # ODE solvers, callbacks etc.
  tspan = (0.0, 1.0)
  ode = semidiscretize(semi, tspan)

  summary_callback = SummaryCallback()

  save_solution = SaveSolutionCallback(interval=10,
                                       save_initial_solution=true,
                                       save_final_solution=true,
                                       solution_variables=cons2prim)

  analysis_interval = 100
  alive_callback = AliveCallback(analysis_interval=analysis_interval)

  analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                       save_analysis=false)

  amr_controller = ControllerThreeLevel(semi, indicatorType(semi.semi_euler),
                                        base_level=initial_refinement_level-1,
                                        med_level=0, med_threshold=0.0,
                                        max_level=initial_refinement_level+1, max_threshold=0.5)

  amr_callback = AMRCallback(semi, amr_controller,
                             interval=10,
                             adapt_initial_condition=false,
                             adapt_initial_condition_only_refine=true,
                             dynamic_load_balancing=false)

  stepsize_callback = StepsizeCallback(cfl=0.8)

  callbacks = CallbackSet(summary_callback,
                          save_solution,
                          analysis_callback, alive_callback,
                          amr_callback,
                          stepsize_callback)


  ###############################################################################
  # run the simulation
  sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false);
              dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
              save_everystep=false, callback=callbacks, kwargs...);
  summary_callback() # print the timer summary
  mpi_println("Number of gravity subcycles: ", semi.gravity_counter.ncalls_since_readout)

  return (; sol, error = analysis_callback(sol))
end



function convergence_table(errors_l2; kwargs...)
  data = zeros(size(errors_l2, 1), 2 * size(errors_l2, 2))
  for j in axes(errors_l2, 2), i in axes(errors_l2, 1)
      data[i, 2 * j - 1] = errors_l2[i, j]
      if i == 1
          data[i, 2 * j] = NaN
      else
          data[i, 2 * j] = log(errors_l2[i - 1, j] / errors_l2[i, j]) / log(2)
      end
  end
  pretty_table(data;
    column_labels = ["rho", "EOC", "rho v1", "EOC", "rho v2", "EOC", "rho e", "EOC"],
    formatters = [fmt__printf("%.2e", [1, 3, 5, 7]),
                  fmt__printf("%.2f", [2, 4, 6, 8])],
    kwargs...)
end

function main_convergence(; dealii_initialised = false, use_dealii_solver = false, polydeg = 3)
  refinements = 1:5

  if use_dealii_solver
    if !dealii_initialised
      dealii_ret = dealii_init_libs(ARGS)
    end
    # deal.II
    errors_l2 = zeros(length(refinements), 4)
    runtimes = zeros(length(refinements))
    for (i, initial_refinement_level) in enumerate(refinements)
      runtimes[i] = @elapsed begin
        results = elixir_convergence(; initial_refinement_level, use_dealii_solver, polydeg)
      end
      errors_l2[i, :] = results.error.l2
    end
    if mpi_isroot()
      @info "deal.II" runtimes
      convergence_table(errors_l2; title = "deal.II")
    end
  else
    # pseudo time stepping
    errors_l2 = zeros(length(refinements), 4)
    runtimes = zeros(length(refinements))
    for (i, initial_refinement_level) in enumerate(refinements)
      gravity_solver = GravitySolverRK(1.1, timestep_gravity_erk52_3Sstar!)
      runtimes[i] = @elapsed begin
        results = elixir_convergence(; initial_refinement_level, use_dealii_solver, polydeg)
      end
      errors_l2[i, :] = results.error.l2
    end
    if mpi_isroot()
      @info "Pseudo time stepping" runtimes
      convergence_table(errors_l2; title = "Pseudo time stepping")
    end
  end

  return errors_l2
end
