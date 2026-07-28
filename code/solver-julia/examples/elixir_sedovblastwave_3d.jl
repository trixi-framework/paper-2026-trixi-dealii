include("../euler_gravity.jl")

"""
initial_condition_sedov_self_gravity(x, t, equations::CompressibleEulerEquations3D)

Adaptation of the Sedov blast wave with self-gravity taken from
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
based on
- https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node187.html#SECTION010114100000000000000
Should be used together with [`boundary_condition_sedov_self_gravity`](@ref).
"""
function initial_condition_sedov_self_gravity(x, t, equations::CompressibleEulerEquations3D)
    # Set up polar coordinates
    r = sqrt(x[1]^2 + x[2]^2 + x[3]^2)

    # Setup based on https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node187.html#SECTION010114100000000000000
    r0 = 0.125 # = 4.0 * smallest dx (for domain length=8 and max-ref=8)
    E = 1.0
    p_inner = 3 * (equations.gamma - 1) * E / (4 * pi * r0^3)
    p_ambient = 1.0e-5 # = true Sedov setup

    # Calculate primitive variables
    # use a logistic function to transfer density value smoothly
    r_ini = 1.0   # center point of function
    k = -150.0 # sharpness of transfer
    logistic_function_rho = 1.0 / (1.0 + exp(-k * (r - r_ini)))
    rho_ambient = 1e-5
    rho = max(logistic_function_rho, rho_ambient) # clip background density to not be so tiny

    # velocities are zero
    v1 = 0.0
    v2 = 0.0
    v3 = 0.0

    # use a logistic function to transfer pressure value smoothly
    logistic_function_p = p_inner / (1.0 + exp(-k * (r - r0)))
    p = max(logistic_function_p, p_ambient)

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

function initial_condition_sedov_self_gravity(x, t,
                                              equations::HyperbolicDiffusionEquations3D)
    # for now just use constant initial condition for sedov blast wave (can likely be improved)
    phi = 0.0
    q1 = 0.0
    q2 = 0.0
    q3 = 0.0
    return SVector(phi, q1, q2, q3)
end

function elixir_sedovblastwave_3d(; initial_refinement_level = 2,
                                  allow_repartitioning = false,
                                  save_analysis = false,
                                  save_solution = false,
                                  analysis_interval = 100,
                                  amr_interval = 1,
                                  resid_tol = 1.0e-6,
                                  tend = 0.5,
                                  polydegree = 3,
                                  dealii_initialised = false)

   initial_condition = initial_condition_sedov_self_gravity

   ###############################################################################
   # semidiscretization of the compressible Euler equations
   gamma = 1.4
   equations_euler = CompressibleEulerEquations3D(gamma)

   surface_flux = FluxLaxFriedrichs(max_abs_speed_naive)
   volume_flux = flux_ranocha
   basis = LobattoLegendreBasis(polydegree)
   indicator_sc = IndicatorHennemannGassner(equations_euler, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
   volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                    volume_flux_dg = volume_flux,
                                                    volume_flux_fv = surface_flux)

   solver_euler = DGSEM(polydeg = polydegree, surface_flux = surface_flux, volume_integral = volume_integral)

   coordinates_min = (-2.0, -2.0, -2.0)
   coordinates_max = ( 2.0,  2.0,  2.0)
   trees_per_dimension = (1, 1, 1)
   mesh = P4estMesh(trees_per_dimension; polydeg=1, coordinates_min, coordinates_max,
                    initial_refinement_level)
   meshsettings = MeshSettings3D(allow_repartitioning, trees_per_dimension, polydeg=1;
                                 coordinates_min, coordinates_max, initial_refinement_level)
   if !dealii_initialised
       dealii_ret = dealii_init_libs(ARGS)
   end
   dealii = dealii_init(polydegree, meshsettings)

   semi_euler = SemidiscretizationHyperbolic(mesh, equations_euler, initial_condition, solver_euler)

   ###############################################################################
   # semidiscretization of the hyperbolic diffusion equations
   equations_gravity = HyperbolicDiffusionEquations3D()

   solver_gravity = DGSEM(polydegree, flux_lax_friedrichs)

   semi_gravity = SemidiscretizationHyperbolic(mesh, equations_gravity,
                                               initial_condition, solver_gravity,
                                               source_terms=source_terms_harmonic)

   ###############################################################################
   # combining both semidiscretizations for Euler + self-gravity
   gravity_solver = GravitySolverDealII(dealii)

   parameters = ParametersEulerGravity(background_density=0.0, #1.5e7, # aka rho0
                                       gravitational_constant=6.674e-8, # aka G
                                       resid_tol = resid_tol,
                                       resid_tol_type = :l2_phi,
                                       n_iterations_max=1000,
                                       gravity_solver=gravity_solver)

   semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters)

   ###############################################################################
   # ODE solvers, callbacks etc.
   tspan = (0.0, tend)
   ode = semidiscretize(semi, tspan)

   summary_callback = SummaryCallback()

   stepsize_callback = StepsizeCallback(cfl=1.0)

   savesol_callback = SaveSolutionCallback(dt=0.01,
                                           save_initial_solution=true,
                                           save_final_solution=true)

   alive_callback = AliveCallback(analysis_interval=analysis_interval)

   analysis_callback = AnalysisCallback(semi_euler, interval=analysis_interval,
                                        save_analysis=save_analysis)

   amr_indicator = IndicatorHennemannGassner(semi,
                                             alpha_max = 1.0,
                                             alpha_min = 0.0,
                                             alpha_smooth = false,
                                             variable = density_pressure)
   amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                         base_level=2,
                                         max_level=7,max_threshold=0.0003)
   amr_callback = AMRCallback(semi, amr_controller, 
                              interval=amr_interval,
                              adapt_initial_condition=true,
                              adapt_initial_condition_only_refine=true,
                              dynamic_load_balancing=allow_repartitioning)

   callbacks = CallbackSet(summary_callback, stepsize_callback,
                           analysis_callback, alive_callback)
   if amr_interval > 0
       callbacks = CallbackSet(amr_callback, callbacks)
   end
   if save_solution
       callbacks = CallbackSet(savesol_callback, callbacks)
   end

   ###############################################################################
   # run the simulation
   sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false);
               dt=0.1, # solve needs some value here but it will be overwritten by the stepsize_callback
               save_everystep=false, callback=callbacks);
   summary_callback() # print the timer summary
   mpi_println("Number of gravity subcycles: ", semi.gravity_counter.ncalls_since_readout)

   error = analysis_callback(sol)

   sol, error
end # elixir_sedovblastwave_3d
