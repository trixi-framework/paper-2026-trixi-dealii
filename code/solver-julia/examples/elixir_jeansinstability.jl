include("../euler_gravity.jl")

γ = 5.0/3.0 # heat capacity ratio
p0 = 1.5e7 # pressure
ϱ0 = 1.5e7 # density
δ = 1e-3 # perturbation
G = 6.674e-8 # gravitational constant
kv = [2π/0.5; 0.0] # perturbation wave vector
k = sum(kv) # wavenumber

# analytic kinetic energy
function E_kin(t,ω,L,k)
  0.125*(ϱ0*δ^2*abs(ω)^2*L^2)/(k^2)*(1 - cos(2ω*t))
end

# analytic relative internal/thermal energy
function E_int(t,ω,L,c0)
    -0.125*(ϱ0*c0^2*δ^2*L^2)*(1 - cos(2ω*t))
end

# analytic potential energy
function E_pot(t,ω,L,k,G)
    -0.5*(π*G*ϱ0^2*δ^2*L^2)/(k^2)*(1 + cos(2ω*t))
end


# Original file: <trixi>/examples/paper_self_gravitating_gas_dynamics/elixir_eulergravity_jeans_instability.jl
"""
    initial_condition_jeans_instability(x, t,
                                        equations::Union{CompressibleEulerEquations2D,
                                                         HyperbolicDiffusionEquations2D})

The classical Jeans instability taken from
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
- Dominik Derigs, Andrew R. Winters, Gregor J. Gassner, Stefanie Walch (2016)
  A Novel High-Order, Entropy Stable, 3D AMR MHD Solver with Guaranteed Positive Pressure
  [arXiv: 1605.03572](https://arxiv.org/abs/1605.03572)
- Flash manual https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
in CGS (centimeter, gram, second) units.
"""
function initial_condition_jeans_instability(x, t,
                                             equations::CompressibleEulerEquations2D)
  # Jeans gravitational instability test case
  # see Derigs et al. https://arxiv.org/abs/1605.03572; Sec. 4.6
  # OBS! this uses cgs (centimeter, gram, second) units
  # periodic boundaries
  # domain size [0,L]^2 depends on the wave number chosen for the perturbation
  # gamma = 5/3
  dens0  = ϱ0 # g/cm^3
  pres0  = p0 # dyn/cm^2
  delta0 = δ
  # set wave vector values for pertubation (units 1/cm)
  # see FLASH manual: https://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel.pdf
  kx = kv[1]
  ky = kv[2]
  # perturb density and pressure away from reference states ρ_0 and p_0
  dens = dens0*(1.0 + delta0*cos(kv'x))                 # g/cm^3
  pres = pres0*(1.0 + equations.gamma*delta0*cos(kv'x)) # dyn/cm^2
  # flow starts as stationary
  velx = 0.0 # cm/s
  vely = 0.0 # cm/s
  return prim2cons((dens, velx, vely, pres), equations)
end

function initial_condition_jeans_instability(x, t,
                                             equations::HyperbolicDiffusionEquations2D)
  # gravity equation: -Δϕ = -4πGρ
  # Constants taken from the FLASH manual
  # https://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel.pdf
  rho0   = ϱ0
  delta0 = δ

  phi = rho0*delta0 # constant background pertubation magnitude
  q1  = 0.0
  q2  = 0.0
  return (phi, q1, q2)
end

Trixi.pretty_form_utf(::Val{:energy_potential}) = "∑e_potential"
Trixi.pretty_form_ascii(::Val{:energy_potential}) = "e_potential"

function Trixi.analyze(::Val{:energy_potential}, du, u_euler, t, semi::SemidiscretizationEulerGravity)

  u_gravity = Trixi.wrap_array(semi.cache.u_ode, semi.semi_gravity)

  mesh, equations_euler, dg, cache = Trixi.mesh_equations_solver_cache(semi.semi_euler)
  _, equations_gravity, _, _ = Trixi.mesh_equations_solver_cache(semi.semi_gravity)

  e_potential = Trixi.integrate_via_indices(u_euler, mesh, equations_euler, dg, cache, equations_gravity, u_gravity) do u, i, j, element, equations_euler, dg, equations_gravity, u_gravity
    u_euler_local   = Trixi.get_node_vars(u_euler,   equations_euler,   dg, i, j, element)
    u_gravity_local = Trixi.get_node_vars(u_gravity, equations_gravity, dg, i, j, element)
    # OBS! subtraction is specific to Jeans instability test where rho0 = 1.5e7
    return 0.5*(u_euler_local[1] - ϱ0) * u_gravity_local[1]
  end
  return e_potential
end

function elixir_jeans(; initial_refinement_level = 4,
                      analysis_interval = 100,
                      resid_tol = 1.0e-7,
                      tend = 2.0,
                      dealii_initialised = false)

  c0(γ,p0,ϱ0) = sqrt(γ*p0/ϱ0)
  kJ(G,p0,ϱ0,c0) = sqrt(4π*G*ϱ0)/c0
  ω(k,G,c0,ϱ0) = sqrt(Complex(c0^2*k^2 - 4π*G*ϱ0))
  L(G,γ,p0,ϱ0) = 0.5*sqrt(π*γ*p0 / (G*ϱ0^2))
  
  val_c0 = c0(γ,p0,ϱ0)
  val_kJ = kJ(G,p0,ϱ0,val_c0)
  val_ω = ω(k,G,val_c0,ϱ0)
  val_L = L(G,γ,p0,ϱ0)
  
  if imag(val_ω) != 0.0
    @error "unstable simulation"
  end

  initial_condition = initial_condition_jeans_instability
  
  ###############################################################################
  # semidiscretization of the compressible Euler equations
  gamma = γ
  equations_euler = CompressibleEulerEquations2D(gamma)
  
  polydegree = 3
  solver_euler = DGSEM(polydegree, flux_hll)
  
  val_L = 1.0
  coordinates_min = (0, 0)
  coordinates_max = (val_L, val_L)
  trees_per_dimension = (1, 1)
  mesh = P4estMesh(trees_per_dimension; polydeg=1,
                   coordinates_min, coordinates_max,
                   initial_refinement_level)
  meshsettings = MeshSettings2D(false, trees_per_dimension; coordinates_min,
                                coordinates_max, initial_refinement_level)
  if !dealii_initialised
    dealii_ret = dealii_init_libs(ARGS)
  end
  dealii = dealii_init(polydegree, meshsettings)
  
  semi_euler = SemidiscretizationHyperbolic(mesh, equations_euler, initial_condition,
                                            solver_euler)
  
  ###############################################################################
  # semidiscretization of the hyperbolic diffusion equations
  equations_gravity = HyperbolicDiffusionEquations2D()
  
  solver_gravity = DGSEM(polydegree, flux_lax_friedrichs)
  
  semi_gravity = SemidiscretizationHyperbolic(mesh, equations_gravity,
                                              initial_condition, solver_gravity;
                                              source_terms=source_terms_harmonic)
  
  ###############################################################################
  # combining both semidiscretizations for Euler + self-gravity
  gravity_solver = GravitySolverDealII(dealii)
  parameters = ParametersEulerGravity(background_density=ϱ0,
                                      gravitational_constant=G,
                                      resid_tol = resid_tol,
                                      resid_tol_type = :l2_phi,
                                      n_iterations_max=10000,
                                      gravity_solver=gravity_solver)
  
  semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters)
  
  ###############################################################################
  # ODE solvers, callbacks etc.
  tspan = (0.0, tend)
  ode = semidiscretize(semi, tspan)
  
  summary_callback = SummaryCallback()
  
  stepsize_callback = StepsizeCallback(cfl=1.0)
  
  save_solution = SaveSolutionCallback(interval=10,
                                       save_initial_solution=true,
                                       save_final_solution=true,
                                       solution_variables=cons2prim)
  
  alive_callback = AliveCallback(analysis_interval=analysis_interval)
  
  analysis_callback = AnalysisCallback(semi_euler, interval=analysis_interval,
                                       save_analysis=false,
                                       extra_analysis_integrals=(energy_total, energy_kinetic, energy_internal, Val(:energy_potential)))
  
  val_t = Vector{Float64}()
  val_energy_total = Vector{Float64}()
  val_energy_kinetic = Vector{Float64}()
  val_energy_internal = Vector{Float64}()
  val_energy_potential = Vector{Float64}()
  val_energy_kin_ana = Vector{Float64}()
  val_energy_int_ana = Vector{Float64}()
  val_energy_pot_ana = Vector{Float64}()
  function affect!(integrator)
    semi = integrator.p
    du = Trixi.wrap_array(get_du(integrator), semi)
    u = Trixi.wrap_array(integrator.u, semi)
    t = integrator.t
  
    val = t
    mpi_isroot() && push!(val_t, val)
    val = Trixi.analyze(energy_total, du, u, t, semi)
    mpi_isroot() && push!(val_energy_total, val)
    val = Trixi.analyze(energy_kinetic, du, u, t, semi)
    mpi_isroot() && push!(val_energy_kinetic, val)
    val = Trixi.analyze(energy_internal, du, u, t, semi)
    mpi_isroot() && push!(val_energy_internal, val)
    val = Trixi.analyze(Val{:energy_potential}(), du, u, t, semi)
    mpi_isroot() && push!(val_energy_potential, val)
    mpi_isroot() && push!(val_energy_kin_ana, E_kin(t, val_ω, val_L, k))
    mpi_isroot() && push!(val_energy_int_ana, E_int(t, val_ω, val_L, val_c0))
    mpi_isroot() && push!(val_energy_pot_ana, E_pot(t, val_ω, val_L, k, G))
    u_modified!(integrator, false)
  end
  energy_callback = DiscreteCallback(
    (u, t, integrator) -> true, affect!;
    initialize = (c, u, t, integrator) -> affect!(integrator),
    save_positions = (false, false)
  )
  
  callbacks = CallbackSet(summary_callback, stepsize_callback,
                          save_solution,
                          energy_callback,
                          analysis_callback, alive_callback)
  
  
  ###############################################################################
  # run the simulation
  sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false);
              dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
              save_everystep=false, callback=callbacks);
  summary_callback() # print the timer summary
  mpi_println("Number of gravity subcycles: ", semi.gravity_counter.ncalls_since_readout)
  
  error = analysis_callback(sol)
  vals = (; val_t, val_energy_kinetic, val_energy_kin_ana, val_energy_internal, val_energy_int_ana, val_energy_potential, val_energy_pot_ana)
  
  sol,error,vals
end # function elixir_jeans
