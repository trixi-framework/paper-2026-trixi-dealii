using LinearAlgebra: norm

using OrdinaryDiffEq
# The only parts used in the code below
using OrdinaryDiffEq: DiscreteCallback, get_du, u_modified!

using Trixi
using Trixi: PerformanceCounter, allocate_coefficients, compute_coefficients!,
             mesh_equations_solver_cache, polydeg, @trixi_timeit, timer,
             wrap_array, max_dt, calc_error_norms,
             summary_header, summary_line, summary_box, summary_footer,
             increment_indent,
             MPI, mpi_isparallel, mpi_comm, mpi_isroot, mpi_println, mpi_nranks,
             ncells, cfunction, iterate_p4est, copy_to_quad_iter_volume, refine!, coarsen!,
             partition!, rebalance_solver!, reinitialize_boundaries!

using PrettyTables: pretty_table, fmt__printf


"""
    ParametersEulerGravity(; background_density=0.0,
                             gravitational_constant=1.0,
                             resid_tol=1.0e-4,
                             resid_tol_type=:l2_phi,
                             n_iterations_max=10^4,
                             maxiters=n_iterations_max,
                             gravity_solver=timestep_gravity_erk52_3Sstar!)

Set up parameters for the gravitational part of a
[`SemidiscretizationEulerGravity`](@ref).
"""
struct ParametersEulerGravity{RealT<:Real, GravitySolver}
  background_density    ::RealT # aka rho0
  gravitational_constant::RealT # aka G
  resid_tol             ::RealT
  resid_tol_type        ::Symbol
  maxiters              ::Int
  gravity_solver        ::GravitySolver
end

function ParametersEulerGravity(; background_density=0.0,
                                  gravitational_constant=1.0,
                                  resid_tol=1.0e-4,
                                  resid_tol_type=:l2_phi, #:l2_phi,:linf_phi,:l2_full
                                  n_iterations_max=10^4,
                                  maxiters=n_iterations_max,
                                  gravity_solver=timestep_gravity_erk52_3Sstar!)
  background_density, gravitational_constant, resid_tol = promote(
    background_density, gravitational_constant, resid_tol)

  return ParametersEulerGravity{typeof(background_density), typeof(gravity_solver)}(
    background_density, gravitational_constant, resid_tol, resid_tol_type,
    maxiters, gravity_solver)
end

function Base.show(io::IO, parameters::ParametersEulerGravity)
  print(io, "ParametersEulerGravity(")
  print(io, "  background_density=", parameters.background_density)
  print(io, ", gravitational_constant=", parameters.gravitational_constant)
  print(io, ", resid_tol=", parameters.resid_tol)
  print(io, ", resid_tol_type=", parameters.resid_tol_type)
  print(io, ", maxiters=", parameters.maxiters)
  print(io, ", gravity_solver=", parameters.gravity_solver)
  print(io, ")")
end
function Base.show(io::IO, ::MIME"text/plain", parameters::ParametersEulerGravity)
  if get(io, :compact, false)
    show(io, parameters)
  else
    setup = [
             "background density (ϱ₀)" => parameters.background_density,
             "gravitational constant (G)" => parameters.gravitational_constant,
             "resid. tol." => parameters.resid_tol,
             "resid. tol. type" => parameters.resid_tol_type,
             "max. #iterations" => parameters.maxiters,
             "gravity solver" => parameters.gravity_solver,
            ]
    summary_box(io, "ParametersEulerGravity", setup)
  end
end


"""
    SemidiscretizationEulerGravity

A struct containing everything needed to describe a spatial semidiscretization
of a the compressible Euler equations with self-gravity, reformulating the
Poisson equation for the gravitational potential as steady-state problem of
the hyperblic diffusion equations.
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  "A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics"
  [arXiv: 2008.10593](https://arXiv.org/abs/2008.10593)
"""
struct SemidiscretizationEulerGravity{SemiEuler, SemiGravity,
                                      Parameters<:ParametersEulerGravity, Cache} <: Trixi.AbstractSemidiscretization
  semi_euler::SemiEuler
  semi_gravity::SemiGravity
  parameters::Parameters
  performance_counter::PerformanceCounter
  gravity_counter::PerformanceCounter
  cache::Cache

  function SemidiscretizationEulerGravity{SemiEuler, SemiGravity, Parameters, Cache}(
      semi_euler::SemiEuler, semi_gravity::SemiGravity,
      parameters::Parameters, cache::Cache) where {SemiEuler, SemiGravity,
                                                   Parameters<:ParametersEulerGravity, Cache}
    @assert ndims(semi_euler) == ndims(semi_gravity)
    @assert typeof(semi_euler.mesh) == typeof(semi_gravity.mesh)
    @assert polydeg(semi_euler.solver) == polydeg(semi_gravity.solver)

    performance_counter = PerformanceCounter()
    gravity_counter = PerformanceCounter()

    new(semi_euler, semi_gravity, parameters, performance_counter, gravity_counter, cache)
  end
end

"""
    SemidiscretizationEulerGravity(semi_euler::SemiEuler, semi_gravity::SemiGravity, parameters)

Construct a semidiscretization of the compressible Euler equations with self-gravity.
`parameters` should be given as [`ParametersEulerGravity`](@ref).
"""
function SemidiscretizationEulerGravity(semi_euler::SemiEuler, semi_gravity::SemiGravity, parameters) where
    {Mesh, SemiEuler<:SemidiscretizationHyperbolic{Mesh, <:Trixi.AbstractCompressibleEulerEquations},
           SemiGravity<:SemidiscretizationHyperbolic{Mesh, <:Trixi.AbstractHyperbolicDiffusionEquations}}

  u_ode = compute_coefficients(zero(real(semi_gravity)), semi_gravity)
  if parameters.gravity_solver isa GravitySolverRK
    du_ode     = similar(u_ode)
    u_tmp1_ode = similar(u_ode)
    u_tmp2_ode = similar(u_ode)
    cache = (; u_ode, du_ode, u_tmp1_ode, u_tmp2_ode)
  else
    vector_source = similar(u_ode, Trixi.ndofs(semi_gravity))
    cache = (; u_ode, vector_source)
  end

  SemidiscretizationEulerGravity{typeof(semi_euler), typeof(semi_gravity), typeof(parameters), typeof(cache)}(
    semi_euler, semi_gravity, parameters, cache)
end

function Base.show(io::IO, semi::SemidiscretizationEulerGravity)
  print(io, "SemidiscretizationEulerGravity using")
  print(io,       semi.semi_euler)
  print(io, ", ", semi.semi_gravity)
  print(io, ", ", semi.parameters)
  print(io, ", cache(")
  for (idx,key) in enumerate(keys(semi.cache))
    idx > 1 && print(io, " ")
    print(io, key)
  end
  print(io, "))")
end

function Base.show(io::IO, mime::MIME"text/plain", semi::SemidiscretizationEulerGravity)
  if get(io, :compact, false)
    show(io, semi)
  else
    summary_header(io, "SemidiscretizationEulerGravity")
    summary_line(io, "semidiscretization Euler", semi.semi_euler |> typeof |> nameof)
    show(increment_indent(io), mime, semi.semi_euler)
    summary_line(io, "semidiscretization gravity", semi.semi_gravity |> typeof |> nameof)
    show(increment_indent(io), mime, semi.semi_gravity)
    summary_line(io, "parameters", semi.parameters |> typeof |> nameof)
    show(increment_indent(io), mime, semi.parameters)
    summary_footer(io)
  end
end


# The compressible Euler semidiscretization is considered to be the main
# semidiscretization. The hyperbolic diffusion equations part is only used
# internally to update the gravitational potential during an rhs! evaluation
# of the flow solver.
@inline function Trixi.mesh_equations_solver_cache(semi::SemidiscretizationEulerGravity)
  mesh_equations_solver_cache(semi.semi_euler)
end

@inline Base.real(semi::SemidiscretizationEulerGravity) = real(semi.semi_euler)

# computes the coefficients of the initial condition
function Trixi.compute_coefficients(t, semi::SemidiscretizationEulerGravity)
  u_ode = allocate_coefficients(mesh_equations_solver_cache(semi.semi_euler)...)
  compute_coefficients!(u_ode, t, semi)
  return u_ode
end

# computes the coefficients of the initial condition and stores the Euler part
# in `u_ode`
function Trixi.compute_coefficients!(u_ode, t, semi::SemidiscretizationEulerGravity)
  compute_coefficients!(semi.cache.u_ode, t, semi.semi_gravity)
  compute_coefficients!(u_ode, t, semi.semi_euler)

  return nothing
end


@inline function Trixi.calc_error_norms(func, u, t, analyzer, semi::SemidiscretizationEulerGravity, cache_analysis)
  calc_error_norms(func, u, t, analyzer, semi.semi_euler, cache_analysis)
end


function Trixi.rhs!(du_ode, u_ode, semi::SemidiscretizationEulerGravity, t)
  (; semi_euler, semi_gravity, cache) = semi

  u_euler   = wrap_array(u_ode , semi_euler)
  du_euler  = wrap_array(du_ode, semi_euler)
  u_gravity = wrap_array(cache.u_ode, semi_gravity)

  time_start = time_ns()

  # standard semidiscretization of the compressible Euler equations
  @trixi_timeit timer() "Euler solver" Trixi.rhs!(du_ode, u_ode, semi_euler, t)

  # compute gravitational potential and forces
  @trixi_timeit timer() "gravity solver" update_gravity!(semi, u_ode, semi.parameters.gravity_solver)

  # add gravitational source source_terms to the Euler part
  if ndims(semi_euler) == 1
    @views @. du_euler[2, .., :] -= u_euler[1, .., :] * u_gravity[2, .., :]
    @views @. du_euler[3, .., :] -= u_euler[2, .., :] * u_gravity[2, .., :]
  elseif ndims(semi_euler) == 2
    @views @. du_euler[2, .., :] -=  u_euler[1, .., :] * u_gravity[2, .., :]
    @views @. du_euler[3, .., :] -=  u_euler[1, .., :] * u_gravity[3, .., :]
    @views @. du_euler[4, .., :] -= (u_euler[2, .., :] * u_gravity[2, .., :] +
                                     u_euler[3, .., :] * u_gravity[3, .., :])
  elseif ndims(semi_euler) == 3
    @views @. du_euler[2, .., :] -=  u_euler[1, .., :] * u_gravity[2, .., :]
    @views @. du_euler[3, .., :] -=  u_euler[1, .., :] * u_gravity[3, .., :]
    @views @. du_euler[4, .., :] -=  u_euler[1, .., :] * u_gravity[4, .., :]
    @views @. du_euler[5, .., :] -= (u_euler[2, .., :] * u_gravity[2, .., :] +
                                     u_euler[3, .., :] * u_gravity[3, .., :] +
                                     u_euler[4, .., :] * u_gravity[4, .., :])
  else
    error("Number of dimensions $(ndims(semi_euler)) not supported.")
  end

  runtime = time_ns() - time_start
  put!(semi.performance_counter, runtime)

  return nothing
end


@inline function rhs_gravity!(du_gravity, du_ode, u_ode, semi_gravity, t, u_euler, gravity_parameters)
  G    = gravity_parameters.gravitational_constant
  rho0 = gravity_parameters.background_density
  grav_scale = -4 * G * pi

  # rhs! has the source term for the harmonic problem
  # We don't need a `@timeit timer() "rhs!"` here since that's already
  # included in the `rhs!` call.
  Trixi.rhs!(du_ode, u_ode, semi_gravity, t)

  # Source term: Jeans instability OR coupling convergence test OR blast wave
  # put in gravity source term proportional to Euler density
  # OBS! subtract off the background density ρ_0 (spatial mean value)
  @views @. du_gravity[1, .., :] += grav_scale * (u_euler[1, .., :] - rho0)

  return nothing
end


@inline function Trixi.save_solution_file(u_ode::AbstractVector, t, dt, iter,
                                          semi::SemidiscretizationEulerGravity, solution_callback,
                                          element_variables=Dict{Symbol,Any}())

  u_euler = wrap_array(u_ode, semi.semi_euler)
  filename_euler = Trixi.save_solution_file(
    u_euler, t, dt, iter,
    mesh_equations_solver_cache(semi.semi_euler)...,
    solution_callback, element_variables, system="euler")

  u_gravity = wrap_array(semi.cache.u_ode, semi.semi_gravity)
  filename_gravity = Trixi.save_solution_file(
    u_gravity, t, dt, iter,
    mesh_equations_solver_cache(semi.semi_gravity)...,
    solution_callback, element_variables, system="gravity")

  return filename_euler, filename_gravity
end



# Default RK pseudo time stepping used in the original Trixi.jl paper
struct GravitySolverRK{CFL, TimestepGravity}
  cfl::CFL
  timestep_gravity!::TimestepGravity
end

function Base.show(io::IO, gravity_solver::GravitySolverRK)
  print(io, "GravitySolverRK(")
  print(io,   gravity_solver.timestep_gravity!)
  print(io,   ", cfl=", gravity_solver.cfl)
  print(io, ")")
end

function update_gravity!(semi::SemidiscretizationEulerGravity, u_ode::AbstractVector, gravity_solver::GravitySolverRK)
  (; semi_euler, semi_gravity, parameters, gravity_counter, cache) = semi
  (; timestep_gravity!, cfl) = gravity_solver

  # Can be changed by AMR
  resize!(cache.du_ode,     length(cache.u_ode))
  resize!(cache.u_tmp1_ode, length(cache.u_ode))
  resize!(cache.u_tmp2_ode, length(cache.u_ode))

  u_euler    = wrap_array(u_ode,        semi_euler)
  u_gravity  = wrap_array(cache.u_ode,  semi_gravity)
  du_gravity = wrap_array(cache.du_ode, semi_gravity)

  # set up main loop
  (; maxiters, resid_tol, resid_tol_type) = parameters
  iter = 0
  t = zero(real(semi_gravity.solver))

  # calculate time step size sing a CFL condition once before integrating in time
  # since the mesh will not change and the linear equations have constant speeds
  dt = @trixi_timeit timer() "calculate dt" cfl * max_dt(
    u_gravity, t, semi_gravity.mesh,
    Trixi.have_constant_speed(semi_gravity.equations), semi_gravity.equations,
    semi_gravity.solver, semi_gravity.cache)

  # Evaluate the RHS after computing a stage to check the termination criterion
  # correctly. Thus, we need to pre-start the RHS evaluations at the beginning
  # in some kind of FSAL-approach.
  rhs_gravity!(du_gravity, cache.du_ode, cache.u_ode, semi_gravity, t, u_euler, parameters)

  # iterate gravity solver until convergence or maximum number of iterations
  # are reached
  while true
    # use an absolute residual tolerance check
    if resid_tol_type === :linf_phi
      residual = maximum(abs, @views du_gravity[1, .., :])
      if mpi_isparallel()
        residual = MPI.Allreduce!(Ref(residual), max, mpi_comm())[]
      end
    elseif resid_tol_type === :l2_phi
      local_norm   = @views norm(du_gravity[1, .., :])
      local_length = @views length(du_gravity[1, .., :])
      if mpi_isparallel()
        global_norm, global_length = MPI.Allreduce([local_norm, local_length], +, mpi_comm())
      else
        global_norm, global_length = local_norm, local_length
      end
      residual = global_norm / global_length
    elseif resid_tol_type === :l2_full
      local_norm   = norm(du_gravity)
      local_length = length(du_gravity)
      if mpi_isparallel()
        global_norm, global_length = MPI.Allreduce([local_norm, local_length], +, mpi_comm())
      else
        global_norm, global_length = local_norm, local_length
      end
      residual = global_norm / global_length
    else # general fallback
      residual = convert(real(eltype(du_gravity)), NaN)
    end

    if residual <= resid_tol
      break
    end

    # check if we reached the maximum number of iterations
    if maxiters > 0 && iter >= maxiters
      if mpi_isroot()
        @warn "Max iterations reached: Gravity solver failed to converge!" residual=residual t=t dt=dt
      end
      break
    end

    # evolve solution by one pseudo-time step
    time_start = time_ns()
    timestep_gravity!(cache, u_euler, t, dt, parameters, semi_gravity)
    runtime = time_ns() - time_start
    put!(gravity_counter, runtime)

    # update iteration counter
    iter += 1
    t += dt
  end

  return nothing
end


# Deal.II solver
include("dealii.jl")

struct GravitySolverDealII{DealII}
  dealii::DealII
end

function update_gravity!(semi::SemidiscretizationEulerGravity, u_ode::AbstractVector, gravity_solver::GravitySolverDealII{DealII_2D})
  (; semi_euler, semi_gravity, parameters, gravity_counter, cache) = semi
  (; dealii) = gravity_solver
  (; vector_source) = cache

  # Can be changed by AMR
  resize!(vector_source, Trixi.ndofs(semi_gravity))

  u_euler   = wrap_array(u_ode,       semi_euler)
  u_gravity = wrap_array(cache.u_ode, semi_gravity)

  # set up parameters
  (; maxiters, resid_tol, resid_tol_type) = parameters
  if resid_tol_type === :l2_phi
    dealii_set_maxiters(dealii, maxiters)
    dealii_set_abstol(dealii, resid_tol)
  else
    error("resid_tol_type $resid_tol_type not supported")
  end

  time_start = time_ns()
  G    = semi.parameters.gravitational_constant
  rho0 = semi.parameters.background_density
  grav_scale = -4 * G * pi
  @views vector_source .= vec(@. grav_scale * (u_euler[1, .., :] - rho0))
  store_f!(dealii, vector_source)
  vector_potential = vec(view(u_gravity, 1, .., :))
  store_u!(dealii, vector_potential)
  dealii_solve(dealii)
  load_u!(vector_potential, dealii)
  vector_grad_u = (vec(view(u_gravity, 2, .., :)),
                   vec(view(u_gravity, 3, .., :)))
  load_grad_u!(vector_grad_u, dealii)

  runtime = time_ns() - time_start
  put!(gravity_counter, runtime)

  return nothing
end

function update_gravity!(semi::SemidiscretizationEulerGravity, u_ode::AbstractVector, gravity_solver::GravitySolverDealII{DealII_3D})
  (; semi_euler, semi_gravity, parameters, gravity_counter, cache) = semi
  (; dealii) = gravity_solver
  (; vector_source) = cache

  # Can be changed by AMR
  resize!(vector_source, Trixi.ndofs(semi_gravity))

  u_euler   = wrap_array(u_ode,       semi_euler)
  u_gravity = wrap_array(cache.u_ode, semi_gravity)

  # set up parameters
  (; maxiters, resid_tol, resid_tol_type) = parameters
  if resid_tol_type === :l2_phi
    dealii_set_maxiters(dealii, maxiters)
    dealii_set_abstol(dealii, resid_tol)
  else
    error("resid_tol_type $resid_tol_type not supported")
  end

  time_start = time_ns()
  G    = semi.parameters.gravitational_constant
  rho0 = semi.parameters.background_density
  grav_scale = -4 * G * pi
  @views vector_source .= vec(@. grav_scale * (u_euler[1, .., :] - rho0))
  store_f!(dealii, vector_source)
  vector_potential = vec(view(u_gravity, 1, .., :))
  store_u!(dealii, vector_potential)
  dealii_solve(dealii)
  load_u!(vector_potential, dealii)
  vector_grad_u = (vec(view(u_gravity, 2, .., :)),
                   vec(view(u_gravity, 3, .., :)),
                   vec(view(u_gravity, 4, .., :)))
  load_grad_u!(vector_grad_u, dealii)

  runtime = time_ns() - time_start
  put!(gravity_counter, runtime)

  return nothing
end

using P4est

# deal.ii uses P{4,8}EST_CONNECT_FULL for regularising the mesh, hence we have to use this flag as well
function Trixi.balance!(mesh::P4estMesh{2}, init_fn = C_NULL)
    return p4est_balance(mesh.p4est, P4EST_CONNECT_FULL, init_fn)
end
function Trixi.balance!(mesh::P4estMesh{3}, init_fn = C_NULL)
    return p8est_balance(mesh.p4est, P8EST_CONNECT_FULL, init_fn)
end

@inline function (amr_callback::AMRCallback)(u_ode::AbstractVector,
                                             semi::SemidiscretizationEulerGravity,
                                             t, iter; kwargs...)
  dealii_amr_was_called = Ref(false)

  if semi.parameters.gravity_solver isa GravitySolverDealII
    # TODO the second tuple causes extra work that should not be necessary
    passive_args = ((semi.parameters.gravity_solver, amr_callback, dealii_amr_was_called, nothing, nothing),
                    (semi.cache.u_ode, mesh_equations_solver_cache(semi.semi_gravity)...))
  else
    passive_args = ((semi.cache.u_ode, mesh_equations_solver_cache(semi.semi_gravity)...),)
  end
  has_changed = amr_callback(u_ode, mesh_equations_solver_cache(semi.semi_euler)..., semi, t, iter;
                             kwargs..., passive_args=passive_args)

  if semi.parameters.gravity_solver isa GravitySolverDealII{DealII_2D}
    trixi_checksum = Trixi.P4est.p4est_checksum(semi.semi_euler.mesh.p4est)
    dealii_checksum = dealii_get_mesh_checksum(semi.parameters.gravity_solver.dealii)
  elseif semi.parameters.gravity_solver isa GravitySolverDealII{DealII_3D}
    trixi_checksum = Trixi.P4est.p8est_checksum(semi.semi_euler.mesh.p4est)
    dealii_checksum = dealii_get_mesh_checksum(semi.parameters.gravity_solver.dealii)
  end

  return has_changed
end

function (amr_callback::Trixi.AMRCallback)(u_ode::AbstractVector, mesh::P4estMesh,
                                     equations, dg::DG, cache, semi,
                                     t, iter;
                                     only_refine=false, only_coarsen=false,
                                     passive_args=())
  @assert !(only_refine && only_coarsen)
  @unpack controller, adaptor = amr_callback

  u = wrap_array(u_ode, mesh, equations, dg, cache)
  lambda = @trixi_timeit timer() "indicator" controller(u, mesh, equations, dg, cache,
                                                 t=t, iter=iter)
  if only_refine
    lambda .= lambda .* (lambda .== 1)
  elseif only_coarsen
    lambda .= lambda .* (lambda .== -1)
  end
  @boundscheck begin
    @assert axes(lambda) == (Base.OneTo(ncells(mesh)),) (
      "Indicator array (axes = $(axes(lambda))) and mesh cells (axes = $(Base.OneTo(ncells(mesh)))) have different axes"
    )
  end

  # Copy controller value of each quad to the quad's user data storage
  iter_volume_c = cfunction(copy_to_quad_iter_volume, Val(ndims(mesh)))

  # The pointer to lambda will be interpreted as Ptr{Int} above
  @assert lambda isa Vector{Int}
  iterate_p4est(mesh.p4est, lambda; iter_volume_c=iter_volume_c)

  global_cell_amount(mesh, "pre-all")
  @trixi_timeit timer() "refine" if !only_coarsen
    # Refine mesh
    refined_original_cells = @trixi_timeit timer() "mesh" refine!(mesh)

    # Refine solver 
    @trixi_timeit timer() "solver" refine!(u_ode, adaptor, mesh, equations, dg,
                                           cache, refined_original_cells)

    for (p_u_ode, p_mesh, p_equations, p_dg, p_cache) in passive_args
      if p_u_ode isa GravitySolverDealII
        @trixi_timeit timer() "passive solver" refine!(p_u_ode, adaptor, p_mesh,
                                                       p_equations, p_dg, lambda,
                                                       refined_original_cells)
      else
        @trixi_timeit timer() "passive solver" refine!(p_u_ode, adaptor, p_mesh,
                                                       p_equations, p_dg, p_cache,
                                                       refined_original_cells)
      end
    end
  else
    # If there is nothing to refine, create empty array for later use
    refined_original_cells = Int[]
  end

  global_cell_amount(mesh, "inbetween")
  @trixi_timeit timer() "coarsen" if !only_refine
    # Coarsen mesh
    coarsened_original_cells = @trixi_timeit timer() "mesh" coarsen!(mesh)

    # coarsen solver
    @trixi_timeit timer() "solver" coarsen!(u_ode, adaptor, mesh, equations, dg,
                                            cache, coarsened_original_cells)
    for (p_u_ode, p_mesh, p_equations, p_dg, p_cache) in passive_args
      if p_u_ode isa GravitySolverDealII
        if !p_equations[] # !dealii_amr_was_called[]
          @trixi_timeit timer() "passive solver" begin
            store_amr_indicator!(gravity_solver.dealii, lambda)
            dealii_adapt_grid(gravity_solver.dealii)
          end
        end
      else
        @trixi_timeit timer() "passive solver" coarsen!(p_u_ode, adaptor, p_mesh,
                                                        p_equations, p_dg, p_cache,
                                                        coarsened_original_cells)
      end
    end
  else
    # If there is nothing to coarsen, create empty array for later use
    coarsened_original_cells = Int[]
  end
  global_cell_amount(mesh, "post-all")

  # Store whether there were any cells coarsened or refined and perform load balancing
  has_changed = !isempty(refined_original_cells) || !isempty(coarsened_original_cells)
  # Check if mesh changed on other processes
  if mpi_isparallel()
    has_changed = MPI.Allreduce!(Ref(has_changed), |, mpi_comm())[]
  end

  if has_changed # TODO: Taal decide, where shall we set this?
    # don't set it to has_changed since there can be changes from earlier calls
    mesh.unsaved_changes = true

    if mpi_isparallel() && amr_callback.dynamic_load_balancing
      @trixi_timeit timer() "dynamic load balancing" begin
        global_first_quadrant = unsafe_wrap(Array, unsafe_load(mesh.p4est).global_first_quadrant, mpi_nranks() + 1)
        old_global_first_quadrant = copy(global_first_quadrant)
        partition!(mesh)
        rebalance_solver!(u_ode, mesh, equations, dg, cache, old_global_first_quadrant)
        # also rebalance the semidiscretisations in the passive args
        for (p_u_ode, p_mesh, p_equations, p_dg, p_cache) in passive_args
          if p_u_ode isa GravitySolverDealII
            # deal.ii rebalances already in `adapt_grid` if the appropriate flag is set
            # so we don't have to rebalance here
          else
            rebalance_solver!(p_u_ode, p_mesh, p_equations, p_dg, p_cache, old_global_first_quadrant)
          end
        end
      end
    end

    # TODO deal.II This needed to be commented to make it work
    # reinitialize_boundaries!(semi.boundary_conditions, cache)
  end

  # Return true if there were any cells coarsened or refined, otherwise false
  return has_changed
end

function global_cell_amount(mesh, position::String)
  num_cells = ncells(mesh)
  num_cells = MPI.Allreduce!(Ref(num_cells), +, mpi_comm())[]
end

function Trixi.refine!(gravity_solver::GravitySolverDealII,
                       _ #= adaptor =#, amr_callback #= mesh =#,
                       dealii_amr_was_called #= p_equations =#, _ #= p_dg =#,
                       indicators #= p_cache =#, _ #= refined_original_cells =#)
  dealii_amr_was_called[] = true

  store_amr_indicator!(gravity_solver.dealii, indicators)

  dealii_adapt_grid(gravity_solver.dealii)
end
