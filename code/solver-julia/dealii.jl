using libdealii_trixi_paper2026_jll

"""
Contains the mesh settings for p4est to initialise it synced in deal.II

As for the P4estMesh constructors: use only one of faces, mapping, and
coordinates_{min,max}!
"""
struct MeshSettings2D
   repartition_after_adapt :: Cuchar
   from_file :: Cuchar
   # iff  from file
   meshfile :: Cstring
   boundary_symbols :: Ptr{Cvoid}
   # iff !from_file
   trees_per_dimension :: NTuple{2,Cint}
   faces :: Ptr{Cvoid}
   coordinates_min :: NTuple{2,Cdouble}
   coordinates_max :: NTuple{2,Cdouble}
   periodicity :: NTuple{2,Cuchar}
   # common
   polydeg :: Cint
   mapping :: Ptr{Cvoid} # 2*dim functions mapping the faces
   initial_refinement_level :: Cint
   unsaved_changes :: Cuchar
   p4est_partition_allow_for_coarsening :: Cuchar

   function MeshSettings2D(repartition_after_adapt::Bool,
                           trees_per_dimension; polydeg=1,
                           mapping=nothing, faces=nothing, coordinates_min=nothing,
                           coordinates_max=nothing, initial_refinement_level=0,
                           periodicity=true, unsaved_changes=true,
                           p4est_partition_allow_for_coarsening=true)
           tpd = NTuple{2,Cint}(Cint.(trees_per_dimension))
           if (coordinates_min != nothing && coordinates_max != nothing)
             coord_min = NTuple{2,Cdouble}(Cdouble.(coordinates_min))
             coord_max = NTuple{2,Cdouble}(Cdouble.(coordinates_max))
           else
           end
           if (mapping != nothing)
           else
             cmapping = C_NULL
           end
           if (faces != nothing)
           else
             cfaces = C_NULL
           end
           if periodicity isa Bool
             cp = Cuchar(periodicity)
             cperiodicity = (cp,cp)
           else
             cperiodicity = NTuple{2,Cuchar}(Cuchar.(periodicity))
           end
           return new(Cuchar(repartition_after_adapt), Cuchar(false), C_NULL, C_NULL, tpd,
                      cfaces, coord_min, coord_max, cperiodicity, Cint(polydeg), cmapping,
                      Cint(initial_refinement_level), Cuchar(unsaved_changes),
                      Cuchar(p4est_partition_allow_for_coarsening))
   end

   function MeshSettings2D(repartition_after_adapt::Bool,meshfile::String;
                           mapping=nothing, polydeg=1,
                           initial_refinement_level=0, unsaved_changes=true,
                           p4est_partition_allow_for_coarsening=true,
                           boundary_symbols=nothing)
       # boundary_symbols contains the boundary names in the meshfile, if it is nothing
       # then all boundaries are named :all
       return new(Cuchar(repartition_after_adapt), Cuchar(true), meshfile,
                  boundary_symbols,
                  C_NULL, C_NULL, C_NULL, C_NULL, (false,false),
                  polydeg, mapping, initial_refinement_level, unsaved_changes,
                  p4est_partition_allow_for_coarsening)
   end
end

struct MeshSettings3D
   repartition_after_adapt :: Cuchar
   from_file :: Cuchar
   # iff  from file
   meshfile :: Cstring
   boundary_symbols :: Ptr{Cvoid}
   # iff !from_file
   trees_per_dimension :: NTuple{3,Cint}
   faces :: Ptr{Cvoid}
   coordinates_min :: NTuple{3,Cdouble}
   coordinates_max :: NTuple{3,Cdouble}
   periodicity :: NTuple{3,Cuchar}
   # common
   polydeg :: Cint
   mapping :: Ptr{Cvoid} # 2*dim functions mapping the faces
   initial_refinement_level :: Cint
   unsaved_changes :: Cuchar
   p4est_partition_allow_for_coarsening :: Cuchar

   function MeshSettings3D(repartition_after_adapt::Bool,
                           trees_per_dimension; polydeg,
                           mapping=nothing, faces=nothing, coordinates_min=nothing,
                           coordinates_max=nothing, initial_refinement_level=0,
                           periodicity=true, unsaved_changes=true,
                           p4est_partition_allow_for_coarsening=true)
           tpd = NTuple{3,Cint}(Cint.(trees_per_dimension))
           if (coordinates_min != nothing && coordinates_max != nothing)
             coord_min = NTuple{3,Cdouble}(Cdouble.(coordinates_min))
             coord_max = NTuple{3,Cdouble}(Cdouble.(coordinates_max))
           else
           end
           if (mapping != nothing)
           else
             cmapping = C_NULL
           end
           if (faces != nothing)
           else
             cfaces = C_NULL
           end
           if periodicity isa Bool
             cp = Cuchar(periodicity)
             cperiodicity = (cp,cp,cp)
           else
             cperiodicity = NTuple{3,Cuchar}(Cuchar.(periodicity))
           end
           return new(Cuchar(repartition_after_adapt), Cuchar(false), C_NULL, C_NULL, tpd,
                      cfaces, coord_min, coord_max, cperiodicity, Cint(polydeg), cmapping,
                      Cint(initial_refinement_level), Cuchar(unsaved_changes),
                      Cuchar(p4est_partition_allow_for_coarsening))
   end

   function MeshSettings3D(repartition_after_adapt::Bool,meshfile::String;
                           mapping=nothing, polydeg=1,
                           initial_refinement_level=0, unsaved_changes=true,
                           p4est_partition_allow_for_coarsening=true,
                           boundary_symbols=nothing)
       # boundary_symbols contains the boundary names in the meshfile, if it is nothing
       # then all boundaries are named :all
       return new(Cuchar(repartition_after_adapt), Cuchar(true), meshfile,
                  boundary_symbols,
                  C_NULL, C_NULL, C_NULL, C_NULL, (false,false,false),
                  polydeg, mapping, initial_refinement_level, unsaved_changes,
                  p4est_partition_allow_for_coarsening)
   end
end

# Wrapper for deal.II pointer for easier dispatch etc.
# Note: `mutable` to support finalizer
abstract type DealII
end

mutable struct DealII_3D <: DealII
  const handle::Ptr{Cvoid}

  # Internal constructor will register the deal.II wrapper for automatic garbage collection
  function DealII_3D(handle)
    dealii = new(handle)
    function destroy!(dealii)
      @ccall libdealii_trixi_paper2026.finalize(dealii::Ptr{Cvoid})::Cvoid
    end
    finalizer(destroy!, dealii)
    return dealii
  end
end

mutable struct DealII_2D <: DealII
  const handle::Ptr{Cvoid}

  # Internal constructor will register the deal.II wrapper for automatic garbage collection
  function DealII_2D(handle)
    dealii = new(handle)
    function destroy!(dealii)
      @ccall libdealii_trixi_paper2026.finalize(dealii::Ptr{Cvoid})::Cvoid
    end
    finalizer(destroy!, dealii)
    return dealii
  end
end

# Allow using `::DealII` in `@ccall`
Base.unsafe_convert(::Type{Ptr{Cvoid}}, dealii::DealII) = dealii.handle

# Initialize libraries handler for deal.II
function dealii_init_libs(args::AbstractVector{<:String})
  handle = @ccall libdealii_trixi_paper2026.init_libs(length(args)::Cint,
                                      args::Ptr{Ptr{Cchar}})::Ptr{Cvoid}
  return handle
end

# Finalize libraries handler for deal.II
function dealii_finalize_libs(handle::Ptr{Cvoid})
  @ccall libdealii_trixi_paper2026.finalize_libs(handle::Ptr{Cvoid})::Cvoid
  return nothing
end

# Initialize deal.II object
function dealii_init(polydeg :: Int,
                     meshsettings :: MeshSettings2D;
                     maxiters = 30,
                     abstol = 1.0e-8)
  handle = @ccall libdealii_trixi_paper2026.init_2d(polydeg::Cint,
                                    meshsettings::MeshSettings2D,
                                    maxiters::Cint,
                                    abstol::Cdouble)::Ptr{Cvoid}
  return DealII_2D(handle)
end

function dealii_init(polydeg :: Int,
                     meshsettings :: MeshSettings3D;
                     maxiters = 30,
                     abstol = 1.0e-8)
  handle = @ccall libdealii_trixi_paper2026.init_3d(polydeg::Cint,
                                    meshsettings::MeshSettings3D,
                                    maxiters::Cint,
                                    abstol::Cdouble)::Ptr{Cvoid}
  return DealII_3D(handle)
end

function dealii_set_maxiters(dealii, maxiters)
  @ccall libdealii_trixi_paper2026.set_maxiters(dealii::Ptr{Cvoid}, maxiters::Cint)::Cvoid
end

function dealii_set_abstol(dealii, abstol)
  @ccall libdealii_trixi_paper2026.set_abstol(dealii::Ptr{Cvoid}, abstol::Cdouble)::Cvoid
end

# Get pointer to source and solution vectors
function dealii_get_pointer_f(dealii)
  pointer = @ccall libdealii_trixi_paper2026.get_pointer_f(dealii::Ptr{Cvoid})::Ptr{Cdouble}
end

function dealii_get_pointer_u(dealii)
  pointer = @ccall libdealii_trixi_paper2026.get_pointer_u(dealii::Ptr{Cvoid})::Ptr{Cdouble}
end

function dealii_get_pointer_grad_u(dealii)
  pointer = @ccall libdealii_trixi_paper2026.get_pointer_grad_u(dealii::Ptr{Cvoid})::Ptr{Ptr{Cdouble}}
end

function dealii_get_pointer_amr_indicator(dealii)
  pointer = @ccall libdealii_trixi_paper2026.get_pointer_amr_indicator(dealii::Ptr{Cvoid})::Ptr{Cint}
end

# Setup to get a pointer to allocated memory in C/C++ and write the values
# of one of our solution vectors into it
function store_values!(p::Ptr, src::AbstractVector)
  dst = unsafe_wrap(Array, p, length(src))
  copyto!(dst, src)
  return nothing
end

function store_u!(dealii, u::AbstractVector{Float64})
  dst = dealii_get_pointer_u(dealii)
  store_values!(dst, u)
  return nothing
end

function store_f!(dealii, f::AbstractVector{Float64})
  dst = dealii_get_pointer_f(dealii)
  store_values!(dst, f)
  return nothing
end

"""
Stores the values of an array in the space reserved for AMR indicators for the cells
of the mesh."""
function store_amr_indicator!(dealii, amr_indicator::AbstractVector)
  dst = dealii_get_pointer_amr_indicator(dealii)
  store_values!(dst,  trunc.(Int32, amr_indicator))
  return nothing
end

"""
Loads values from a pointer to a vector on the C++ side into a Julia vector."""
function load_values!(dst::AbstractVector, p::Ptr)
  src = unsafe_wrap(Array, p, length(dst))
  copyto!(dst, src)
  return nothing
end

function load_u!(u::AbstractVector{Float64}, dealii)
  src = dealii_get_pointer_u(dealii)
  load_values!(u, src)
  return nothing
end

"""
Loads the gradient from the deal.II side."""
function load_grad_u!(grad_u::NTuple{N, AbstractVector{Float64}}, dealii) where {N}
  src = dealii_get_pointer_grad_u(dealii)
  for i in 1:N
    load_values!(grad_u[i], unsafe_load(src, i))
  end
  return nothing
end

"""
Solve Poisson problem."""
function dealii_solve(dealii)
  @ccall libdealii_trixi_paper2026.solve(dealii::Ptr{Cvoid})::Cvoid
end

"""
Adapt mesh in deal.II."""
function dealii_adapt_grid(dealii)
  @ccall libdealii_trixi_paper2026.adapt_grid(dealii::Ptr{Cvoid})::Cvoid
end

"""
Repartition mesh"""
function dealii_repartition(dealii)
  @ccall libdealii_trixi_paper2026.repartition_grid(dealii::Ptr{Cvoid})::Cvoid
end

"""
Gets the amount of currently locally owned active cells (per MPI rank, total amount
of globally active cells is the sum of the return values of all MPI ranks)."""
function dealii_get_num_local_active_cells(dealii)
  @ccall libdealii_trixi_paper2026.get_num_local_active_cells(dealii::Ptr{Cvoid})::Cuint
end

# Get checksum for mesh
function dealii_get_mesh_checksum(dealii::DealII)
  @ccall libdealii_trixi_paper2026.get_mesh_checksum(dealii::Ptr{Cvoid})::Cuint
end
