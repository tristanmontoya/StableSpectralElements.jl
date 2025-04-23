module Solvers

import LinearAlgebra
using StaticArrays
using MuladdMacro
using SparseArrays
using LinearAlgebra:
                     Diagonal,
                     eigvals,
                     inv,
                     mul!,
                     lmul!,
                     diag,
                     diagm,
                     factorize,
                     cholesky,
                     ldiv!,
                     Factorization,
                     Cholesky,
                     Symmetric,
                     I,
                     UniformScaling
using TimerOutputs
using LinearMaps: LinearMap, UniformScalingMap, TransposeMap
using OrdinaryDiffEq: ODEProblem, OrdinaryDiffEqAlgorithm, solve
using StartUpDG: num_faces

using ..MatrixFreeOperators
using ..ConservationLaws
using ..SpatialDiscretizations
using ..GridFunctions

export AbstractResidualForm,
       StandardForm,
       FluxDifferencingForm,
       AbstractMappingForm,
       AbstractStrategy,
       AbstractDiscretizationOperators,
       AbstractMassMatrixSolver,
       AbstractParallelism,
       ReferenceOperators,
       PhysicalOperators,
       FluxDifferencingOperators,
       AbstractPreAllocatedArrays,
       PreAllocatedArraysFirstOrder,
       PreAllocatedArraysSecondOrder,
       PhysicalOperator,
       ReferenceOperator,
       Solver,
       StandardMapping,
       SkewSymmetricMapping,
       Serial,
       Threaded,
       get_dof,
       semi_discrete_residual!,
       auxiliary_variable!,
       make_operators,
       entropy_projection!,
       facet_correction!,
       nodal_values!,
       time_derivative!,
       project_function!,
       flux_differencing_operators,
       initialize,
       semidiscretize

abstract type AbstractResidualForm end
abstract type AbstractMappingForm end
abstract type AbstractStrategy end
abstract type AbstractDiscretizationOperators end
abstract type AbstractMassMatrixSolver end
abstract type AbstractParallelism end
abstract type AbstractPreAllocatedArrays end

struct StandardMapping <: AbstractMappingForm end
struct SkewSymmetricMapping <: AbstractMappingForm end
struct PhysicalOperator <: AbstractStrategy end
struct ReferenceOperator <: AbstractStrategy end
struct Serial <: AbstractParallelism end
struct Threaded <: AbstractParallelism end

@doc raw"""
    StandardForm(; mapping_form = SkewSymmetricMapping(),
                   inviscid_numerical_flux = LaxFriedrichsNumericalFlux(),
                   viscous_numerical_flux = BR1())

Type used for dispatch indicating the use of a standard (i.e. not flux-differencing) weak-form discontinuous spectral-element method. The `mapping_form` argument can be set to 
`StandardMapping()` to recover a standard discontinuous Galerkin formulation.
"""
Base.@kwdef struct StandardForm{MappingForm, InviscidNumericalFlux, ViscousNumericalFlux} <:
                   AbstractResidualForm
    mapping_form::MappingForm = SkewSymmetricMapping()
    inviscid_numerical_flux::InviscidNumericalFlux = LaxFriedrichsNumericalFlux()
    viscous_numerical_flux::ViscousNumericalFlux = BR1()
end

@doc raw"""
    FluxDifferencingForm(; mapping_form = SkewSymmetricMapping(),
                   inviscid_numerical_flux = LaxFriedrichsNumericalFlux(),
                   viscous_numerical_flux = BR1(),
                   two_point_flux = EntropyConservativeFlux())

Type used for dispatch indicating the use of a flux-differencing (e.g. entropy-stable or 
kinetic-energy-preserving) discontinuous spectral-element formulation based on two-point
flux functions.
"""
Base.@kwdef struct FluxDifferencingForm{MappingForm,
    InviscidNumericalFlux,
    ViscousNumericalFlux,
    TwoPointFlux} <: AbstractResidualForm
    mapping_form::MappingForm = SkewSymmetricMapping()
    inviscid_numerical_flux::InviscidNumericalFlux = LaxFriedrichsNumericalFlux()
    viscous_numerical_flux::ViscousNumericalFlux = BR1()
    two_point_flux::TwoPointFlux = EntropyConservativeFlux()
end

@doc raw"""
    ReferenceOperators{D_type, Dt_type, V_type, Vt_type, R_type, Rt_type} <:
       AbstractDiscretizationOperators

Set of operators and nodal values of geometric factors used to evaluate the
semi-discrete residual for the [`StandardForm`](@ref) in reference space.
"""
struct ReferenceOperators{D_type, Dt_type, V_type, Vt_type, R_type, Rt_type} <:
       AbstractDiscretizationOperators
    D::D_type
    Dᵀ::Dt_type
    V::V_type
    Vᵀ::Vt_type
    R::R_type
    Rᵀ::Rt_type
    W::Diagonal{Float64, Vector{Float64}}
    B::Diagonal{Float64, Vector{Float64}}
    halfWΛ::Array{Diagonal{Float64, Vector{Float64}}, 3} # d x d x N_e
    halfN::Matrix{Diagonal{Float64, Vector{Float64}}} # d x N_e
    BJf::Vector{Diagonal{Float64, Vector{Float64}}} # N_e
    n_f::Array{Float64, 3} # d x N_f x N_e
end

@doc raw"""
    PhysicalOperators{d, VOL_type, FAC_type, V_type, R_type} <:
       AbstractDiscretizationOperators

Set of operators used to evaluate the semi-discrete residual for the [`StandardForm`](@ref)
by precomputing matrices for each physical element.
"""
struct PhysicalOperators{d, VOL_type, FAC_type, V_type, R_type} <:
       AbstractDiscretizationOperators
    VOL::Vector{NTuple{d, VOL_type}}
    FAC::Vector{FAC_type}
    V::V_type
    R::R_type
    n_f::Array{Float64, 3} # d x N_f x N_e
end

@doc raw"""
    FluxDifferencingOperators{S_type, C_type, V_type, Vt_type, R_type, Rt_type} <:
       AbstractDiscretizationOperators

Set of operators and nodal values of geometric factors used to evaluate the semi-discrete 
residual for the [`FluxDifferencingForm`](@ref) in reference space. Note that no 
physical-operator formulation is implemented for the flux-differencing form.
"""
struct FluxDifferencingOperators{S_type, C_type, V_type, Vt_type, R_type, Rt_type} <:
       AbstractDiscretizationOperators
    S::S_type
    C::C_type
    V::V_type
    Vᵀ::Vt_type
    R::R_type
    Rᵀ::Rt_type
    W::Diagonal{Float64, Vector{Float64}}
    B::Diagonal{Float64, Vector{Float64}}
    WJ::Vector{Diagonal{Float64, Vector{Float64}}}
    Λ_q::Array{Float64, 4} # N_q x d x d x N_e
    BJf::Vector{Diagonal{Float64, Vector{Float64}}} # N_e
    n_f::Array{Float64, 3} # d x N_f x N_e
    halfnJf::Array{Float64, 3} # d x N_f x N_e
    halfnJq::Array{Float64, 4} # d x N_q x num_faces x N_e
    nodes_per_face::Int64
end

struct PreAllocatedArraysFirstOrder <: AbstractPreAllocatedArrays
    f_q::Array{Float64, 4}
    f_f::Array{Float64, 3}
    f_n::Array{Float64, 3}
    u_q::Array{Float64, 3}
    r_q::Array{Float64, 3}
    u_f::Array{Float64, 3}
    temp::Array{Float64, 3}
    CI::CartesianIndices{2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}

    function PreAllocatedArraysFirstOrder(d,
            N_q,
            N_f,
            N_c,
            N_e,
            temp_size = N_q,
            N_r = Threads.nthreads())
        return new(Array{Float64}(undef, N_q, N_c, d, N_r),
            Array{Float64}(undef, N_f, N_c, N_r),
            Array{Float64}(undef, N_f, N_c, N_r),
            Array{Float64}(undef, N_q, N_c, N_e),
            Array{Float64}(undef, N_q, N_c, N_r),
            Array{Float64}(undef, N_f, N_e, N_c), #note switched order
            Array{Float64}(undef, temp_size, N_c, N_r),
            CartesianIndices((N_f, N_e)))
    end
end

struct PreAllocatedArraysSecondOrder <: AbstractPreAllocatedArrays
    f_q::Array{Float64, 4}
    f_f::Array{Float64, 3}
    f_n::Array{Float64, 3}
    u_q::Array{Float64, 3}
    r_q::Array{Float64, 3}
    u_f::Array{Float64, 3}
    temp::Array{Float64, 3}
    u_n::Array{Float64, 4}
    q_q::Array{Float64, 4}
    q_f::Array{Float64, 4}
    CI::CartesianIndices{2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}

    function PreAllocatedArraysSecondOrder(d,
            N_q,
            N_f,
            N_c,
            N_e,
            temp_size = N_q,
            N_r = Threads.nthreads())
        return new(Array{Float64}(undef, N_q, N_c, d, N_r),
            Array{Float64}(undef, N_f, N_c, N_r),
            Array{Float64}(undef, N_f, N_c, N_r),
            Array{Float64}(undef, N_q, N_c, N_e),
            Array{Float64}(undef, N_q, N_c, N_r),
            Array{Float64}(undef, N_f, N_e, N_c), #note switched order
            Array{Float64}(undef, temp_size, N_c, N_r),
            Array{Float64}(undef, N_f, N_c, d, N_r),
            Array{Float64}(undef, N_q, N_c, d, N_e),
            Array{Float64}(undef, N_f, N_e, N_c, d), #note switched order
            CartesianIndices((N_f, N_e)))
    end
end

@doc raw"""
    Solver{ConservationLaw <: AbstractConservationLaw, 
           Operators <: AbstractDiscretizationOperators,
           MassSolver <: AbstractMassMatrixSolver,
           ResidualForm <: AbstractMassMatrixSolver,
           Parallelism <: AbstractParallelism,
           PreAllocatedArrays <: AbstractPreAllocatedArrays}

Composite type defining the spatial discretization which is passed into
[`semi_discrete_residual!`](@ref) function to compute the time derivative within an
ordinary differential equation solver from 
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl). The algorithm used to
compute the semi-discrete residual is dispatched based on the type parameters.
"""
struct Solver{ConservationLaw,
    Operators,
    MassSolver,
    ResidualForm,
    Parallelism,
    PreAllocatedArrays}
    conservation_law::ConservationLaw
    operators::Operators
    mass_solver::MassSolver
    connectivity::Matrix{Int}
    form::ResidualForm
    parallelism::Parallelism
    preallocated_arrays::PreAllocatedArrays
end

# Gets tuple containing the number of nodal/modal coefficients, number of conservative
# variables, and number of mesh elements from the solver
function Base.size(solver::Solver{
        <:AbstractConservationLaw{d, <:AbstractPDEType, N_c},
}) where {
        d,
        N_c
}
    N_p = size(solver.operators.V, 2)
    N_e = size(solver.preallocated_arrays.CI, 2)
    return (N_p, N_c, N_e)
end

function Solver(conservation_law::AbstractConservationLaw{d, FirstOrder, N_c},
        spatial_discretization::SpatialDiscretization{d},
        form::StandardForm,
        ::ReferenceOperator,
        alg::AbstractOperatorAlgorithm,
        mass_solver::AbstractMassMatrixSolver,
        parallelism::AbstractParallelism) where {d, N_c}
    (; reference_approximation, geometric_factors, N_e, mesh) = spatial_discretization
    (; N_q, N_f, reference_mapping) = reference_approximation

    lumped_geometric_factors = apply_reference_mapping(geometric_factors, reference_mapping)
    (; Λ_q, nJf, J_f) = lumped_geometric_factors

    operators = ReferenceOperators(reference_approximation, alg, Λ_q, nJf, J_f)

    return Solver(conservation_law,
        operators,
        mass_solver,
        mesh.mapP,
        form,
        parallelism,
        PreAllocatedArraysFirstOrder(d, N_q, N_f, N_c, N_e, N_q))
end

function Solver(conservation_law::AbstractConservationLaw{d, FirstOrder, N_c},
        spatial_discretization::SpatialDiscretization{d},
        form::FluxDifferencingForm,
        ::ReferenceOperator,
        alg::AbstractOperatorAlgorithm,
        mass_solver::AbstractMassMatrixSolver,
        parallelism::AbstractParallelism) where {d, N_c}
    (; reference_approximation, N_e, mesh) = spatial_discretization
    (; N_p, N_q, N_f) = reference_approximation
    (; J_q, Λ_q, nJf, nJq, J_f) = spatial_discretization.geometric_factors
    (; reference_approximation, N_e, mesh) = spatial_discretization

    operators = FluxDifferencingOperators(reference_approximation, alg, J_q, Λ_q, nJq, nJf,
        J_f)

    return Solver(conservation_law,
        operators,
        mass_solver,
        mesh.mapP,
        form,
        parallelism,
        PreAllocatedArraysFirstOrder(d, N_q, N_f, N_c, N_e, N_p))
end

function Solver(conservation_law::AbstractConservationLaw{d, FirstOrder, N_c},
        spatial_discretization::SpatialDiscretization{d},
        form::StandardForm,
        ::PhysicalOperator,
        alg::AbstractOperatorAlgorithm,
        mass_solver::AbstractMassMatrixSolver,
        parallelism::AbstractParallelism) where {d, N_c}
    (; N_e) = spatial_discretization
    (; N_p, N_q, N_f) = spatial_discretization.reference_approximation

    operators = PhysicalOperators(spatial_discretization, form, alg, mass_solver)

    return Solver(conservation_law,
        operators,
        mass_solver,
        spatial_discretization.mesh.mapP,
        form,
        parallelism,
        PreAllocatedArraysFirstOrder(d, N_q, N_f, N_c, N_e, N_p,
            Threads.nthreads()))
end

function Solver(conservation_law::AbstractConservationLaw{d, SecondOrder, N_c},
        spatial_discretization::SpatialDiscretization{d},
        form::StandardForm,
        ::AbstractStrategy,
        alg::AbstractOperatorAlgorithm,
        mass_solver::AbstractMassMatrixSolver,
        parallelism::AbstractParallelism) where {d, N_c}
    (; N_e) = spatial_discretization
    (; N_p, N_q, N_f) = spatial_discretization.reference_approximation

    operators = PhysicalOperators(spatial_discretization, form, alg, mass_solver)

    return Solver(conservation_law,
        operators,
        mass_solver,
        spatial_discretization.mesh.mapP,
        form,
        parallelism,
        PreAllocatedArraysSecondOrder(d, N_q, N_f, N_c, N_e, N_p))
end

# Gets tuple containing the number of nodal/modal coefficients, number of conservative
# variables, and number of mesh elements from the spatial discretization and 
# conservation law
@inline function get_dof(spatial_discretization::SpatialDiscretization{d},
        ::AbstractConservationLaw{d, PDEType, N_c}) where {d, PDEType, N_c}
    return (spatial_discretization.reference_approximation.N_p,
        N_c,
        spatial_discretization.N_e)
end

@inline function project_function(initial_data::AbstractGridFunction{d},
        ::UniformScalingMap,
        W::Diagonal,
        J_q::Matrix{Float64},
        x::NTuple{d, Matrix{Float64}}) where {d}
    return evaluate(initial_data, x, 0.0)
end

@inline function project_function(initial_data,
        V::LinearMap,
        W::Diagonal,
        J_q::Matrix{Float64},
        x::NTuple{d, Matrix{Float64}}) where {d}
    N_p = size(V, 2)
    N_e = size(J_q, 2)

    u_q = evaluate(initial_data, x, 0.0)
    u0 = Array{Float64}(undef, N_p, size(u_q, 2), N_e)
    VDM = Matrix(V)

    @inbounds @views for k in 1:N_e
        WJ = Diagonal(W .* J_q[:, k])

        # this will throw if M is not SPD
        M = cholesky(Symmetric(VDM' * WJ * VDM))
        lmul!(WJ, u_q[:, :, k])
        mul!(u0[:, :, k], VDM', u_q[:, :, k])
        ldiv!(M, u0[:, :, k])
    end
    return u0
end

# Returns an array of initial data as nodal or modal DOF
@inline function initialize(initial_data, spatial_discretization::SpatialDiscretization)
    (; J_q) = spatial_discretization.geometric_factors
    (; V, W) = spatial_discretization.reference_approximation
    (; xyzq) = spatial_discretization.mesh

    return project_function(initial_data, V, W, J_q, xyzq)
end

@inline function semidiscretize(conservation_law::AbstractConservationLaw{d, PDEType},
        spatial_discretization::SpatialDiscretization{d},
        initial_data,
        form::AbstractResidualForm,
        tspan::NTuple{2, Float64},
        strategy::AbstractStrategy = ReferenceOperator(),
        alg::AbstractOperatorAlgorithm = DefaultOperatorAlgorithm();
        mass_matrix_solver::AbstractMassMatrixSolver = default_mass_matrix_solver(
            spatial_discretization,
            alg),
        parallelism::AbstractParallelism = Threaded()) where {d,
        PDEType
}
    u0 = initialize(initial_data, spatial_discretization)

    solver = Solver(conservation_law,
        spatial_discretization,
        form,
        strategy,
        alg,
        mass_matrix_solver,
        parallelism)

    return ODEProblem(semi_discrete_residual!, u0, tspan, solver)
end
@doc raw"""
    semi_discrete_residual!(dudt::AbstractArray{Float64,3},
                           u::AbstractArray{Float64, 3},
                           solver::Solver,
                           t::Float64)

In-place function for evaluating the right-hand side of $\underline{u}'(t) = \underline{R}(\underline{u}(t),t)$ within the ordinary differential equation solver.
# Arguments
- `dudt::AbstractArray{Float64,3}`: Time derivative (first index is nodal or modal
  coefficient, second index is solution variable, third index is mesh element) 
- `u::AbstractArray{Float64,3}`: Current global solution state (first index is nodal or
  modal coefficient, second index is solution variable, third index is mesh element) 
- `solver::Solver`: Parameter of type [`Solver`](@ref) containing all the information    
  defining the spatial discretization, the algorithms used to evaluate the semi-discrete 
  residual, and preallocated arrays used for temporary storage; type parameters of Solver 
  are used to dispatch the resulting algorithm based on the equation type, operator type, 
  mass matrix solver (i.e. `CholeskySolver`, `DiagonalSolver`, or `WeightAdjustedSolver`),
  residual form (i.e. `StandardForm` or `FluxDifferencingForm`), and parallelism 
  (i.e. `Serial` or `Threaded`).
- `t::Float64`: Time variable
"""
@timeit "semi-disc. residual" function semi_discrete_residual!(
        dudt::AbstractArray{Float64,
            3},
        u::AbstractArray{Float64, 3},
        solver::Solver{<:AbstractConservationLaw{<:Any,
                FirstOrder},
            <:AbstractDiscretizationOperators,
            <:AbstractMassMatrixSolver,
            <:AbstractResidualForm,
            Serial},
        t::Float64 = 0.0)
    @inbounds for k in axes(u, 3)
        @timeit "nodal values" nodal_values!(u, solver, k)
    end

    @inbounds for k in axes(u, 3)
        @timeit "time deriv." time_derivative!(dudt, solver, k)
    end

    return dudt
end

@timeit "semi-disc. residual" function semi_discrete_residual!(
        dudt::AbstractArray{Float64,
            3},
        u::AbstractArray{Float64, 3},
        solver::Solver{<:AbstractConservationLaw{<:Any,
                FirstOrder},
            <:AbstractDiscretizationOperators,
            <:AbstractMassMatrixSolver,
            <:AbstractResidualForm,
            Threaded},
        t::Float64 = 0.0)
    Threads.@threads for k in axes(u, 3)
        nodal_values!(u, solver, k)
    end

    Threads.@threads for k in axes(u, 3)
        time_derivative!(dudt, solver, k)
    end

    return dudt
end

@timeit "semi-disc. residual" function semi_discrete_residual!(
        dudt::AbstractArray{Float64,
            3},
        u::AbstractArray{Float64, 3},
        solver::Solver{<:AbstractConservationLaw{<:Any,
                SecondOrder},
            <:AbstractDiscretizationOperators,
            <:AbstractMassMatrixSolver,
            <:AbstractResidualForm,
            Serial},
        t::Float64 = 0.0)
    @inbounds for k in axes(u, 3)
        nodal_values!(u, solver, k)
    end

    @inbounds for k in axes(u, 3)
        auxiliary_variable!(dudt, solver, k)
    end

    @inbounds for k in axes(u, 3)
        time_derivative!(dudt, solver, k)
    end

    return dudt
end

@timeit "semi-disc. residual" function semi_discrete_residual!(
        dudt::AbstractArray{Float64,
            3},
        u::AbstractArray{Float64, 3},
        solver::Solver{<:AbstractConservationLaw{<:Any,
                SecondOrder},
            <:AbstractDiscretizationOperators,
            <:AbstractMassMatrixSolver,
            <:AbstractResidualForm,
            Threaded},
        t::Float64 = 0.0)
    @inbounds Threads.@threads for k in axes(u, 3)
        nodal_values!(u, solver, k)
    end

    @inbounds Threads.@threads for k in axes(u, 3)
        auxiliary_variable!(dudt, solver, k)
    end

    @inbounds Threads.@threads for k in axes(u, 3)
        time_derivative!(dudt, solver, k)
    end

    return dudt
end

export CholeskySolver,
       WeightAdjustedSolver,
       DiagonalSolver,
       mass_matrix,
       mass_matrix_inverse,
       mass_matrix_solve!,
       default_mass_matrix_solver
include("mass_matrix.jl")

include("operators.jl")
include("standard_form_first_order.jl")
include("standard_form_second_order.jl")
include("flux_differencing_form.jl")

export LinearResidual
include("linear.jl")
end
