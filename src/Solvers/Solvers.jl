module Solvers
    
    import LinearAlgebra
    using GFlops
    using StaticArrays
    using LinearAlgebra: Diagonal, eigvals, inv, mul!, lmul!, diag, diagm, factorize, cholesky, ldiv!, Factorization, Cholesky, Symmetric, I, UniformScaling
    using TimerOutputs
    using LinearMaps: LinearMap, UniformScalingMap
    using OrdinaryDiffEq: ODEProblem, OrdinaryDiffEqAlgorithm, solve
    
    using ..MatrixFreeOperators: AbstractOperatorAlgorithm,  DefaultOperatorAlgorithm, make_operator
    using ..ConservationLaws: AbstractConservationLaw, AbstractPDEType, FirstOrder, SecondOrder, AbstractInviscidNumericalFlux, AbstractViscousNumericalFlux, AbstractTwoPointFlux, NoInviscidFlux, NoViscousFlux, NoTwoPointFlux, NoSourceTerm, physical_flux!, numerical_flux!, compute_two_point_flux, LaxFriedrichsNumericalFlux, BR1, EntropyConservativeFlux
    using ..SpatialDiscretizations: ReferenceApproximation, SpatialDiscretization, check_facet_nodes, check_normals
    using ..GridFunctions: AbstractGridFunction, AbstractGridFunction, NoSourceTerm, evaluate
    
    export AbstractResidualForm, StandardForm, FluxDifferencingForm, AbstractMappingForm, AbstractStrategy, AbstractDiscretizationOperators, AbstractPreallocatedArrays, AbstractMassMatrixSolver, PhysicalOperators, FluxDifferencingOperators, PreAllocatedArrays,  PreAllocatedArraysFluxDifferencing, PhysicalOperator, ReferenceOperator, Solver, StandardMapping, SkewSymmetricMapping, get_dof, rhs!, rhs_static!, make_operators, StandardForm

    abstract type AbstractResidualForm{MappingForm, TwoPointFlux} end
    abstract type AbstractMappingForm end
    abstract type AbstractStrategy end
    abstract type AbstractDiscretizationOperators{d} end
    abstract type AbstractPreallocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e} end
    abstract type AbstractMassMatrixSolver end

    struct StandardMapping <: AbstractMappingForm end
    struct SkewSymmetricMapping <: AbstractMappingForm end

    struct PhysicalOperator <: AbstractStrategy end
    struct ReferenceOperator <: AbstractStrategy end

    Base.@kwdef struct StandardForm{MappingForm} <: AbstractResidualForm{MappingForm,NoTwoPointFlux}
        mapping_form::MappingForm = SkewSymmetricMapping()
        inviscid_numerical_flux::AbstractInviscidNumericalFlux =
            LaxFriedrichsNumericalFlux()
        viscous_numerical_flux::AbstractViscousNumericalFlux = BR1()
    end

    Base.@kwdef struct FluxDifferencingForm{MappingForm,TwoPointFlux} <: AbstractResidualForm{MappingForm,TwoPointFlux}
        mapping_form::MappingForm = StandardMapping()
        inviscid_numerical_flux::AbstractInviscidNumericalFlux =
            LaxFriedrichsNumericalFlux()
        viscous_numerical_flux::AbstractViscousNumericalFlux = BR1()
        two_point_flux::TwoPointFlux = EntropyConservativeFlux()
        facet_correction::Bool = false
    end

    struct ReferenceOperators{d} <: AbstractDiscretizationOperators{d}
        D::NTuple{d,LinearMap}
        V::LinearMap
        R::LinearMap
        W::Diagonal
        B::Diagonal
        halfWΛ::Array{Diagonal,3} # d x d x N_e
        halfN::Matrix{Diagonal}
        BJf::Vector{Diagonal}
        n_f::Vector{NTuple{d, Vector{Float64}}}
    end

    struct PhysicalOperators{d} <: AbstractDiscretizationOperators{d}
        VOL::Vector{NTuple{d,LinearMap}}
        FAC::Vector{LinearMap}
        V::Vector{LinearMap}
        R::Vector{LinearMap}
        n_f::Vector{NTuple{d, Vector{Float64}}}
    end

    struct FluxDifferencingOperators{d} <: AbstractDiscretizationOperators{d}
        S::NTuple{d,LinearMap}
        V::LinearMap
        R::LinearMap
        W::Diagonal
        B::Diagonal
        Λ_q::Array{Float64,4} # N_q x d x d x N_e
        BJf::Vector{Diagonal}
        n_f::Vector{NTuple{d, Vector{Float64}}}
    end

    struct PreAllocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e} <: AbstractPreallocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e}
        f_q::Array{Float64,4}
        f_f::Array{Float64,3}
        f_n::Array{Float64,3}
        u_q::Array{Float64,3}
        r_q::Array{Float64,3}
        u_f::Array{Float64,3}
        temp::Array{Float64,3}
        CI::CartesianIndices
        u_n::Union{Nothing,Array{Float64,4}}
        q_q::Union{Nothing,Array{Float64,4}}
        q_f::Union{Nothing,Array{Float64,4}}
    end

    struct Solver{d,ResidualForm,PDEType,OperatorType,N_p,N_q,N_f,N_c,N_e}
        conservation_law::AbstractConservationLaw{d,PDEType}
        operators::OperatorType
        mass_solver::AbstractMassMatrixSolver
        connectivity::Matrix{Int}
        form::ResidualForm
        preallocated_arrays::AbstractPreallocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e}
    end

    function PreAllocatedArrays{d,FirstOrder,N_p,N_q,N_f,N_c,N_e}() where {d,N_p,N_q,N_f,N_c,N_e}
        return PreAllocatedArrays{d,FirstOrder,N_p,N_q,N_f,N_c,N_e}(
            Array{Float64}(undef,N_q, N_c, d, N_e),
            Array{Float64}(undef,N_f, N_c, N_e),
            Array{Float64}(undef,N_f, N_c, N_e),
            Array{Float64}(undef,N_q, N_c, N_e),
            Array{Float64}(undef,N_q, N_c, N_e),
            Array{Float64}(undef,N_f, N_e, N_c), #note switched order
            Array{Float64}(undef,N_q, N_c, N_e),
            CartesianIndices((N_f,N_e)), nothing, nothing, nothing)
    end

    function PreAllocatedArrays{d,SecondOrder,N_p,N_q,N_f,N_c,N_e}() where {d,N_p,N_q,N_f,N_c,N_e}
        return PreAllocatedArrays{d,SecondOrder,N_p,N_q,N_f,N_c,N_e}(
            Array{Float64}(undef,N_q, N_c, d, N_e),
            Array{Float64}(undef,N_f, N_c, N_e),
            Array{Float64}(undef,N_f, N_c, N_e),
            Array{Float64}(undef,N_q, N_c, N_e),
            Array{Float64}(undef,N_q, N_c, N_e),
            Array{Float64}(undef,N_f, N_e, N_c), #note switched order
            Array{Float64}(undef,N_q, N_c, N_e),
            CartesianIndices((N_f,N_e)),
            Array{Float64}(undef,N_f, N_c, d, N_e),
            Array{Float64}(undef,N_q, N_c, d, N_e),
            Array{Float64}(undef,N_f, N_e, N_c, d)) #note switched order
    end

    function Solver(conservation_law::AbstractConservationLaw{d,PDEType},     
        spatial_discretization::SpatialDiscretization{d},
        form::ResidualForm,
        ::PhysicalOperator,
        operator_algorithm::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm(),
        mass_solver::AbstractMassMatrixSolver=WeightAdjustedSolver(spatial_discretization)) where {d, PDEType, ResidualForm<:StandardForm}

        (; N_e) = spatial_discretization
        (; N_p, N_q, N_f) = spatial_discretization.reference_approximation
        (; N_c) = conservation_law

        operators = make_operators(spatial_discretization, form,
            operator_algorithm, mass_solver)
            
        return Solver{d,ResidualForm,PDEType,PhysicalOperators{d}, N_p, N_q, N_f, N_c, N_e}(
            conservation_law, operators, mass_solver, 
            spatial_discretization.mesh.mapP, form,
            PreAllocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e}())
    end
    
    function Solver(conservation_law::AbstractConservationLaw{d,PDEType},     
        spatial_discretization::SpatialDiscretization{d},
        form::ResidualForm,
        ::ReferenceOperator,
        alg::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm(),
        mass_solver::AbstractMassMatrixSolver=WeightAdjustedSolver(spatial_discretization)) where {d, PDEType, ResidualForm<:StandardForm}
    
        (; N_p, N_q, N_f, D, V, W, R, B) = spatial_discretization.reference_approximation
        (; Λ_q, nJf, J_f) = spatial_discretization.geometric_factors
        (; N_e) = spatial_discretization
        (; N_c) = conservation_law

        halfWΛ = Array{Diagonal,3}(undef, d, d, N_e)
        halfN = Matrix{Diagonal}(undef, d, N_e)
        BJf = Vector{Diagonal}(undef, N_e)
        n_f = Vector{NTuple{d, Vector{Float64}}}(undef,N_e)
    
        Threads.@threads for k in 1:N_e
            halfWΛ[:,:,k] = [Diagonal(0.5 * W * Λ_q[:,m,n,k]) 
                for m in 1:d, n in 1:d]
            halfN[:,k] = [Diagonal(0.5 * nJf[m][:,k] ./ J_f[:,k]) 
                for m in 1:d]
            BJf[k] = Diagonal(B .* J_f[:,k])
            n_f[k] = Tuple(nJf[m][:,k] ./ J_f[:,k] for m in 1:d)
        end
    
        operators = ReferenceOperators{d}(
            Tuple(make_operator(D[m], alg) for m in 1:d), 
            make_operator(V, alg), make_operator(R, alg), 
            W, B, halfWΛ, halfN, BJf, n_f)
    
        return Solver{d,ResidualForm,PDEType,ReferenceOperators{d}, N_p, N_q, N_f, N_c, N_e}(
            conservation_law, operators, mass_solver, 
            spatial_discretization.mesh.mapP, form,
            PreAllocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e}())
    end    

    function Solver(conservation_law::AbstractConservationLaw{d,PDEType},     
        spatial_discretization::SpatialDiscretization{d},
        form::ResidualForm,
        ::AbstractStrategy,
        alg::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm(),
        mass_solver::AbstractMassMatrixSolver=WeightAdjustedSolver(
            spatial_discretization)) where {d, PDEType,ResidualForm<:FluxDifferencingForm}
    
        (; N_p, N_q, N_f, D, V, W, R, B) = spatial_discretization.reference_approximation
        (; Λ_q, nJf, J_f) = spatial_discretization.geometric_factors
        (; N_e) = spatial_discretization
        (; N_c) = conservation_law
    
        BJf = Vector{Diagonal}(undef, N_e)
        n_f = Vector{NTuple{d, Vector{Float64}}}(undef,N_e)
    
        Threads.@threads for k in 1:N_e
            BJf[k] = Diagonal(B .* J_f[:,k])
            n_f[k] = Tuple(nJf[m][:,k] ./ J_f[:,k] for m in 1:d)
        end
    
        S = Tuple(make_operator(Matrix(W*D[m] - D[m]'*W), alg) for m in 1:d)

        operators = FluxDifferencingOperators{d}(S, 
            make_operator(V, alg), make_operator(R, alg), 
            W, B, Λ_q, BJf, n_f)
    
        return Solver{d,ResidualForm,PDEType,FluxDifferencingOperators{d},N_p,N_q,N_f, N_c, N_e}(conservation_law, operators, mass_solver,
            spatial_discretization.mesh.mapP,form, 
            PreAllocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e}())
    end    

    @inline function get_dof(spatial_discretization::SpatialDiscretization{d}, 
        conservation_law::AbstractConservationLaw{d}) where {d}
        return (spatial_discretization.reference_approximation.N_p, 
            conservation_law.N_c, spatial_discretization.N_e)
    end

    export CholeskySolver, WeightAdjustedSolver, DiagonalSolver, mass_matrix, mass_matrix_inverse, mass_matrix_solve!
    include("mass_matrix.jl") 

    export initialize, semidiscretize, precompute
    include("preprocessing.jl")

    include("make_operators.jl")
    
    export diff_with_extrap_flux!
    include("standard_form_first_order.jl")
    include("standard_form_second_order.jl")

    export flux_matrix!
    include("flux_differencing_form.jl")

    export LinearResidual
    include("linear.jl")

    export rhs_benchmark!
    include("benchmark.jl")
end