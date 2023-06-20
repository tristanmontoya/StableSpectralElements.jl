module Solvers
    
    import LinearAlgebra
    using GFlops
    using StaticArrays
    using LinearAlgebra: Diagonal, eigvals, inv, mul!, lmul!, diag, diagm, factorize, cholesky, ldiv!, Factorization, Cholesky, Symmetric, I, UniformScaling
    using TimerOutputs
    using LinearMaps: LinearMap, UniformScalingMap
    using OrdinaryDiffEq: ODEProblem, OrdinaryDiffEqAlgorithm, solve
    
    using ..MatrixFreeOperators: AbstractOperatorAlgorithm,  DefaultOperatorAlgorithm, make_operator
    using ..ConservationLaws: AbstractConservationLaw, AbstractPDEType, FirstOrder, SecondOrder, AbstractInviscidNumericalFlux, AbstractViscousNumericalFlux, AbstractTwoPointFlux, NoInviscidFlux, NoViscousFlux, NoTwoPointFlux, NoSourceTerm, physical_flux!, physical_flux, numerical_flux, numerical_flux!, LaxFriedrichsNumericalFlux, BR1
    using ..SpatialDiscretizations: ReferenceApproximation, SpatialDiscretization, check_facet_nodes, check_normals
    using ..GridFunctions: AbstractGridFunction, AbstractGridFunction, NoSourceTerm, evaluate   
    
    export AbstractResidualForm, StandardForm, AbstractMappingForm, AbstractStrategy, AbstractDiscretizationOperators, AbstractMassMatrixSolver, PhysicalOperators, PreAllocatedArrays, PhysicalOperator, ReferenceOperator, Solver, StandardMapping, SkewSymmetricMapping, get_dof, rhs!, rhs_static!, make_operators, StandardForm

    abstract type AbstractResidualForm{MappingForm, TwoPointFlux} end
    abstract type AbstractMappingForm end
    abstract type AbstractStrategy end
    abstract type AbstractDiscretizationOperators{d} end
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

    struct ReferenceOperators{d} <: AbstractDiscretizationOperators{d}
        D::NTuple{d,LinearMap}
        V::LinearMap
        R::LinearMap
        W::LinearMap
        B::LinearMap
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
    struct PreAllocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e}
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
    
    function PreAllocatedArrays{d,FirstOrder,N_p,N_q,N_f,N_c,N_e}(
        temp_size::Int=N_p) where {d,N_p,N_q,N_f,N_c,N_e}
        return PreAllocatedArrays{d,FirstOrder,N_p,N_q,N_f,N_c,N_e}(
            Array{Float64}(undef,N_q, N_c, d, N_e),
            Array{Float64}(undef,N_f, N_c, N_e),
            Array{Float64}(undef,N_f, N_c, N_e),
            Array{Float64}(undef,N_q, N_c, N_e),
            Array{Float64}(undef,N_q, N_c, N_e),
            Array{Float64}(undef,N_f, N_e, N_c), #note switched order
            Array{Float64}(undef,temp_size, N_c, N_e),
            CartesianIndices((N_f,N_e)), nothing, nothing, nothing)
    end

    function PreAllocatedArrays{d,SecondOrder,N_p,N_q,N_f,N_c,N_e}(
        temp_size::Int=N_p) where {d,N_p,N_q,N_f,N_c,N_e}
        return PreAllocatedArrays{d,SecondOrder,N_p,N_q,N_f,N_c,N_e}(
            Array{Float64}(undef,N_q, N_c, d, N_e),
            Array{Float64}(undef,N_f, N_c, N_e),
            Array{Float64}(undef,N_f, N_c, N_e),
            Array{Float64}(undef,N_q, N_c, N_e),
            Array{Float64}(undef,N_q, N_c, N_e),
            Array{Float64}(undef,N_f, N_e, N_c), #note switched order
            Array{Float64}(undef,temp_size, N_c, N_e),
            CartesianIndices((N_f,N_e)),
            Array{Float64}(undef,N_f, N_c, d, N_e),
            Array{Float64}(undef,N_q, N_c, d, N_e),
            Array{Float64}(undef,N_f, N_e, N_c, d)) #note switched order
    end

    struct Solver{d,ResidualForm,PDEType,OperatorType,N_p,N_q,N_f,N_c,N_e}
        conservation_law::AbstractConservationLaw{d,PDEType}
        operators::OperatorType
        mass_solver::AbstractMassMatrixSolver
        connectivity::Matrix{Int}
        form::ResidualForm
        preallocated_arrays::PreAllocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e}
    end

    function Solver(conservation_law::AbstractConservationLaw{d,PDEType},
        operators::ReferenceOperators{d},
        mass_solver::AbstractMassMatrixSolver,
        connectivity::Matrix{Int},
        form::ResidualForm) where {d,ResidualForm,PDEType}

        (; N_c) = conservation_law
        N_e = size(operators.halfWΛ,3)
        (N_q,N_p) = size(operators.V)
        N_f = size(operators.R,1)

        return Solver{d,ResidualForm,PDEType,ReferenceOperators{d}, N_p, N_q, N_f, N_c, N_e}(
            conservation_law, operators, mass_solver, connectivity, form,
            PreAllocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e}(N_q))
    end

    function Solver(conservation_law::AbstractConservationLaw{d,PDEType},
        operators::PhysicalOperators{d},
        mass_solver::AbstractMassMatrixSolver,
        connectivity::Matrix{Int},
        form::ResidualForm) where {d,ResidualForm,PDEType}

        # get array sizes
        (; N_c) = conservation_law
        N_e = length(operators.V)
        (N_q,N_p) = size(operators.V[1])
        N_f = size(operators.R[1],1)

        return Solver{d,ResidualForm,PDEType,PhysicalOperators{d}, N_p, N_q, N_f, N_c, N_e}(
            conservation_law, operators, mass_solver, connectivity, form,
            PreAllocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e}(N_q))
    end

    function Solver(conservation_law::AbstractConservationLaw,     
        spatial_discretization::SpatialDiscretization,
        form::AbstractResidualForm,
        ::PhysicalOperator,
        operator_algorithm::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm(),
        mass_solver::AbstractMassMatrixSolver=WeightAdjustedSolver(spatial_discretization))
    
        operators = make_operators(spatial_discretization, form,
            operator_algorithm, mass_solver)
    
        return Solver(conservation_law, operators, mass_solver,
            spatial_discretization.mesh.mapP, form)
    end
    
    function Solver(conservation_law::AbstractConservationLaw,     
        spatial_discretization::SpatialDiscretization{d},
        form::AbstractResidualForm,
        ::ReferenceOperator,
        alg::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm(),
        mass_solver::AbstractMassMatrixSolver=WeightAdjustedSolver(spatial_discretization)) where {d}
    
        (; D, V, W, R, B) = spatial_discretization.reference_approximation
        (; J_q, Λ_q, nJf, J_f) = spatial_discretization.geometric_factors
        (; N_e) = spatial_discretization
    
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
    
        return Solver(conservation_law, operators, mass_solver,
            spatial_discretization.mesh.mapP, form)
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

    export LinearResidual
    include("linear.jl")

    export rhs_benchmark!
    include("benchmark.jl")
end