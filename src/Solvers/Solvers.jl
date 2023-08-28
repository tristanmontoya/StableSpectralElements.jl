module Solvers
    
    import LinearAlgebra
    using GFlops
    using StaticArrays
    using MuladdMacro
    using SparseArrays
    using LinearAlgebra: Diagonal, eigvals, inv, mul!, lmul!, diag, diagm, factorize, cholesky, ldiv!, Factorization, Cholesky, Symmetric, I, UniformScaling
    using TimerOutputs
    using LinearMaps: LinearMap, UniformScalingMap
    using OrdinaryDiffEq: ODEProblem, OrdinaryDiffEqAlgorithm, solve
    using StartUpDG: num_faces
    using ..MatrixFreeOperators: AbstractOperatorAlgorithm,  DefaultOperatorAlgorithm, make_operator
    using ..ConservationLaws
    using ..SpatialDiscretizations: ReferenceApproximation, SpatialDiscretization, apply_reference_mapping, reference_derivative_operators, check_facet_nodes, check_normals, NodalTensor, ModalTensor
    using ..GridFunctions: AbstractGridFunction, AbstractGridFunction, NoSourceTerm, evaluate
    
    export AbstractResidualForm, StandardForm, FluxDifferencingForm, AbstractMappingForm, AbstractStrategy, AbstractDiscretizationOperators, AbstractPreallocatedArrays, AbstractMassMatrixSolver, PhysicalOperators, FluxDifferencingOperators, PreAllocatedArrays,  PreAllocatedArraysFluxDifferencing, PhysicalOperator, ReferenceOperator, Solver, StandardMapping, SkewSymmetricMapping, get_dof, rhs!, rhs_static!, make_operators, get_nodal_values!, facet_correction!, flux_differencing_operators

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
        entropy_projection::Bool = false
    end

    struct ReferenceOperators{d} <: AbstractDiscretizationOperators{d}
        D::NTuple{d,LinearMap}
        Dᵀ::NTuple{d,LinearMap}
        V::LinearMap
        Vᵀ::LinearMap
        R::LinearMap
        Rᵀ::LinearMap
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
        n_f::Vector{NTuple{d,Vector{Float64}}}
    end
    struct FluxDifferencingOperators{d} <: AbstractDiscretizationOperators{d}
        S::NTuple{d,AbstractMatrix}
        C::AbstractMatrix
        V::LinearMap
        Vᵀ::LinearMap
        R::LinearMap
        Rᵀ::LinearMap
        W::Diagonal
        B::Diagonal
        WJ::Vector{Diagonal}
        Λ_q::Array{Float64,4} # N_q x d x d x N_e
        BJf::Vector{Diagonal}
        n_f::Vector{NTuple{d, Vector{Float64}}}
        halfnJf::Array{Float64,3}
        halfnJq::Array{Float64,4}
        nodes_per_face::Int
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

    function PreAllocatedArrays{d,FirstOrder,N_p,N_q,N_f,N_c,N_e}(
        temp_size::Int=N_q) where {d,N_p,N_q,N_f,N_c,N_e}
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
        temp_size::Int=N_q) where {d,N_p,N_q,N_f,N_c,N_e}
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

    function Solver(
        conservation_law::AbstractConservationLaw{d,PDEType, N_c},     
        spatial_discretization::SpatialDiscretization{d},
        form::ResidualForm,
        ::PhysicalOperator,
        operator_algorithm::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm(),
        mass_solver::AbstractMassMatrixSolver=WeightAdjustedSolver(spatial_discretization,operator_algorithm)) where {d, PDEType, N_c,
            ResidualForm<:StandardForm}

        (; N_e) = spatial_discretization
        (; N_p, N_q, N_f) = spatial_discretization.reference_approximation

        operators = make_operators(spatial_discretization, form,
            operator_algorithm, mass_solver)
            
        return Solver{d,ResidualForm,PDEType,PhysicalOperators{d}, N_p, N_q, N_f, N_c, N_e}(
            conservation_law, operators, mass_solver, 
            spatial_discretization.mesh.mapP, form,
            PreAllocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e}(N_p))
    end
    
    function Solver(
        conservation_law::AbstractConservationLaw{d,PDEType,N_c},     
        spatial_discretization::SpatialDiscretization{d},
        form::ResidualForm,
        ::ReferenceOperator,
        alg::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm(),
        mass_solver::AbstractMassMatrixSolver=WeightAdjustedSolver(
            spatial_discretization,operator_algorithm)) where {d, PDEType, N_c,
            ResidualForm<:StandardForm}
    
        (; reference_approximation, geometric_factors, 
            N_e, mesh) = spatial_discretization
        (; N_p, N_q, N_f, D, V, W, R, B) = reference_approximation
        (; Λ_q, nJf, J_f) = apply_reference_mapping(geometric_factors,
            reference_approximation.reference_mapping)

        halfWΛ = Array{Diagonal,3}(undef, d, d, N_e)
        halfN = Matrix{Diagonal}(undef, d, N_e)
        BJf = Vector{Diagonal}(undef, N_e)
        n_f = Vector{NTuple{d, Vector{Float64}}}(undef,N_e)
    
        Threads.@threads for k in 1:N_e
            halfWΛ[:,:,k] = [Diagonal(0.5 * W * Λ_q[:,m,n,k]) 
                for m in 1:d, n in 1:d]
            halfN[:,k] = [Diagonal(0.5 * nJf[m,:,k] ./ J_f[:,k]) for m in 1:d]
            BJf[k] = Diagonal(B .* J_f[:,k])
            n_f[k] = Tuple(nJf[m,:,k] ./ J_f[:,k] for m in 1:d)
        end
    
        operators = ReferenceOperators{d}(
            Tuple(make_operator(D[m], alg) for m in 1:d), 
            Tuple(transpose(make_operator(D[m], alg)) for m in 1:d), 
            make_operator(V, alg), transpose(make_operator(V, alg)),
            make_operator(R, alg), transpose(make_operator(R, alg)),
            W, B, halfWΛ, halfN, BJf, n_f)
    
        return Solver{d,ResidualForm,PDEType,ReferenceOperators{d}, N_p, N_q, N_f, N_c, N_e}(
            conservation_law, operators, mass_solver, mesh.mapP, form,
            PreAllocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e}())
    end

    function Solver(
        conservation_law::AbstractConservationLaw{d,PDEType,N_c},     
        spatial_discretization::SpatialDiscretization{d},
        form::ResidualForm,
        ::ReferenceOperator,
        alg::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm(),
        mass_solver::AbstractMassMatrixSolver=WeightAdjustedSolver(
            spatial_discretization,operator_algorithm)) where {d, PDEType, N_c,ResidualForm<:FluxDifferencingForm}
        (; reference_approximation, N_e) = spatial_discretization
        (; N_p, N_q, N_f, V, W, R, B) = reference_approximation
        (; J_q, Λ_q, nJf, nJq, J_f) = spatial_discretization.geometric_factors
        (; element_type) = reference_approximation.reference_element

        WJ = Vector{Diagonal}(undef, N_e)
        BJf = Vector{Diagonal}(undef, N_e)
        n_f = Vector{NTuple{d, Vector{Float64}}}(undef, N_e)
    
        Threads.@threads for k in 1:N_e
            WJ[k] = Diagonal(W .* J_q[:,k])
            BJf[k] = Diagonal(B .* J_f[:,k])
            n_f[k] = Tuple(nJf[m,:,k] ./ J_f[:,k] for m in 1:d)
        end
        
        S, C = flux_differencing_operators(reference_approximation)

        operators = FluxDifferencingOperators{d}(S, C, make_operator(V, alg),
            transpose(make_operator(V, alg)), make_operator(R, alg), transpose(make_operator(R, alg)), W, B, WJ, Λ_q, BJf, n_f, 0.5*nJf, 
            0.5*nJq, N_f÷num_faces(element_type))
    
        return Solver{d,ResidualForm,PDEType,FluxDifferencingOperators{d},N_p,N_q,N_f, N_c, N_e}(conservation_law, operators, mass_solver,
            spatial_discretization.mesh.mapP, form, 
            PreAllocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e}(N_p))
    end    

    @inline function get_dof(spatial_discretization::SpatialDiscretization{d}, 
        ::AbstractConservationLaw{d,PDEType,N_c}) where {d, PDEType,N_c}
        return (spatial_discretization.reference_approximation.N_p, N_c, 
            spatial_discretization.N_e)
    end

    function flux_differencing_operators(
        reference_approximation::ReferenceApproximation{d}) where {d}

        (; D, W, R, B, approx_type, reference_mapping) = reference_approximation

        D_ξ = reference_derivative_operators(D, reference_mapping)

        S = Tuple(0.5*Matrix(W*D_ξ[m] - D_ξ[m]'*W) for m in 1:d)
        C = Matrix(R'*B)
        
        if (approx_type isa Union{NodalTensor,ModalTensor}) && (d > 1)
            return Tuple(sparse(S[m]) for m in 1:d), sparse(C)
        else
            return S, C
        end
    end

    export CholeskySolver, WeightAdjustedSolver, DiagonalSolver, mass_matrix, mass_matrix_inverse, mass_matrix_solve!
    include("mass_matrix.jl") 

    export initialize, semidiscretize
    include("preprocessing.jl")

    include("standard_form_first_order.jl")
    include("standard_form_second_order.jl")
    include("flux_differencing_form.jl")

    export LinearResidual
    include("linear.jl")

    export rhs_benchmark!, rhs_volume!, rhs_facet!
    include("benchmark.jl")
end