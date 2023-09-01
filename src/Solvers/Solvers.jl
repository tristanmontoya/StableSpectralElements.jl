module Solvers
    
    import LinearAlgebra
    using GFlops
    using StaticArrays
    using MuladdMacro
    using SparseArrays
    using LinearAlgebra: Diagonal, eigvals, inv, mul!, lmul!, diag, diagm, factorize, cholesky, ldiv!, Factorization, Cholesky, Symmetric, I, UniformScaling
    using TimerOutputs
    using LinearMaps: LinearMap, UniformScalingMap, TransposeMap
    using OrdinaryDiffEq: ODEProblem, OrdinaryDiffEqAlgorithm, solve
    using StartUpDG: num_faces
    using ..MatrixFreeOperators: AbstractOperatorAlgorithm,  DefaultOperatorAlgorithm, make_operator
    using ..ConservationLaws
    using ..SpatialDiscretizations: AbstractApproximationType, ReferenceApproximation, SpatialDiscretization, apply_reference_mapping, reference_derivative_operators, check_facet_nodes, check_normals, NodalTensor, ModalTensor
    using ..GridFunctions: AbstractGridFunction, AbstractGridFunction, NoSourceTerm, evaluate
    
    export AbstractResidualForm, StandardForm, FluxDifferencingForm, AbstractMappingForm, AbstractStrategy, AbstractDiscretizationOperators,  AbstractMassMatrixSolver, AbstractParallelism, ReferenceOperators, PhysicalOperators, FluxDifferencingOperators, PreAllocatedArrays, PhysicalOperator, ReferenceOperator, Solver, StandardMapping, SkewSymmetricMapping, Serial, Threaded, get_dof, semi_discrete_residual!, auxiliary_variable!, make_operators, entropy_projection!, facet_correction!, nodal_values!, time_derivative!, project_function!,flux_differencing_operators

    abstract type AbstractResidualForm{MappingForm, TwoPointFlux} end
    abstract type AbstractMappingForm end
    abstract type AbstractStrategy end
    abstract type AbstractDiscretizationOperators end
    abstract type AbstractMassMatrixSolver end
    abstract type AbstractParallelism end

    struct StandardMapping <: AbstractMappingForm end
    struct SkewSymmetricMapping <: AbstractMappingForm end
    struct PhysicalOperator <: AbstractStrategy end
    struct ReferenceOperator <: AbstractStrategy end
    struct Serial <: AbstractParallelism end
    struct Threaded <: AbstractParallelism end

    Base.@kwdef struct StandardForm{MappingForm,InviscidNumericalFlux,
        ViscousNumericalFlux} <: AbstractResidualForm{MappingForm,NoTwoPointFlux}
        mapping_form::MappingForm = SkewSymmetricMapping()
        inviscid_numerical_flux::InviscidNumericalFlux =
            LaxFriedrichsNumericalFlux()
        viscous_numerical_flux::ViscousNumericalFlux = BR1()
    end

    Base.@kwdef struct FluxDifferencingForm{MappingForm,
        InviscidNumericalFlux,ViscousNumericalFlux,
        TwoPointFlux} <: AbstractResidualForm{MappingForm,TwoPointFlux}
        mapping_form::MappingForm = SkewSymmetricMapping()
        inviscid_numerical_flux::InviscidNumericalFlux =
            LaxFriedrichsNumericalFlux()
        viscous_numerical_flux::ViscousNumericalFlux = BR1()
        two_point_flux::TwoPointFlux = EntropyConservativeFlux()
        facet_correction::Bool = false
        entropy_projection::Bool = false
    end

    struct ReferenceOperators{D_type, Dt_type, V_type, Vt_type,
        R_type, Rt_type} <: AbstractDiscretizationOperators
        D::D_type
        Dᵀ::Dt_type
        V::V_type
        Vᵀ::Vt_type
        R::R_type
        Rᵀ::Rt_type
        W::Diagonal{Float64, Vector{Float64}}
        B::Diagonal{Float64, Vector{Float64}}
        halfWΛ::Array{Diagonal{Float64, Vector{Float64}},3} # d x d x N_e
        halfN::Matrix{Diagonal{Float64, Vector{Float64}}}
        BJf::Vector{Diagonal{Float64, Vector{Float64}}}
        n_f::Array{Float64,3}
    end

    struct PhysicalOperators{d, VOL_type, FAC_type, V_type, 
        R_type} <: AbstractDiscretizationOperators
        VOL::Vector{NTuple{d,VOL_type}}
        FAC::Vector{FAC_type}
        V::Vector{V_type}
        R::Vector{R_type}
        n_f::Array{Float64,3}
    end

    struct FluxDifferencingOperators{S_type,
        C_type, V_type, Vt_type, R_type, 
        Rt_type} <: AbstractDiscretizationOperators
        S::S_type
        C::C_type
        V::V_type
        Vᵀ::Vt_type
        R::R_type
        Rᵀ::Rt_type
        W::Diagonal{Float64, Vector{Float64}}
        B::Diagonal{Float64, Vector{Float64}}
        WJ::Vector{Diagonal{Float64, Vector{Float64}}}
        Λ_q::Array{Float64,4}
        BJf::Vector{Diagonal{Float64, Vector{Float64}}}
        n_f::Array{Float64,3}
        halfnJf::Array{Float64,3}
        halfnJq::Array{Float64,4}
        nodes_per_face::Int
    end

    struct PreAllocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e}
        f_q::Array{Float64,4}
        f_f::Array{Float64,3}
        f_n::Array{Float64,3}
        u_q::Array{Float64,3}
        r_q::Array{Float64,3}
        u_f::Array{Float64,3}
        temp::Array{Float64,3}
        CI::CartesianIndices{2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}
        u_n::Union{Nothing,Array{Float64,4}}  # for viscous terms
        q_q::Union{Nothing,Array{Float64,4}}  # for viscous terms
        q_f::Union{Nothing,Array{Float64,4}}  # for viscous terms
    end

    struct Solver{d,ResidualForm,PDEType,ConservationLaw,
        Operators,MassSolver,Parallelism,N_p,N_q,N_f,N_c,N_e}
        conservation_law::ConservationLaw
        operators::Operators
        mass_solver::MassSolver
        connectivity::Matrix{Int}
        form::ResidualForm
        parallelism::Parallelism
        preallocated_arrays::PreAllocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e}
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
        conservation_law::AbstractConservationLaw{d,PDEType,N_c},     
        spatial_discretization::SpatialDiscretization{d},
        form::ResidualForm, ::PhysicalOperator,
        alg::AbstractOperatorAlgorithm,
        mass_solver::AbstractMassMatrixSolver,
        parallelism::AbstractParallelism) where {d, PDEType, N_c,
            ResidualForm<:StandardForm}

        (; N_e) = spatial_discretization
        (; N_p, N_q, N_f) = spatial_discretization.reference_approximation

        operators = make_operators(spatial_discretization, form,
            alg, mass_solver)
            
        return Solver(conservation_law, operators, mass_solver, 
            spatial_discretization.mesh.mapP, form,
            parallelism, PreAllocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e}(N_p))
    end
    
    function Solver(
        conservation_law::AbstractConservationLaw{d,PDEType,N_c},     
        spatial_discretization::SpatialDiscretization{d},
        form::ResidualForm, ::ReferenceOperator,
        alg::AbstractOperatorAlgorithm,
        mass_solver::AbstractMassMatrixSolver,
        parallelism::AbstractParallelism) where {d, PDEType, N_c,
            ResidualForm<:StandardForm}
    
        (; reference_approximation, geometric_factors, 
            N_e, mesh) = spatial_discretization
        (; N_p, N_q, N_f, D, V, W, R, B) = reference_approximation
        (; Λ_q, nJf, J_f) = apply_reference_mapping(geometric_factors,
            reference_approximation.reference_mapping)

        halfWΛ = Array{Diagonal{Float64, Vector{Float64}},3}(undef, d, d, N_e)
        halfN = Matrix{Diagonal{Float64, Vector{Float64}}}(undef, d, N_e)
        BJf = Vector{Diagonal{Float64, Vector{Float64}}}(undef, N_e)
        n_f = Array{Float64,3}(undef, d, N_f, N_e)
    
        @inbounds Threads.@threads for k in 1:N_e
            @inbounds for m in 1:d
                halfWΛ[m,:,k] .= [Diagonal(0.5 * W * Λ_q[:,m,n,k]) for n in 1:d]
                n_f[m,:,k] .= nJf[m,:,k] ./ J_f[:,k]
                halfN[m,k] = Diagonal(0.5 * n_f[m,:,k])
            end
            BJf[k] = Diagonal(B .* J_f[:,k])
        end
    
        operators = ReferenceOperators(
            Tuple(make_operator(D[m], alg) for m in 1:d), 
            Tuple(transpose(make_operator(D[m], alg)) for m in 1:d), 
            make_operator(V, alg), transpose(make_operator(V, alg)),
            make_operator(R, alg), transpose(make_operator(R, alg)),
            W, B, halfWΛ, halfN, BJf, n_f)
    
        return Solver(conservation_law, operators, mass_solver, mesh.mapP, form,
            parallelism, PreAllocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e}())
    end

    function Solver(
        conservation_law::AbstractConservationLaw{d,PDEType,N_c},     
        spatial_discretization::SpatialDiscretization{d},
        form::ResidualForm, ::ReferenceOperator,
        alg::AbstractOperatorAlgorithm,
        mass_solver::AbstractMassMatrixSolver,
        parallelism::AbstractParallelism) where {d, PDEType, N_c,  
            ResidualForm<:FluxDifferencingForm}

        (; reference_approximation, N_e, mesh) = spatial_discretization
        (; N_p, N_q, N_f, V, W, R, B) = reference_approximation
        (; J_q, Λ_q, nJf, nJq, J_f) = spatial_discretization.geometric_factors
        (; element_type) = reference_approximation.reference_element

        WJ = Vector{Diagonal{Float64, Vector{Float64}}}(undef, N_e)
        BJf = Vector{Diagonal{Float64, Vector{Float64}}}(undef, N_e)
        n_f = Array{Float64,3}(undef, d, N_f, N_e)
    
        @inbounds Threads.@threads for k in 1:N_e
            WJ[k] = Diagonal(W .* J_q[:,k])
            BJf[k] = Diagonal(B .* J_f[:,k])
            @inbounds for m in 1:d
                n_f[m,:,k] = nJf[m,:,k] ./ J_f[:,k]
            end
        end
        
        S, C = flux_differencing_operators(reference_approximation)

        operators = FluxDifferencingOperators(S, C, make_operator(V, alg),
            transpose(make_operator(V, alg)), make_operator(R, alg), 
            transpose(make_operator(R, alg)), W, B, WJ, Λ_q, BJf, n_f, 0.5*nJf, 
            0.5*nJq, N_f÷num_faces(element_type))
    
        return Solver(conservation_law, operators, mass_solver, mesh.mapP, form,
            parallelism, PreAllocatedArrays{d,PDEType,N_p,N_q,N_f,N_c,N_e}(N_p))
    end    

    function flux_differencing_operators(
        reference_approximation::ReferenceApproximation{1, ElemShape, 
        ApproxType}) where {ElemShape, 
        ApproxType<:Union{NodalTensor,ModalTensor}}

        (; D, W, R, B, reference_mapping) = reference_approximation

        D_ξ = reference_derivative_operators(D, reference_mapping)
        S = (0.5*Matrix(W*D_ξ[1] - D_ξ[1]'*W),)
        C = Matrix(R'*B)

        return S, C
    end

    function flux_differencing_operators(
        reference_approximation::ReferenceApproximation{d, ElemShape, 
        ApproxType}) where {d, ElemShape, 
        ApproxType<:Union{NodalTensor,ModalTensor}}

        (; D, W, R, B, reference_mapping) = reference_approximation

        D_ξ = reference_derivative_operators(D, reference_mapping)

        S = Tuple(0.5*Matrix(W*D_ξ[m] - D_ξ[m]'*W) for m in 1:d)
        C = Matrix(R'*B)
        
        return Tuple(sparse(S[m]) for m in 1:d), sparse(C)
    end

    function flux_differencing_operators(
        reference_approximation::ReferenceApproximation{d, ElemShape, 
        ApproxType}) where {d, ElemShape, 
        ApproxType<:AbstractApproximationType}

        (; D, W, R, B, reference_mapping) = reference_approximation

        D_ξ = reference_derivative_operators(D, reference_mapping)
        S = Tuple(0.5*Matrix(W*D_ξ[m] - D_ξ[m]'*W) for m in 1:d)
        C = Matrix(R'*B)
        
        return S, C
    end

    @timeit "semi-disc. residual" function semi_discrete_residual!(
        dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3},
        solver::Solver{d,ResidualForm,FirstOrder,ConservationLaw,Operators,
        MassSolver,Serial,N_p,N_q,N_f,N_c,N_e}, t::Float64=0.0) where {d,ResidualForm,ConservationLaw,Operators,MassSolver,N_p,N_q,N_f,N_c,N_e}

        @inbounds for k in 1:N_e
            @timeit "nodal values" nodal_values!(u, solver, k)
        end

        @inbounds for k in 1:N_e
            @timeit "time deriv." time_derivative!(dudt, solver, k)
        end

        return dudt
    end

    @timeit "semi-disc. residual" function semi_discrete_residual!(
        dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3},
        solver::Solver{d,ResidualForm,FirstOrder,ConservationLaw,Operators,
        MassSolver,Threaded,N_p,N_q,N_f,N_c,N_e}, t::Float64=0.0) where {d,ResidualForm,ConservationLaw,Operators,MassSolver,N_p,N_q,N_f,N_c,N_e}
        
        @inbounds Threads.@threads for k in 1:N_e
            nodal_values!(u, solver, k)
        end

        @inbounds Threads.@threads for k in 1:N_e
            time_derivative!(dudt, solver, k)
        end

        return dudt
    end

    @timeit "semi-disc. residual" function semi_discrete_residual!(
        dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3},
        solver::Solver{d,ResidualForm,SecondOrder,ConservationLaw,Operators,
        MassSolver,Serial,N_p,N_q,N_f,N_c,N_e}, t::Float64=0.0) where {d,ResidualForm,ConservationLaw,Operators,MassSolver,N_p,N_q,N_f,N_c,N_e}

        @inbounds for k in 1:N_e
            nodal_values!(u, solver, k)
        end
        
        @inbounds for k in 1:N_e
            auxiliary_variable!(dudt, solver, k)
        end

        @inbounds for k in 1:N_e
            time_derivative!(dudt, solver, k)
        end

        return dudt
    end

    @timeit "semi-disc. residual" function semi_discrete_residual!(
        dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3},
        solver::Solver{d,ResidualForm,SecondOrder,ConservationLaw,Operators,
        MassSolver,Threaded,N_p,N_q,N_f,N_c,N_e}, t::Float64=0.0) where {d,ResidualForm,ConservationLaw,Operators,MassSolver,N_p,N_q,N_f,N_c,N_e}

        @inbounds Threads.@threads for k in 1:N_e
            nodal_values!(u, solver, k)
        end
        
        @inbounds Threads.@threads for k in 1:N_e
            auxiliary_variable!(dudt, solver, k)
        end

        @inbounds Threads.@threads for k in 1:N_e
            time_derivative!(dudt, solver, k)
        end

        return dudt
    end

    @inline function get_dof(spatial_discretization::SpatialDiscretization{d}, 
        ::AbstractConservationLaw{d,PDEType,N_c}) where {d, PDEType,N_c}
        return (spatial_discretization.reference_approximation.N_p, N_c, 
            spatial_discretization.N_e)
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

    export rhs_benchmark!, rhs_volume!, rhs_facet!, rhs_benchmark_notime!
    include("benchmark.jl")
end