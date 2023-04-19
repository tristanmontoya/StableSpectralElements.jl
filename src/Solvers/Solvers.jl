module Solvers

    using UnPack
    import LinearAlgebra
    using LinearAlgebra: Diagonal, eigvals, inv, mul!, factorize, cholesky,ldiv!, Factorization, Cholesky, Symmetric
    using TimerOutputs
    using LinearMaps: LinearMap, UniformScalingMap
    using OrdinaryDiffEq: ODEProblem, OrdinaryDiffEqAlgorithm, solve
    
    using ..MatrixFreeOperators: AbstractOperatorAlgorithm, BLASAlgorithm, GenericMatrixAlgorithm, DefaultOperatorAlgorithm, WeightAdjustedMap, ZeroMap, make_operator
    using ..ConservationLaws: AbstractConservationLaw, AbstractPDEType, FirstOrder, SecondOrder, AbstractInviscidNumericalFlux, AbstractViscousNumericalFlux, AbstractTwoPointFlux, NoInviscidFlux, NoViscousFlux, NoTwoPointFlux, NoSourceTerm, physical_flux!, physical_flux, numerical_flux, numerical_flux!, LaxFriedrichsNumericalFlux, BR1
    using ..SpatialDiscretizations: ReferenceApproximation, SpatialDiscretization, check_facet_nodes, check_normals
    using ..GridFunctions: AbstractGridFunction, AbstractGridFunction, NoSourceTerm, evaluate   
    
    export AbstractResidualForm, StandardForm, AbstractMappingForm, AbstractStrategy, AbstractDiscretizationOperators, PhysicalOperators, PreAllocatedArrays, PhysicalOperator, ReferenceOperator, Solver, StandardMapping, SkewSymmetricMapping, get_dof, rhs!, make_operators

    abstract type AbstractResidualForm{MappingForm, TwoPointFlux} end
    abstract type AbstractMappingForm end
    abstract type AbstractStrategy end
    abstract type AbstractDiscretizationOperators{d} end

    StandardForm = AbstractResidualForm{<:AbstractMappingForm, NoTwoPointFlux}

    struct StandardMapping <: AbstractMappingForm end
    struct SkewSymmetricMapping <: AbstractMappingForm end
    struct PhysicalOperator <: AbstractStrategy end
    struct ReferenceOperator <: AbstractStrategy end

    struct PhysicalOperators{d} <: AbstractDiscretizationOperators{d}
        VOL::NTuple{d,LinearMap}
        FAC::LinearMap
        SRC::LinearMap
        NTR::NTuple{d,LinearMap}
        M::Union{Cholesky, AbstractMatrix, WeightAdjustedMap}
        V::LinearMap
        R::LinearMap
        n_f::NTuple{d, Vector{Float64}}
        N_p::Int
        N_q::Int
        N_f::Int
    end

    struct PreAllocatedArrays{PDEType}
        f_q::Array{Float64,4}
        f_f::Array{Float64,3}
        f_n::Array{Float64,3}
        u_q::Array{Float64,3}
        r_q::Array{Float64,3}
        u_f::Array{Float64,3}
        CI::CartesianIndices
        u_n::Union{Nothing,Array{Float64,4}}
        q_q::Union{Nothing,Array{Float64,4}}
        q_f::Union{Nothing,Array{Float64,4}}
    end
    
    @timeit "allocate" function PreAllocatedArrays{FirstOrder}( 
        N_q::Int,N_f::Int,N_c::Int,d::Int,N_e::Int)
        return PreAllocatedArrays{FirstOrder}(
            Array{Float64}(undef,N_q, N_c, d, N_e),
            Array{Float64}(undef,N_f, N_c, N_e),
            Array{Float64}(undef,N_f, N_c, N_e),
            Array{Float64}(undef,N_q, N_c, N_e),
            Array{Float64}(undef,N_q, N_c, N_e),
            Array{Float64}(undef,N_f, N_e, N_c), #note switched order
            CartesianIndices((N_f,N_e)), nothing, nothing, nothing)
    end

    function PreAllocatedArrays{SecondOrder}(
        N_q::Int, N_f::Int, N_c::Int, d::Int, N_e::Int)
        return PreAllocatedArrays{SecondOrder}(
            Array{Float64}(undef,N_q, N_c, d, N_e),
            Array{Float64}(undef,N_f, N_c, N_e),
            Array{Float64}(undef,N_f, N_c, N_e),
            Array{Float64}(undef,N_q, N_c, N_e),
            Array{Float64}(undef,N_q, N_c, N_e),
            Array{Float64}(undef,N_f, N_e, N_c), #note switched order
            CartesianIndices((N_f,N_e)),
            Array{Float64}(undef,N_f, N_c, d, N_e),
            Array{Float64}(undef,N_q, N_c, d, N_e),
            Array{Float64}(undef,N_f, N_e, N_c, d)) #note switched order
    end

    struct Solver{d,ResidualForm,PDEType,OperatorType}
        conservation_law::AbstractConservationLaw{d,PDEType}
        operators::Vector{OperatorType}
        connectivity::Matrix{Int}
        form::ResidualForm
        N_q::Int
        N_f::Int
        N_c::Int
        N_e::Int
        preallocated_arrays::PreAllocatedArrays{PDEType}
    end

    function Solver(conservation_law::AbstractConservationLaw{d,PDEType},
        operators::Vector{PhysicalOperators{d}},
        connectivity::Matrix{Int},
        form::ResidualForm) where {d,ResidualForm,PDEType}

        @unpack N_c = conservation_law
        N_e = length(operators)
        @unpack N_q, N_f = operators[1]

        return Solver{d,ResidualForm,PDEType,PhysicalOperators{d}}(conservation_law, operators, connectivity, form, N_q, N_f, N_c, N_e,
            PreAllocatedArrays{PDEType}(N_q,N_f,N_c,d,N_e))
    end

    @inline function get_dof(spatial_discretization::SpatialDiscretization{d}, 
        conservation_law::AbstractConservationLaw{d}) where {d}
        return (spatial_discretization.reference_approximation.N_p, 
            conservation_law.N_c, spatial_discretization.N_e)
    end

    @inline function diff_with_extrap_flux!(f_f::AbstractMatrix{Float64}, 
        ::AbstractMatrix{Float64}, ::NTuple{d, ZeroMap}, 
        ::AbstractArray{Float64,3}) where {d}
        return f_f
    end
    
    @inline function diff_with_extrap_flux!(f_f::AbstractMatrix{Float64}, 
        f_n::AbstractMatrix{Float64}, NTR::NTuple{d, <:LinearMap}, 
        f_q::AbstractArray{Float64,3}) where {d}
    
        @inbounds for m in 1:d
            mul!(f_n, NTR[m], f_q[:,:,m])
            f_f .+= f_n
        end
        return f_f
    end    

    export AbstractMassMatrixSolver, CholeskySolver, WeightAdjustedSolver, PreInvert, mass_matrix
    include("mass_matrix.jl") 

    export initialize, semidiscretize, precompute
    include("preprocessing.jl")

    export StrongConservationForm, WeakConservationForm, SplitConservationForm
    include("make_operators.jl")
    
    export diff_with_extrap_flux!
    include("standard_form_first_order.jl")
    include("standard_form_second_order.jl")

    export LinearResidual
    include("linear.jl")
end