module Solvers

    using OrdinaryDiffEq: ODEProblem, OrdinaryDiffEqAlgorithm, solve
    using UnPack
    using LinearAlgebra: Diagonal, inv
    using LinearMaps: LinearMap
    using TimerOutputs

    using ..ConservationLaws: ConservationLaw, physical_flux, numerical_flux
    using ..SpatialDiscretizations: ReferenceApproximation, SpatialDiscretization
    using ..InitialConditions: AbstractInitialData, evaluate
    
    export AbstractResidualForm, AbstractPhysicalOperators, AbstractStrategy, Solver, PhysicalOperatorsEager, PhysicalOperatorsEager, Eager, Lazy, initialize, semidiscretize, apply_operators, combine, get_dof, rhs!

    abstract type AbstractResidualForm end
    abstract type AbstractPhysicalOperators{d} end
    abstract type AbstractStrategy end

    struct Eager <: AbstractStrategy end
    struct Lazy <: AbstractStrategy end

    struct Solver{ResidualForm,PhysicalOperators,d,N_eq}
        conservation_law::ConservationLaw{d,N_eq}
        operators::Vector{PhysicalOperators}
        connectivity::Matrix{Int}
        form::ResidualForm
    end

    struct PhysicalOperatorsEager{d} <: AbstractPhysicalOperators{d}
        VOL::NTuple{d,LinearMap}
        FAC::LinearMap
        V::LinearMap
        R::LinearMap
        NTR::NTuple{d,Union{LinearMap,AbstractMatrix}} 
        scaled_normal::NTuple{d, Vector{Float64}}
    end

    struct PhysicalOperatorsLazy{d} <: AbstractPhysicalOperators{d}
        vol::NTuple{d,LinearMap}  # not pre-multipled by mass inverse
        fac::LinearMap  # not pre-multiplied by mass inverse
        M::AbstractMatrix
        V::LinearMap
        R::LinearMap
        NTR::NTuple{d,Union{LinearMap,AbstractMatrix}}  
        scaled_normal::NTuple{d, Vector}
    end
    
    function initialize(initial_data::AbstractInitialData,
        conservation_law::ConservationLaw{d,N_eq},
        spatial_discretization::SpatialDiscretization{d}) where {d, N_eq}

        @unpack N_el, M, geometric_factors = spatial_discretization
        @unpack N_p, N_q, V, W = spatial_discretization.reference_approximation
        @unpack xyzq = spatial_discretization.mesh

        u0 = Array{Float64}(undef, N_p, N_eq, N_el)
        for k in 1:N_el
            # project to solution DOF
            u0[:,:,k] = M[k] \ convert(Matrix, transpose(V) * W * 
                Diagonal(geometric_factors.J[:,k]) * 
                evaluate(initial_data, Tuple(xyzq[m][:,k] for m in 1:d)))
        end
        return u0
    end

    function semidiscretize(
        conservation_law::ConservationLaw,spatial_discretization::SpatialDiscretization,
        initial_data::AbstractInitialData, 
        form::AbstractResidualForm,
        tspan::NTuple{2,Float64}, 
        strategy::AbstractStrategy)

        u0 = initialize(
            initial_data,
            conservation_law,
            spatial_discretization)

        return semidiscretize(conservation_law, spatial_discretization, 
            u0,form,tspan, strategy)
    end

    function apply_operators(operators::PhysicalOperatorsEager{d},
        f::NTuple{d,Matrix{Float64}}, f_fac::Matrix{Float64}) where {d}
        return sum(convert(Matrix, operators.VOL[m] * f[m]) for m in 1:d) +     
            convert(Matrix,operators.FAC * f_fac)
    end

    function apply_operators(operators::PhysicalOperatorsLazy{d},
        f::NTuple{d,Matrix{Float64}}, f_fac::Matrix{Float64}) where {d}
        rhs = sum(convert(Matrix, operators.vol[m] * f[m]) for m in 1:d) +  convert(Matrix,operators.fac * f_fac)

        return operators.M \ rhs
    end

    # utils
    function combine(operator::LinearMap)
        return LinearMap(convert(Matrix,operator))
    end

    function get_dof(spatial_discretization::SpatialDiscretization{d}, 
        ::ConservationLaw{d,N_eq}) where {d, N_eq}
        return (spatial_discretization.reference_approximation.N_p, 
            N_eq, spatial_discretization.N_el)
    end

    export StrongConservationForm
    include("strong_conservation_form.jl")

    export WeakConservationForm
    include("weak_conservation_form.jl")

end