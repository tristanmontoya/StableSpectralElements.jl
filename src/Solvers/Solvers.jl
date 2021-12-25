module Solvers

    using OrdinaryDiffEq: ODEProblem, OrdinaryDiffEqAlgorithm, solve
    using UnPack
    using LinearAlgebra: Diagonal, inv
    using LinearMaps: LinearMap
    using TimerOutputs

    using ..ConservationLaws: ConservationLaw, physical_flux, numerical_flux
    using ..SpatialDiscretizations: ReferenceApproximation, SpatialDiscretization
    using ..InitialConditions: AbstractInitialData, evaluate
    
    export AbstractResidualForm, AbstractPhysicalOperators, PhysicaOperators, AbstractStrategy, Eager, Lazy, Solver, initialize, semidiscretize, precompute, apply_operators, combine, get_dof, rhs!

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
        strategy::AbstractStrategy
    end

    struct PhysicalOperators{d} <: AbstractPhysicalOperators{d}
        VOL::NTuple{d,LinearMap}
        FAC::LinearMap
        M::AbstractMatrix
        V::LinearMap
        R::LinearMap
        NTR::NTuple{d,Union{LinearMap,AbstractMatrix}} 
        scaled_normal::NTuple{d, Vector{Float64}}
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
            u0, form, tspan, strategy)
    end

    function semidiscretize(conservation_law::ConservationLaw{d,N_eq},
        spatial_discretization::SpatialDiscretization{d},
        u0::Array{Float64,3},
        form::AbstractResidualForm,
        tspan::NTuple{2,Float64}, strategy::Lazy) where {d,N_eq}
        
        return ODEProblem(rhs!, u0, tspan, Solver(conservation_law,            
            make_operators(spatial_discretization, form),
            spatial_discretization.mesh.mapP, form, strategy))
    end

    function semidiscretize(conservation_law::ConservationLaw{d,N_eq},
        spatial_discretization::SpatialDiscretization{d},
        u0::Array{Float64,3}, 
        form::AbstractResidualForm,
        tspan::NTuple{2,Float64}, strategy::Eager) where{d, N_eq}
        
        operators = make_operators(spatial_discretization, form)

        return ODEProblem(rhs!, u0, tspan, Solver(conservation_law, 
        [precompute(operators[k]) for k in 1:spatial_discretization.N_el],
            spatial_discretization.mesh.mapP, form, strategy))
    end

    function apply_operators(operators::PhysicalOperators{d},       
        f::NTuple{d,Matrix{Float64}}, f_fac::Matrix{Float64}, ::Eager) where {d}
        return sum(convert(Matrix, operators.VOL[m] * f[m]) for m in 1:d) +     
            convert(Matrix,operators.FAC * f_fac)
    end

    function apply_operators(operators::PhysicalOperators{d},
        f::NTuple{d,Matrix{Float64}}, f_fac::Matrix{Float64}, ::Lazy) where {d}
        rhs = sum(convert(Matrix, operators.VOL[m] * f[m]) for m in 1:d) +  convert(Matrix,operators.FAC * f_fac)
        return operators.M \ rhs
    end

    function precompute(operators::PhysicalOperators{d}) where {d}
        @unpack VOL, FAC, M, V, R, NTR, scaled_normal = operators
        inv_M = inv(M)
        return PhysicalOperators(
            Tuple(combine(inv_M*VOL[n]) for n in 1:d),
            combine(inv_M*FAC), M, V, R, NTR, scaled_normal)
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