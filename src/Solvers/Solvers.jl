module Solvers

    using UnPack
    import LinearAlgebra
    using LinearAlgebra: Diagonal, inv, mul!
    using TimerOutputs
    using LinearMaps: LinearMap
    using OrdinaryDiffEq: ODEProblem, OrdinaryDiffEqAlgorithm, solve
    
    using ..MatrixFreeOperators: combine
    using ..ConservationLaws: AbstractConservationLaw, AbstractPDEType, FirstOrder, SecondOrder, AbstractInviscidNumericalFlux, AbstractViscousNumericalFlux, AbstractTwoPointFlux, NoInviscidFlux, NoViscousFlux, NoTwoPointFlux, NoSourceTerm, physical_flux, numerical_flux, LaxFriedrichsNumericalFlux, BR1
    using ..SpatialDiscretizations: ReferenceApproximation, SpatialDiscretization
    using ..GridFunctions: AbstractGridFunction, AbstractGridFunction, NoSourceTerm, evaluate    
    
    export AbstractResidualForm, AbstractMappingForm, AbstractStrategy, DiscretizationOperators, PhysicalOperator, ReferenceOperator, Solver, StandardMapping, SkewSymmetricMapping, apply_operators, auxiliary_variable, get_dof, CLOUD_print_timer, CLOUD_reset_timer!, thread_timer, rhs!

    abstract type AbstractResidualForm{MappingForm, TwoPointFlux} end
    abstract type AbstractMappingForm end
    abstract type AbstractStrategy end

    struct PhysicalOperator <: AbstractStrategy end
    struct ReferenceOperator <: AbstractStrategy end
    struct StandardMapping <: AbstractMappingForm end
    struct SkewSymmetricMapping <: AbstractMappingForm end

    struct DiscretizationOperators{d}
        VOL::NTuple{d,LinearMap}
        FAC::LinearMap
        SRC::LinearMap
        M::AbstractMatrix
        V::LinearMap
        Vf::LinearMap
        scaled_normal::NTuple{d, Vector{Float64}}
    end

    struct Solver{d,ResidualForm,PDEType}
        conservation_law::AbstractConservationLaw{d,PDEType}
        operators::Vector{<:DiscretizationOperators}
        x_q::NTuple{d,Matrix{Float64}}
        connectivity::Matrix{Int}
        form::ResidualForm
        strategy::AbstractStrategy
    end

    function get_dof(spatial_discretization::SpatialDiscretization{d}, 
        conservation_law::AbstractConservationLaw{d}) where {d}
        return (spatial_discretization.reference_approximation.N_p, 
            conservation_law.N_eq, spatial_discretization.N_el)
    end

    function CLOUD_reset_timer!()
        for t in 1:Threads.nthreads()
            thread_timer = get_timer(string("thread_timer_",t))
            reset_timer!(thread_timer)
        end
    end
    
    function CLOUD_print_timer()
        for t in 1:Threads.nthreads()
            thread_timer = get_timer(string("thread_timer_",t))
            print_timer(thread_timer, title=string("Thread ", t))
        end
    end

    @inline function thread_timer()
        return get_timer(string("thread_timer_", Threads.threadid()))
    end
        
    include("reference_operator_strategy.jl")

    export combine, precompute
    include("physical_operator_strategy.jl")

    export initialize, semidiscretize
    include("preprocessing.jl")

    export StrongConservationForm, WeakConservationForm
    include("conservation_form.jl")

    export LinearResidual
    include("linear.jl")

end