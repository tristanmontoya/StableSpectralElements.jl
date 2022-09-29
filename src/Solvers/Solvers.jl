module Solvers

    using UnPack
    import LinearAlgebra
    using LinearAlgebra: Diagonal, inv, mul!
    using TimerOutputs
    using LinearMaps: LinearMap
    using OrdinaryDiffEq: ODEProblem, OrdinaryDiffEqAlgorithm, solve
    

    using ..ConservationLaws: AbstractConservationLaw, AbstractPDEType, FirstOrder, SecondOrder, AbstractInviscidNumericalFlux, AbstractViscousNumericalFlux, AbstractTwoPointFlux, NoInviscidFlux, NoViscousFlux, NoTwoPointFlux, NoSourceTerm, physical_flux, numerical_flux, LaxFriedrichsNumericalFlux, BR1

    using ..SpatialDiscretizations: ReferenceApproximation, SpatialDiscretization
    using ..GridFunctions: AbstractGridFunction, AbstractGridFunction, NoSourceTerm, evaluate

    const timer = TimerOutput(); export timer
    
    macro time_first_thread(args...)
        TimerOutputs.timer_expr(__module__, false, :($timer), args...)
    end
 
    macro CLOUD_timeit(args...)
        esc(quote
            if Threads.threadid() == 1
                @time_first_thread($(args...))
            else
                esc($(last(args)))
            end
        end)
    end
    
    export AbstractResidualForm, AbstractMappingForm, AbstractStrategy, DiscretizationOperators, PhysicalOperator, ReferenceOperator, Solver, StandardMapping, SkewSymmetricMapping, apply_operators!, auxiliary_variable!, get_dof, rhs!

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