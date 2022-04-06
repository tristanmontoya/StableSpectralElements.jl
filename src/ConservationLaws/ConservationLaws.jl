module ConservationLaws

    import ..ParametrizedFunctions: AbstractParametrizedFunction

    export AbstractFirstOrderFlux, AbstractSecondOrderFlux, AbstractFirstOrderNumericalFlux, AbstractSecondOrderNumericalFlux,AbstractTwoPointFlux, ConservationLaw, LaxFriedrichsNumericalFlux, EntropyConservativeNumericalFlux, ConservativeFlux, EntropyConservativeFlux, physical_flux, numerical_flux, two_point_flux

    abstract type AbstractFirstOrderFlux{d, N_eq} end
    abstract type AbstractSecondOrderFlux{d, N_eq} end
    abstract type AbstractFirstOrderNumericalFlux{FluxType} end
    abstract type AbstractSecondOrderNumericalFlux{FluxType} end
    abstract type AbstractTwoPointFlux{FluxType} end
    
    struct ConservationLaw{d, N_eq}
        first_order_flux::Union{AbstractFirstOrderFlux{d,N_eq},Nothing}
        second_order_flux::Union{AbstractSecondOrderFlux{d,N_eq},Nothing}
        first_order_numerical_flux::Union{AbstractFirstOrderNumericalFlux, Nothing}
        second_order_numerical_flux::Union{AbstractSecondOrderNumericalFlux,Nothing}
        source_term::Union{AbstractParametrizedFunction{d}, Nothing}
        two_point_flux::Union{AbstractTwoPointFlux, Nothing}
    end

    struct LaxFriedrichsNumericalFlux{FluxType} <: AbstractFirstOrderNumericalFlux{FluxType} 
        λ::Float64
    end

    struct EntropyConservativeNumericalFlux{FluxType} <: AbstractFirstOrderNumericalFlux{FluxType} end

    struct ConservativeFlux{FluxType} <: AbstractTwoPointFlux{FluxType} end
    struct EntropyConservativeFlux{FluxType} <: AbstractTwoPointFlux{FluxType} end
    
    export ConstantLinearAdvectionFlux, ConstantLinearAdvectionNumericalFlux, linear_advection_equation
    include("linear_advection.jl")

    export BurgersFlux, burgers_equation, burgers_central_flux, burgers_lax_friedrichs_flux
    include("burgers.jl")

end