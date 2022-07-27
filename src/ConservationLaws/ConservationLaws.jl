module ConservationLaws

    using LinearMaps: LinearMap
    using LinearAlgebra: mul!

    import ..ParametrizedFunctions: AbstractParametrizedFunction

    export AbstractConservationLaw, AbstractPDEType, Parabolic, Hyperbolic, Mixed, AbstractFirstOrderNumericalFlux, AbstractSecondOrderNumericalFlux, NoFirstOrderFlux, NoSecondOrderFlux, LaxFriedrichsNumericalFlux, EntropyConservativeNumericalFlux, AbstractTwoPointFlux, EntropyConservativeFlux

    abstract type AbstractConservationLaw{d, N_eq, PDEType} end
    abstract type AbstractPDEType end

    """
    Hyperbolic conservation law:

    ∂U/∂t + ∇⋅F(U) = 0
    """
    struct Hyperbolic <: AbstractPDEType end
        
    """
    Parabolic conservation law:

    ∂U/∂t + ∇⋅F(U,Q) = 0
    Q = ∇U 
    """
    struct Parabolic <: AbstractPDEType end

    """
    Mixed hyperbolic-parabolic conservation law:

    ∂U/∂t + ∇⋅F¹(U) + ∇⋅F²(U,Q) = 0
    Q = ∇U 
    """
    struct Mixed <: AbstractPDEType end

    """Numerical interface fluxes"""
    abstract type AbstractFirstOrderNumericalFlux end
    abstract type AbstractSecondOrderNumericalFlux end
    struct NoFirstOrderFlux <: AbstractFirstOrderNumericalFlux end
    struct NoSecondOrderFlux <: AbstractSecondOrderNumericalFlux end
    struct LaxFriedrichsNumericalFlux <: AbstractFirstOrderNumericalFlux 
        λ::Float64
    end

    struct EntropyConservativeNumericalFlux <: AbstractFirstOrderNumericalFlux end

    """Two-point fluxes"""
    abstract type AbstractTwoPointFlux{d} end
    struct ConservativeFlux{d} <: AbstractTwoPointFlux{d} end
    struct EntropyConservativeFlux{d} <: AbstractTwoPointFlux{d} end
    
    export LinearAdvectionEquation, LinearAdvectionDiffusionEquation,LinearAdvectionNumericalFlux, BR1
    include("linear_advection_diffusion.jl")

    #TODO add back burgers
    #export BurgersFlux, burgers_equation, burgers_central_flux, burgers_lax_friedrichs_flux
    #include("burgers.jl")

end