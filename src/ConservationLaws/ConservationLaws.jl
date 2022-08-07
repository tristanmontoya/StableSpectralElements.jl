module ConservationLaws

    using LinearMaps: LinearMap
    using LinearAlgebra: mul!
    using UnPack

    import ..ParametrizedFunctions: AbstractParametrizedFunction, NoSourceTerm, InitialDataGaussian, evaluate


    export AbstractConservationLaw, AbstractPDEType, Parabolic, Hyperbolic, Mixed, AbstractFirstOrderNumericalFlux, AbstractSecondOrderNumericalFlux, NoFirstOrderFlux, NoSecondOrderFlux, LaxFriedrichsNumericalFlux, EntropyConservativeNumericalFlux, AbstractTwoPointFlux, EntropyConservativeFlux, NoTwoPointFlux

    abstract type AbstractConservationLaw{d, N_eq, PDEType} end
    abstract type AbstractPDEType end

    """
    Hyperbolic conservation law:

    `∂ₜu + ∇⋅F(u) = s`
    """
    struct Hyperbolic <: AbstractPDEType end
        
    """
    Parabolic conservation law:

    `∂ₜu + ∇⋅F(u,q) = s, q = ∇u` 
    """
    struct Parabolic <: AbstractPDEType end

    """
    Mixed hyperbolic-parabolic conservation law:

    `∂ₜu + ∇⋅(F¹(u) + F²(u,q)) = s, q = ∇u`
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
    abstract type AbstractTwoPointFlux end
    struct ConservativeFlux <: AbstractTwoPointFlux end
    struct EntropyConservativeFlux <: AbstractTwoPointFlux end
    struct NoTwoPointFlux <: AbstractTwoPointFlux end
    
    export LinearAdvectionEquation, LinearAdvectionDiffusionEquation,LinearAdvectionNumericalFlux, BR1, DiffusionSolution
    include("linear_advection_diffusion.jl")

    #TODO add back burgers
    #export BurgersFlux, burgers_equation, burgers_central_flux, burgers_lax_friedrichs_flux
    #include("burgers.jl")

end