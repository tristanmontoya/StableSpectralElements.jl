module ConservationLaws

    using LinearMaps: LinearMap
    using LinearAlgebra: mul!
    using UnPack

    import ..ParametrizedFunctions: AbstractParametrizedFunction, NoSourceTerm, InitialDataGaussian, InitialDataGassner, SourceTermGassner, evaluate


    export AbstractConservationLaw, AbstractPDEType, Parabolic, Hyperbolic, Mixed, AbstractFirstOrderNumericalFlux, AbstractSecondOrderNumericalFlux, NoFirstOrderFlux, NoSecondOrderFlux, LaxFriedrichsNumericalFlux, BR1, EntropyConservativeNumericalFlux, AbstractTwoPointFlux, EntropyConservativeFlux, NoTwoPointFlux

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


    """First-order numerical fluxes"""
    abstract type AbstractFirstOrderNumericalFlux end
    struct NoFirstOrderFlux <: AbstractFirstOrderNumericalFlux end
    struct LaxFriedrichsNumericalFlux <: AbstractFirstOrderNumericalFlux 
        λ::Float64
    end
    struct EntropyConservativeNumericalFlux <: AbstractFirstOrderNumericalFlux end
    LaxFriedrichsNumericalFlux() = LaxFriedrichsNumericalFlux(1.0)
    
    """Second-order numerical fluxes"""
    abstract type AbstractSecondOrderNumericalFlux end
    struct BR1 <: AbstractSecondOrderNumericalFlux end
    struct NoSecondOrderFlux <: AbstractSecondOrderNumericalFlux end


    """Two-point fluxes (for split forms and entropy-stable schemes)"""
    abstract type AbstractTwoPointFlux end
    struct ConservativeFlux <: AbstractTwoPointFlux end
    struct EntropyConservativeFlux <: AbstractTwoPointFlux end
    struct NoTwoPointFlux <: AbstractTwoPointFlux end
    
    export LinearAdvectionEquation, LinearAdvectionDiffusionEquation, DiffusionSolution
    include("linear_advection_diffusion.jl")

    export InviscidBurgersEquation, ViscousBurgersEquation, BurgersSolution
    include("burgers.jl")

end