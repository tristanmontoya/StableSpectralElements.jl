module ConservationLaws

    using LinearMaps: LinearMap
    using LinearAlgebra: mul!, I
    using UnPack

    import ..ParametrizedFunctions: AbstractParametrizedFunction, NoSourceTerm, InitialDataGaussian, InitialDataGassner, SourceTermGassner, evaluate


    export AbstractConservationLaw, AbstractPDEType, Parabolic, Hyperbolic, Mixed, AbstractInviscidNumericalFlux, AbstractViscousNumericalFlux, NoInviscidFlux, NoViscousFlux, LaxFriedrichsNumericalFlux, BR1, EntropyConservativeNumericalFlux, AbstractTwoPointFlux, EntropyConservativeFlux, NoTwoPointFlux, num_equations

    abstract type AbstractConservationLaw{d, PDEType} end
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
    abstract type AbstractInviscidNumericalFlux end
    struct NoInviscidFlux <: AbstractInviscidNumericalFlux end
    struct LaxFriedrichsNumericalFlux <: AbstractInviscidNumericalFlux 
        λ::Float64
    end
    struct EntropyConservativeNumericalFlux <: AbstractInviscidNumericalFlux end
    LaxFriedrichsNumericalFlux() = LaxFriedrichsNumericalFlux(1.0)
    
    """Second-order numerical fluxes"""
    abstract type AbstractViscousNumericalFlux end
    struct BR1 <: AbstractViscousNumericalFlux end
    struct NoViscousFlux <: AbstractViscousNumericalFlux end

    """Two-point fluxes (for split forms and entropy-stable schemes)"""
    abstract type AbstractTwoPointFlux end
    struct ConservativeFlux <: AbstractTwoPointFlux end
    struct EntropyConservativeFlux <: AbstractTwoPointFlux end
    struct NoTwoPointFlux <: AbstractTwoPointFlux end

    export LinearAdvectionEquation, LinearAdvectionDiffusionEquation, DiffusionSolution
    include("linear_advection_diffusion.jl")

    export InviscidBurgersEquation, ViscousBurgersEquation, BurgersSolution
    include("burgers.jl")

    export EulerEquations, NavierStokesEquations, EntropyWave1D
    include("euler_navierstokes.jl")

end