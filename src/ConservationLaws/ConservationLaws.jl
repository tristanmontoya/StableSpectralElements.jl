module ConservationLaws

    using LinearMaps: LinearMap
    using LinearAlgebra: mul!, I
    using UnPack

    import ..GridFunctions: AbstractGridFunction, NoSourceTerm, InitialDataSine, InitialDataGaussian, InitialDataGassner, SourceTermGassner, evaluate


    export AbstractConservationLaw, AbstractPDEType, FirstOrder, SecondOrder, AbstractInviscidNumericalFlux, AbstractViscousNumericalFlux, NoInviscidFlux, NoViscousFlux, LaxFriedrichsNumericalFlux, BR1, EntropyConservativeNumericalFlux, AbstractTwoPointFlux, EntropyConservativeFlux, NoTwoPointFlux, ExactSolution

    abstract type AbstractConservationLaw{d, PDEType} end
    abstract type AbstractPDEType end

    """
    First-order conservation law:

    `∂ₜu + ∇⋅F(u) = s`
    """
    struct FirstOrder <: AbstractPDEType end
    
    """
    Second-order conservation law:

    `∂ₜu + ∇⋅(F¹(u) + F²(u,q)) = s, q = ∇u`
    """
    struct SecondOrder <: AbstractPDEType end

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

    """Generic structure for exact solution to PDE"""
    struct ExactSolution{d,ConservationLaw,InitialData,SourceTerm} <: AbstractGridFunction{d}
        conservation_law::ConservationLaw
        initial_data::InitialData
        periodic::Bool
        N_c::Int

        function ExactSolution(
            conservation_law::AbstractConservationLaw{d,PDEType},
            initial_data::AbstractGridFunction{d};
            periodic::Bool=false) where {d, PDEType}

            return new{d,typeof(conservation_law),typeof(initial_data),typeof(conservation_law.source_term)}(
                conservation_law,
                initial_data,
                periodic,
                conservation_law.N_c)
        end
    end

    export LinearAdvectionEquation, LinearAdvectionDiffusionEquation
    include("linear_advection_diffusion.jl")

    export InviscidBurgersEquation, ViscousBurgersEquation
    include("burgers.jl")

    export EulerEquations, NavierStokesEquations, EntropyWave1D
    include("euler_navierstokes.jl")

end