module ConservationLaws

    using LinearMaps: LinearMap
    using StaticArrays: SVector, SMatrix, MVector
    using LinearAlgebra: mul!, I
    import ..GridFunctions: AbstractGridFunction, NoSourceTerm, InitialDataSine, InitialDataGaussian, InitialDataGassner, SourceTermGassner, evaluate

    export physical_flux!, numerical_flux!, entropy, conservative_to_primitive, conservative_to_entropy, entropy_to_conservative, compute_two_point_flux, wave_speed, logmean, AbstractConservationLaw, AbstractPDEType, FirstOrder, SecondOrder, AbstractInviscidNumericalFlux, AbstractViscousNumericalFlux, NoInviscidFlux, NoViscousFlux, LaxFriedrichsNumericalFlux, RoeNumericalFlux, BR1, EntropyConservativeNumericalFlux, AbstractTwoPointFlux, EntropyConservativeFlux, NoTwoPointFlux, ExactSolution

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
    struct RoeNumericalFlux <: AbstractInviscidNumericalFlux end
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

    function numerical_flux!(
        f_star::AbstractMatrix{Float64},
        conservation_law::AbstractConservationLaw{d}, 
        numerical_flux::LaxFriedrichsNumericalFlux,
        u_in::AbstractMatrix{Float64}, u_out::AbstractMatrix{Float64}, 
        n::NTuple{d, Vector{Float64}},
        two_point_flux::AbstractTwoPointFlux=ConservativeFlux()) where {d}
        
        @views for i in axes(u_in, 1)
            f_s = compute_two_point_flux(conservation_law, two_point_flux,
                u_in[i,:], u_out[i,:])
            f_star[i,:] .= sum(f_s[:,m]*n[m][i] for m in 1:d) .- 
                numerical_flux.λ*0.5 * 
                wave_speed(conservation_law, u_in[i,:], u_out[i,:], 
                    Tuple(n[m][i] for m in 1:d)) .* 
                    (u_out[i,:] .- u_in[i,:])
        end
    end

    function numerical_flux!(
        f_star::AbstractMatrix{Float64},
        conservation_law::AbstractConservationLaw{d}, 
        ::EntropyConservativeNumericalFlux,
        u_in::AbstractMatrix{Float64}, u_out::AbstractMatrix{Float64}, 
        n::NTuple{d, Vector{Float64}},
        two_point_flux::AbstractTwoPointFlux=ConservativeFlux()) where {d}
        
        @views for i in axes(u_in, 1)
            f_s = compute_two_point_flux(conservation_law, two_point_flux,
                u_in[i,:], u_out[i,:])
            f_star[i,:] .= sum(f_s[:,m]*n[m][i] for m in 1:d)
        end
    end

    """ 
        Algorithm based on the Taylor series trick from Ismail and Roe (2009). There are further optimizations that could be made, but let's leave it like this for now.
    """
    @inline function logmean(x::Float64, y::Float64)
        # f = (y/x - 1) / (y/x + 1)
        #   = (y - x) / (x + y)
        # rearrange to avoid divisions using trick from https://trixi-framework.github.io/Trixi.jl/stable/reference-trixi/#Trixi.ln_mean
        f² = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y)
        if f² < 1.0e-4
            return (x + y) * 105 / (210 + f² * (70 + f² * (42 + f² * 30)))
            # faster factorized way to compute
            # (x + y) / (2 + 2/3 * f^2 + 2/5 * f^4 + 2/7 * f^6)
            # see https://trixi-framework.github.io/Trixi.jl/stable/reference-trixi/#Trixi.ln_mean
        else
            return (y - x) / log(y/x)
        end
      end

    """Generic structure for exact solution to PDE (may be deprecated in future versions)"""
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
                conservation_law, initial_data,periodic,conservation_law.N_c)
        end
    end

    export LinearAdvectionEquation, LinearAdvectionDiffusionEquation
    include("linear_advection_diffusion.jl")

    export InviscidBurgersEquation, ViscousBurgersEquation
    include("burgers.jl")

    export EulerEquations, NavierStokesEquations, EulerPeriodicTest, TaylorGreenVortex, IsentropicVortex
    include("euler_navierstokes.jl")

end