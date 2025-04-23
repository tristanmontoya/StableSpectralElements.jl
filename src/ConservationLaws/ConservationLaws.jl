module ConservationLaws

using LinearMaps: LinearMap
using MuladdMacro
using StaticArrays: SVector, SMatrix, MVector
using LinearAlgebra: mul!, I
using ..GridFunctions

import ..GridFunctions: evaluate

export physical_flux,
       physical_flux!,
       numerical_flux!,
       entropy,
       conservative_to_entropy!,
       entropy_to_conservative!,
       compute_two_point_flux,
       wave_speed,
       logmean,
       inv_logmean,
       AbstractConservationLaw,
       AbstractPDEType,
       FirstOrder,
       SecondOrder,
       AbstractInviscidNumericalFlux,
       AbstractViscousNumericalFlux,
       NoInviscidFlux,
       NoViscousFlux,
       LaxFriedrichsNumericalFlux,
       CentralNumericalFlux,
       BR1,
       EntropyConservativeNumericalFlux,
       AbstractTwoPointFlux,
       ConservativeFlux,
       EntropyConservativeFlux,
       NoTwoPointFlux,
       ExactSolution

@doc raw"""
    AbstractConservationLaw{d, PDEType, N_c}

Abstract type for a conservation law, where `d` is the number of spatial dimensions,
`PDEType <: AbstractPDEType` is either `FirstOrder` or `SecondOrder`, and `N_c` is the number of conservative
variables.
"""
abstract type AbstractConservationLaw{d, PDEType, N_c} end
abstract type AbstractPDEType end
struct FirstOrder <: AbstractPDEType end # PDE with only first derivatives
struct SecondOrder <: AbstractPDEType end # PDE with first and second derivatives

# First-order numerical fluxes
abstract type AbstractInviscidNumericalFlux end
struct NoInviscidFlux <: AbstractInviscidNumericalFlux end
struct LaxFriedrichsNumericalFlux <: AbstractInviscidNumericalFlux
    halfλ::Float64
    function LaxFriedrichsNumericalFlux(λ::Float64)
        return new(0.5 * λ)
    end
end
struct EntropyConservativeNumericalFlux <: AbstractInviscidNumericalFlux end
struct CentralNumericalFlux <: AbstractInviscidNumericalFlux end
LaxFriedrichsNumericalFlux() = LaxFriedrichsNumericalFlux(1.0)

# Second-order numerical fluxes
abstract type AbstractViscousNumericalFlux end
struct BR1 <: AbstractViscousNumericalFlux end
struct NoViscousFlux <: AbstractViscousNumericalFlux end

# Two-point fluxes (for split forms and entropy-stable schemes)
abstract type AbstractTwoPointFlux end
struct ConservativeFlux <: AbstractTwoPointFlux end
struct EntropyConservativeFlux <: AbstractTwoPointFlux end
struct NoTwoPointFlux <: AbstractTwoPointFlux end

@inline @views function numerical_flux!(f_star::AbstractMatrix{Float64},
        conservation_law::AbstractConservationLaw{d,
            PDEType,
            N_c},
        numerical_flux::LaxFriedrichsNumericalFlux,
        u_in::AbstractMatrix{Float64},
        u_out::AbstractMatrix{Float64},
        n_f::AbstractMatrix{Float64},
        two_point_flux = ConservativeFlux()) where {d,
        PDEType,
        N_c}
    @inbounds for i in axes(u_in, 1)
        f_s = compute_two_point_flux(conservation_law,
            two_point_flux,
            u_in[i, :],
            u_out[i, :])
        a = numerical_flux.halfλ *
            wave_speed(conservation_law, u_in[i, :], u_out[i, :], n_f[:, i])
        for e in 1:N_c
            f_n_avg = 0.0
            for m in 1:d
                @muladd f_n_avg = f_n_avg + f_s[e, m] * n_f[m, i]
            end
            @muladd f_star[i, e] = f_n_avg + a * (u_in[i, e] - u_out[i, e])
        end
    end
end

@inline @views function numerical_flux!(f_star::AbstractMatrix{Float64},
        conservation_law::AbstractConservationLaw{d,
            PDEType,
            N_c},
        ::Union{CentralNumericalFlux,
            EntropyConservativeNumericalFlux},
        u_in::AbstractMatrix{Float64},
        u_out::AbstractMatrix{Float64},
        n_f::AbstractMatrix{Float64},
        two_point_flux = ConservativeFlux()) where {d,
        PDEType,
        N_c}
    @inbounds for i in axes(u_in, 1)
        f_s = compute_two_point_flux(conservation_law,
            two_point_flux,
            u_in[i, :],
            u_out[i, :])
        for e in 1:N_c
            temp = 0.0
            for m in 1:d
                @muladd temp = temp + f_s[e, m] * n_f[m, i]
            end
            f_star[i, e] = temp
        end
    end
end

# Algorithm based on the Taylor series trick from Ismail and Roe (2009). There are further 
# optimizations that could be made, but let's leave it like this for now.
@inline function logmean(x::Float64, y::Float64)
    # f = (y/x - 1) / (y/x + 1)
    #   = (y - x) / (x + y)
    # rearrange to avoid divisions using trick from 
    # https://trixi-framework.github.io/Trixi.jl/stable/reference-trixi/#Trixi.ln_mean
    f² = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y)
    if f² < 1.0e-4
        return (x + y) * 105 / (210 + f² * (70 + f² * (42 + f² * 30)))
        # faster factorized way to compute
        # (x + y) / (2 + 2/3 * f^2 + 2/5 * f^4 + 2/7 * f^6)
    else
        return (y - x) / log(y / x)
    end
end

@inline function inv_logmean(x::Float64, y::Float64)
    f² = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y)
    if f² < 1.0e-4
        return (210 + f² * (70 + f² * (42 + f² * 30))) / ((x + y) * 105)
        # faster factorized way to compute
        # (x + y) / (2 + 2/3 * f^2 + 2/5 * f^4 + 2/7 * f^6)
    else
        return log(y / x) / (y - x)
    end
end

# Generic structure for exact solution to PDE (may be deprecated in future versions)
struct ExactSolution{d, ConservationLaw, InitialData, SourceTerm} <: AbstractGridFunction{d}
    conservation_law::ConservationLaw
    initial_data::InitialData
    periodic::Bool
    N_c::Int

    function ExactSolution(conservation_law::AbstractConservationLaw{d, PDEType, N_c},
            initial_data::AbstractGridFunction{d};
            periodic::Bool = false) where {d, PDEType, N_c}
        return new{d,
            typeof(conservation_law),
            typeof(initial_data),
            typeof(conservation_law.source_term)}(conservation_law,
            initial_data,
            periodic,
            N_c)
    end
end

@inbounds function entropy_to_conservative!(u::AbstractVector{Float64},
        ::AbstractConservationLaw,
        w::AbstractVector{Float64})
    copyto!(u, w)
    return
end

@inbounds function conservative_to_entropy(w::AbstractVector{Float64},
        ::AbstractConservationLaw,
        u::AbstractVector{Float64})
    copyto!(w, u)
    return
end

export LinearAdvectionEquation, LinearAdvectionDiffusionEquation
include("linear_advection_diffusion.jl")

export InviscidBurgersEquation, ViscousBurgersEquation
include("burgers.jl")

export EulerEquations,
       NavierStokesEquations,
       EulerPeriodicTest,
       TaylorGreenVortex,
       IsentropicVortex,
       KelvinHelmholtzInstability
include("euler_navierstokes.jl")

end
