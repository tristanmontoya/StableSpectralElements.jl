"""
Euler equations
"""
struct EulerEquations{d} <: AbstractConservationLaw{d,Hyperbolic}
    γ::Float64
end

struct NavierStokesEquations{d} <: AbstractConservationLaw{d,Mixed} end

const EulerType{d} = Union{EulerEquations{d}, NavierStokesEquations{d}}

struct EntropyWave1D <: AbstractParametrizedFunction{1}
    conservation_law::EulerType{1}
    ϵ::Float64
    N_eq::Int
end

function num_equations(::EulerEquations{d}) where {d}
    return d+2 
end
 
function pressure(conservation_law::EulerType{d}, 
    u::Matrix{Float64}) where {d}
    @unpack γ = conservation_law
    ρ = u[:,1]
    ρV = u[:,2:end-1]
    E = u[:,end]
    return (γ-1).*(E .- 0.5./ρ.*(sum(ρV[:,m].^2 for m in 1:d)))
end

function velocity(conservation_law::EulerType{d}, 
    u::Matrix{Float64}) where {d}
    @unpack γ = conservation_law
    ρ = u[:,1]
    ρV = u[:,2:end-1]
    return hcat([ρV[:,m] ./ ρ for m in 1:d]...)
end

"""
Evaluate the flux for the Euler equations
"""
@inline function physical_flux(conservation_law::EulerType{d}, 
    u::Matrix{Float64}) where {d}

    @unpack γ = conservation_law
    ρV = u[:,2:end-1]
    E = u[:,end]
    V = velocity(conservation_law, u)
    p = pressure(conservation_law, u)
    
    # evaluate flux
    return Tuple(hcat(ρV[:,m], 
        [ρV[:,m].*V[:,n] .+ I[m,n]*p for n in 1:d]...,
        V[:,m].*(E .+ p))  for m in 1:d)
end

"""Lax-Friedrichs/Rusanov flux for the Euler equations"""
function numerical_flux(
    conservation_law::AbstractConservationLaw{d},
    numerical_flux::LaxFriedrichsNumericalFlux, 
    u_in::Matrix{Float64}, u_out::Matrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    @unpack γ = conservation_law

    ρ_in = u_in[:,1]
    V_in = velocity(conservation_law, u_in)
    p_in = pressure(conservation_law, u_in)

    ρ_out = u_out[:,1]
    V_out = velocity(conservation_law, u_out)
    p_out = pressure(conservation_law, u_out)

    f_in = physical_flux(conservation_law,u_in)
    f_out = physical_flux(conservation_law,u_out)

    fn_avg = 0.5*hcat([
        sum((f_in[m][:,e] + f_out[m][:,e]) .* n[m] for m in 1:d)
            for e in 1:num_equations(conservation_law)]...)

    a = sqrt.(sum(n[m].^2 for m in 1:d)) .*
        max.(abs.(sum(V_in[:,m].^2 for m in 1:d) 
            + sqrt.(abs.(γ*p_in./ρ_in))), 
        abs.(sum(V_out[:,m].^2 for m in 1:d) 
            + sqrt.(abs.(γ*p_out./ρ_out))))

    return fn_avg - numerical_flux.λ*0.5*a.*(u_out - u_in)
end

function evaluate(f::EntropyWave1D, 
    x::NTuple{1,Float64},t::Float64=0.0)
    ρ = 2.0 + f.ϵ*sin(π*x[1])
    u = 1.0
    p = 1.0

    return [ρ, ρ*u, 
        p/(1-f.conservation_law.γ) + 0.5*ρ*u^2]
end