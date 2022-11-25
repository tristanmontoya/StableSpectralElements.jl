@doc raw"""
    EulerEquations{d}(γ::Float64) where {d}

Define an Euler system of the form
```math
\frac{\partial}{\partial t}\left[\begin{array}{c}
\rho(\bm{x}, t) \\
\rho(\bm{x}, t) V_1(\bm{x}, t) \\
\vdots \\
\rho(\bm{x}, t) V_d(\bm{x}, t) \\
E(\bm{x}, t)
\end{array}\right]+\sum_{m=1}^d \frac{\partial}{\partial x_m}\left[\begin{array}{c}
\rho(\bm{x}, t) V_m(\bm{x}, t) \\
\rho(\bm{x}, t) V_1(\bm{x}, t) V_m(\bm{x}, t)+P(\bm{x}, t) \delta_{1 m} \\
\vdots \\
\rho(\bm{x}, t) V_d(\bm{x}, t) V_m(\bm{x}, t)+P(\bm{x}, t) \delta_{d m} \\
V_m(\bm{x}, t)(E(\bm{x}, t)+P(\bm{x}, t))
\end{array}\right]=\underline{0},
```
where $\rho(\bm{x},t) \in \mathbb{R}$ is the fluid density, $\bm{V}(\bm{x},t) \in \mathbb{R}^d$ is the flow velocity, $E(\bm{x},t) \in \mathbb{R}$ is the total energy per unit volume, and the pressure is given for an ideal gas with constant specific heat as
```math
P(\bm{x},t) = (\gamma - 1)\Big(E(\bm{x},t) - \frac{1}{2}\rho(\bm{x},t) \lVert \bm{V}(\bm{x},t)\rVert^2\Big).
```
The specific heat ratio is specified as a parameter `γ::Float64`, which must be greater than unity.
"""
struct EulerEquations{d} <: AbstractConservationLaw{d,FirstOrder}
    γ::Float64
    source_term::AbstractGridFunction{d}
    N_c::Int

    function EulerEquations{d}(γ::Float64, 
        source_term::AbstractGridFunction{d}) where {d}
        return new{d}(γ,source_term, d+2)
    end
end

struct NavierStokesEquations{d} <: AbstractConservationLaw{d,SecondOrder} end

const EulerType{d} = Union{EulerEquations{d}, NavierStokesEquations{d}}

function EulerEquations{d}(γ::Float64) where {d}
    return EulerEquations{d}(γ,NoSourceTerm{d}())
end
struct EntropyWave1D <: AbstractGridFunction{1}
    conservation_law::EulerType{1}
    ϵ::Float64
    N_c::Int
end

@inline function pressure(conservation_law::EulerType{d}, 
    u::Matrix{Float64}) where {d}
    @unpack γ = conservation_law
    ρ = @view u[:,1]
    ρV = @view  u[:,2:end-1]
    E = @view u[:,end]
    return (γ-1).*(E .- 0.5./ρ.*(sum(ρV[:,m].^2 for m in 1:d)))
end

@inline function velocity(conservation_law::EulerType{d}, 
    u::Matrix{Float64}) where {d}
    @unpack γ = conservation_law
    ρ = @view u[:,1]
    ρV = @view u[:,2:end-1]
    return hcat([ρV[:,m] ./ ρ for m in 1:d]...)
end

"""
Evaluate the flux for the Euler equations
"""
@inline function physical_flux(conservation_law::EulerType{d}, 
    u::Matrix{Float64}) where {d}

    @unpack γ = conservation_law
    ρV = @view u[:,2:end-1]
    E = @view u[:,end]
    V = velocity(conservation_law, u)
    p = pressure(conservation_law, u)
    
    # evaluate flux
    return Tuple(hcat(ρV[:,m], 
        [ρV[:,m].*V[:,n] .+ I[m,n]*p for n in 1:d]...,
        V[:,m].*(E .+ p))  for m in 1:d)
end

"""Lax-Friedrichs/Rusanov flux for the Euler equations"""
@inline function numerical_flux(
    conservation_law::EulerType{d},
    numerical_flux::LaxFriedrichsNumericalFlux, 
    u_in::Matrix{Float64}, u_out::Matrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    @unpack γ, N_c = conservation_law

    ρ_in = @view u_in[:,1]
    V_in = velocity(conservation_law, u_in)
    p_in = pressure(conservation_law, u_in)

    ρ_out =@view u_out[:,1]
    V_out = velocity(conservation_law, u_out)
    p_out = pressure(conservation_law, u_out)

    f_in = physical_flux(conservation_law,u_in)
    f_out = physical_flux(conservation_law,u_out)

    fn_avg = 0.5*hcat([
        sum((f_in[m][:,e] + f_out[m][:,e]) .* n[m] for m in 1:d)
            for e in 1:N_c]...)

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