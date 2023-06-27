@doc raw"""
    EulerEquations{d}(γ::Float64) where {d}

Define an Euler system governing compressible, adiabatic fluid flow, taking the form
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

struct EulerPeriodicTest{d} <: AbstractGridFunction{d}
    γ::Float64
    ϵ::Float64
    N_c::Int

    function EulerPeriodicTest{d}(γ::Float64,
        ϵ::Float64) where {d}
        return new{d}(γ,ϵ,d+2)
    end
end

struct TaylorGreenVortex <: AbstractGridFunction{3}
    Ma::Float64
    N_c::Int

    function TaylorGreenVortex(Ma::Float64)
        return new(Ma,5)
    end
end

struct IsentropicVortex <: AbstractGridFunction{2}
    γ::Float64
    Ma::Float64
    ϵ::Float64
    x_0::NTuple{2,Float64}
    N_c::Int

    function IsentropicVortex(γ::Float64,
        Ma::Float64,
        ϵ::Float64,
        x_0::NTuple{2,Float64})
        return new(γ, Ma, ϵ, x_0, 4)
    end
end

@inline function pressure(conservation_law::EulerType{d}, 
    u::AbstractMatrix{Float64}) where {d}
    (; γ) = conservation_law
    ρ = @view u[:,1]
    ρV = @view u[:,2:end-1]
    E = @view u[:,end]
    return (γ-1) .* (E .- (0.5./ρ) .* (sum(ρV[:,m].^2 for m in 1:d)))
end

@inline function velocity(conservation_law::EulerType{d}, 
    u::AbstractMatrix{Float64}) where {d}
    (; γ) = conservation_law
    ρ = @view u[:,1]
    ρV = @view u[:,2:end-1]
    return hcat([ρV[:,m] ./ ρ for m in 1:d]...)
end

"""
Evaluate the flux for the Euler equations
"""
@inline function physical_flux!(f::AbstractArray{Float64,3},    
    conservation_law::EulerType{d}, 
    u::AbstractMatrix{Float64}) where {d}

    (; γ) = conservation_law
    ρV = @view u[:,2:end-1]
    E = @view u[:,end]
    V = velocity(conservation_law, u)
    p = pressure(conservation_law, u)
    
    @inbounds for m in 1:d 
        f[:,1,m] .= ρV[:,m]
        f[:,2:end-1,m] .= hcat([ρV[:,m].*V[:,n] .+ I[m,n]*p for n in 1:d]...)
        f[:,end,m] .= V[:,m] .* (E .+ p)
    end
end

"""Lax-Friedrichs/Rusanov flux for the Euler equations"""
@inline function numerical_flux(
    conservation_law::EulerType{d},
    numerical_flux::LaxFriedrichsNumericalFlux, 
    u_in::AbstractMatrix{Float64}, u_out::AbstractMatrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    (; γ, N_c) = conservation_law
    (; λ) = numerical_flux

    ρ_in = @view u_in[:,1]
    V_in = velocity(conservation_law, u_in)
    p_in = pressure(conservation_law, u_in)
    f_in = Array{Float64}(undef, size(u_in)..., d)
    physical_flux!(f_in, conservation_law, u_in)

    ρ_out = @view u_out[:,1]
    V_out = velocity(conservation_law, u_out)
    p_out = pressure(conservation_law, u_out)
    f_out = Array{Float64}(undef, size(u_out)..., d)
    physical_flux!(f_out, conservation_law, u_out)

    Vn_in = sum(V_in[:,m] .* n[m] for m in 1:d)
    Vn_out = sum(V_out[:,m] .* n[m] for m in 1:d)
    c_in = sqrt.(abs.(γ*p_in ./ ρ_in))
    c_out = sqrt.(abs.(γ*p_out ./ ρ_out))

    a = max.(abs.(Vn_in), abs.(Vn_out)) .+ max.(c_in, c_out)

    return 0.5*(hcat([sum((f_in[:,e,m] .+ f_out[:,e,m]) .* n[m] for m in 1:d)
            for e in 1:N_c]...) .- λ*a.*(u_out .- u_in))
end

function evaluate(f::EulerPeriodicTest{d}, 
    x::NTuple{d,Float64},t::Float64=0.0) where {d}

    ρ = 1.0 + f.ϵ*sin(π*sum(x[m] for m in 1:d))

    return [ρ, fill(ρ,d)..., 1.0/(1.0-f.γ) + 0.5*ρ*d]
end

function evaluate(f::TaylorGreenVortex,  x::NTuple{3,Float64}, t::Float64=0.0)

    p = 1.0/(f.γ * f.Ma^2) + 
        1.0/16 * (2*cos(2*x[1]) + 2*cos(2*x[2]) +
        cos(2*x[1])*cos(2*x[3]) + cos(2*x[2])*cos(2*x[3]))
    u = sin(x[1])*cos(x[2])*cos(x[3])
    v = -cos(x[1])*sin(x[2])*cos(x[3])

    return [1.0, u, v, 0.0,  p/(1.0-f.γ) + 0.5*ρ*(u^2 + v^2)]
end

function evaluate(f::IsentropicVortex, x::NTuple{2,Float64}, t::Float64=0.0)
    fx = 1.0 - (x[1] - f.x_0[1] - t)^2 - (x[2] - f.x_0[2])^2
    u = 1.0 - (f.ϵ*(x[2] - f.x_0[2]))/(2.0*π)*exp(0.5*fx)
    v = (f.ϵ*(x[1] - f.x_0[1] - t))/(2.0*π)*exp(0.5*fx)
    ρ = (1.0 - ((f.ϵ)^2*(f.γ-1.0)*(f.Ma)^2/(8.0*π^2) * exp(fx)))^(1.0/(1.0-f.γ))
    p = ρ^(f.γ)/(f.γ * f.Ma^2)

    return [ρ, ρ*u, ρ*v, p/(1.0-f.γ) + 0.5*ρ*(u^2 + v^2)]
end

function evaluate(
    exact_solution::ExactSolution{d,EulerEquations{d}, EulerPeriodicTest{d},NoSourceTerm{d}},
    x::NTuple{d,Float64},t::Float64=0.0) where {d}
    (; initial_data, conservation_law) = exact_solution
    
    if !exact_solution.periodic
        ρ = 1.0 + initial_data.ϵ*sin(π*(sum(x[m] for m in 1:d) - t*d))
        return [ρ, [ρ for m in 1:d]..., 1.0/(1.0-conservation_law.γ) + 0.5*ρ*d]
    else
       return evaluate(initial_data,x)
    end
end

function evaluate(
    exact_solution::ExactSolution{2,EulerEquations{2}, IsentropicVortex,NoSourceTerm{2}},
    x::NTuple{2,Float64},t::Float64=0.0)
    (; initial_data, conservation_law) = exact_solution
    
    if !exact_solution.periodic
        return evaluate(initial_data, 
            (x[1] - initial_data.Ma*cos(initial_data.θ)*t,
            (x[2] - initial_data.Ma*sin(initial_data.θ))*t ))
    else
       return evaluate(initial_data,x)
    end
end