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
struct EulerEquations{d,N_c} <: AbstractConservationLaw{d,FirstOrder,N_c}
    γ::Float64
    source_term::AbstractGridFunction{d}

    function EulerEquations{d}(γ::Float64=1.4, 
        source_term::AbstractGridFunction{d}=NoSourceTerm{d}()) where {d}
        return new{d,d+2}(γ,source_term)
    end
end

struct NavierStokesEquations{d,N_c} <: AbstractConservationLaw{d,SecondOrder,N_c} 
    γ::Float64
    source_term::AbstractGridFunction{d}

    function NavierStokesEquations{d}(γ::Float64=1.4, 
        source_term::AbstractGridFunction{d}=NoSourceTerm{d}()) where {d}
        @error "Navier-Stokes not implemented."
        return new{d,N_c}(γ,source_term)
    end
end

const EulerType{d} = Union{EulerEquations{d}, NavierStokesEquations{d}}

"""
Evaluate the flux for the Euler equations
"""
@inline function physical_flux(conservation_law::EulerType{d}, 
    u::AbstractVector{Float64}) where {d}
    V = SVector{d}(u[m+1] / u[1] for m in 1:d)
    p = (conservation_law.γ-1.0) * (u[end] - 0.5* 
     (sum(u[m+1]*V[m] for m in 1:d)))
    h_t = u[end] + p
    return vcat(
        SMatrix{1,d}(u[n+1] for n in 1:d),
        SMatrix{d,d}(u[m+1]*V[n] + I[m,n]*p for m in 1:d, n in 1:d),
        SMatrix{1,d}(h_t*V[n] for n in 1:d))
end

@inline function physical_flux(conservation_law::EulerType{d}, 
    u::AbstractVector{Float64}, n::NTuple{d, Float64}) where {d}
    V = SVector{d}(u[m+1] / u[1] for m in 1:d)
    p = (conservation_law.γ-1.0) * (u[end] - 0.5*u[1]* 
     (sum(V[m]^2 for m in 1:d)))
    h_t = u[end] + p
    V_n = sum(V[m]*n[m] for m in 1:d)
    
    return SVector{d+2}(
        [u[1]*V_n, [u[1]*V_n*V[m] + p*n[m] for m in 1:d]..., h_t*V_n])
end

@inline @views function physical_flux!(f::AbstractArray{Float64,3},    
    conservation_law::EulerType{d}, 
    u::AbstractMatrix{Float64}) where {d}

    @inbounds for i in axes(u,1)
        f[i,:,:] .= physical_flux(conservation_law, u[i,:])
    end
end

@inline function entropy(conservation_law::EulerType{d}, 
    u::AbstractVector{Float64}) where {d}
    (; γ) = conservation_law
    p = (γ-1) * (u[end] - (0.5/u[1]) * (sum(u[m+1]^2 for m in 1:d)))
    return -u[1]*log(p/u[1]^γ)/(γ-1)
end

@inline function conservative_to_primitive(conservation_law::EulerType{d},
    u::AbstractVector{Float64}) where {d}
    (; γ) = conservation_law
    return vcat(u[1], SVector{d}(u[m+1] / u[1] for m in 1:d),
        (γ-1) * (u[end] - (0.5/u[1]) * (sum(u[m+1]^2 for m in 1:d))))
end

@inline function conservative_to_entropy(conservation_law::EulerType{d}, 
    u::AbstractVector{Float64}) where {d}
    (; γ) = conservation_law
    k = (0.5/u[1]) * (sum(u[m+1]^2 for m in 1:d))
    p = (γ-1) * (u[end] - k)
    return vcat((γ-log(p/(u[1]^γ)))/(γ-1) - k/p,
        SVector{d}(u[m+1] / p for m in 1:d), -u[1]/p)
end

@inline function entropy_to_conservative(conservation_law::EulerType{d}, 
    w::AbstractVector{Float64}) where {d}
    (; γ) = conservation_law
    w = w * (γ-1)
    k = sum(w[m+1]^2 for m in 1:d)/(2*w[end])
    s = γ - w[1] + k
    ρe = ((γ-1)/((-w[end])^γ))^(1/(γ-1))*exp(-s/(γ-1))
    return vcat(-w[end]*ρe, SVector{d}(w[m+1] * ρe for m in 1:d), ρe*(1-k))
end

@inline function wave_speed(conservation_law::EulerType{d},
    u_in::AbstractVector{Float64}, u_out::AbstractVector{Float64},
    n_f::NTuple{d, Float64}) where {d}

    V_in = SVector{d}(u_in[m+1] / u_in[1] for m in 1:d)
    p_in = (conservation_law.γ-1) * (u_in[end] - (0.5/u_in[1]) * 
     (sum(u_in[m+1]^2 for m in 1:d)))
    V_out = SVector{d}(u_out[m+1] / u_out[1] for m in 1:d)
    p_out = (conservation_law.γ-1) * (u_out[end] - (0.5/u_out[1]) * 
     (sum(u_out[m+1]^2 for m in 1:d)))
    Vn_in = sum(V_in[m] * n_f[m] for m in 1:d) 
    Vn_out = sum(V_out[m] * n_f[m] for m in 1:d)
    c_in = sqrt(conservation_law.γ*p_in / u_in[1])
    c_out = sqrt(conservation_law.γ*p_out / u_out[1])

    return max(abs(Vn_in), abs(Vn_out)) + max(c_in, c_out)
end

@inline function compute_two_point_flux(conservation_law::EulerType{d}, 
    ::ConservativeFlux, u_L::AbstractVector{Float64}, 
    u_R::AbstractVector{Float64}) where {d}
    
    return 0.5*(physical_flux(conservation_law, u_L) .+ 
        physical_flux(conservation_law, u_R))
end

@inline function compute_two_point_flux(conservation_law::EulerType{d}, 
    ::ConservativeFlux, u_L::AbstractVector{Float64}, 
    u_R::AbstractVector{Float64}, n::NTuple{d, Float64}) where {d}
    
    return 0.5*(physical_flux(conservation_law, u_L, n) .+ 
        physical_flux(conservation_law, u_R, n))
end

"""
Entropy-conservative, kinetic-energy-preserving, and pressure-equilibrium-preserving numerical flux from Ranocha (see his 2018 PhD thesis)
"""
@inline function compute_two_point_flux(conservation_law::EulerType{d}, 
    ::EntropyConservativeFlux, u_L::AbstractVector{Float64}, 
    u_R::AbstractVector{Float64}) where {d}
    (; γ) = conservation_law

    # velocities and pressures
    V_L = SVector{d}(u_L[m+1] / u_L[1] for m in 1:d)
    V_R = SVector{d}(u_R[m+1] / u_R[1] for m in 1:d)
    p_L = (conservation_law.γ-1) * (u_L[end] - 0.5*u_L[1]* 
     (sum(V_L[m]^2 for m in 1:d)))
    p_R = (conservation_law.γ-1) * (u_R[end] - 0.5*u_R[1]* 
     (sum(V_R[m]^2 for m in 1:d)))

    # mean quantities
    ρ_avg = logmean(u_L[1], u_R[1])
    V_avg = 0.5*(V_L + V_R)
    p_avg = 0.5*(p_L + p_R)
    C = 0.5*sum(V_L[m]*V_R[m] for m in 1:d) + 
        1.0/((γ-1)*logmean(u_L[1]/p_L, u_R[1]/p_R))

    # flux tensor
    f_ρ = SMatrix{1,d}(ρ_avg*V_avg[n] for n in 1:d)
    f_ρV = SMatrix{d,d}(f_ρ[m]*V_avg[n] + I[m,n]*p_avg for m in 1:d, n in 1:d)
    f_E = SMatrix{1,d}(f_ρ[n]*C + 0.5*(p_L*V_R[n] + p_R*V_L[n]) for n in 1:d)

    return vcat(f_ρ, f_ρV, f_E)
end

"""
Isentropic vortex problem, taken verbatim from the Trixi.jl examples (https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_euler_vortex.jl).

Domain should be [-10,10] × [-10,10].
"""
struct TrixiIsentropicVortex <: AbstractGridFunction{2} 
    γ::Float64
    strength::Float64
    N_c::Int
    function TrixiIsentropicVortex(conservation_law::EulerEquations{2}, 
        strength::Float64=5.0)
        return new(conservation_law.γ,strength,4)
    end
end

function evaluate(f::TrixiIsentropicVortex, x::NTuple{2,Float64},t::Float64=0.0)
    inicenter = SVector(0.0, 0.0)
    iniamplitude = f.strength

    # base flow
    gamma = f.γ
    rho = 1.0
    v1 = 1.0
    v2 = 1.0
    vel = SVector(v1, v2)
    p = 25.0

    rt = p / rho                  
    t_loc = 0.0
    cent = inicenter + vel*t_loc
    cent = SVector(x) - cent 
    cent = SVector(-cent[2], cent[1])
    r2 = cent[1]^2 + cent[2]^2
    du = iniamplitude / (2*π) * exp(0.5 * (1 - r2))
    dtemp = -(gamma - 1) / (2 * gamma * rt) * du^2
    rho = rho * (1 + dtemp)^(1 / (gamma - 1))
    vel = vel + du * cent
    v1, v2 = vel
    p = p * (1 + dtemp)^(gamma / (gamma - 1))
    return [rho, rho*v1, rho*v2, p/(gamma-1) + 0.5*rho*(v1^2 + v2^2)]
end

struct IsentropicVortex <: AbstractGridFunction{2} 
    γ::Float64
    Ma::Float64 
    θ::Float64
    R::Float64
    β::Float64
    σ²::Float64
    x_0::NTuple{2,Float64}
    N_c::Int
    
    function IsentropicVortex(conservation_law::EulerEquations{2}; 
        Ma::Float64=0.4, θ::Float64=π/4, R::Float64=1.0, 
        β::Float64=1.0, σ::Float64=1.0, x_0::NTuple{2,Float64}=(0.0,0.0))
        return new(conservation_law.γ,Ma, θ, R, β, σ^2, x_0, 4)
    end
end

function evaluate(f::IsentropicVortex, x::NTuple{2,Float64},t::Float64=0.0)
    (; γ, Ma, θ, R, β, σ², x_0) = f
    x_rel = ((x[1] - x_0[1])/R, (x[2] - x_0[2])/R)
    Ω = β*exp(-0.5/σ² * (x_rel[1]^2 + x_rel[2]^2))
    dv = (-x_rel[2]*Ω, x_rel[1]*Ω)
    dT = -0.5*(γ-1)*Ω^2
    ρ = (1+dT)^(1/(γ-1))
    v = (Ma*cos(θ) + dv[1], Ma*sin(θ) + dv[2])
    p = (ρ^γ)/γ
    E = p/(γ-1) + 0.5*ρ*(v[1]^2 + v[2]^2)
    return [ρ, ρ*v[1], ρ*v[2], E]
end

"""
Periodic wave test case used in the following papers:
- Veilleux et al., "Stable Spectral Difference approach using Raviart-Thomas elements for 3D computations on tetrahedral grids," JSC 2022.
- Pazner and Persson, "Approximate tensor-product preconditioners for very high order discontinuous Galerkin methods," JCP 2018.
- Jiang and Shu, "Efficient Implementation of Weighted ENO Schemes," JCP 1996.

Domain should be [0,2]ᵈ.
"""
struct EulerPeriodicTest{d} <: AbstractGridFunction{d} 
    γ::Float64
    strength::Float64
    N_c::Int

    function EulerPeriodicTest(conservation_law::EulerEquations{d}, 
        strength::Float64=0.2) where {d}
        return new{d}(conservation_law.γ,strength,d+2)
    end
end

function evaluate(f::EulerPeriodicTest{d}, 
    x::NTuple{d,Float64},t::Float64=0.0) where {d}

    ρ = 1.0 + f.strength*sin(π*sum(x[m] for m in 1:d))
    return [ρ, fill(ρ,d)..., 1.0/(f.γ-1.0) + 0.5*ρ*d]
end

"""Inviscid 3D Taylor-Green vortex, I think I got this version of it from Shadpey and Zingg, "Entropy-Stable Multidimensional Summation-by-Parts Discretizations on hp-Adaptive Curvilinear Grids for Hyperbolic Conservation Laws," JSC 2020.

Domain should be [-π,π]³.
"""
struct TaylorGreenVortex <: AbstractGridFunction{3} 
    γ::Float64
    Ma::Float64
    N_c::Int
    function TaylorGreenVortex(conservation_law::EulerEquations{3},
        Ma::Float64=0.1)
        return new(conservation_law.γ,Ma,5)
    end
end

function evaluate(f::TaylorGreenVortex,  x::NTuple{3,Float64}, t::Float64=0.0)

    p = 1.0/(f.γ * f.Ma^2) + 
        1.0/16 * (2*cos(2*x[1]) + 2*cos(2*x[2]) +
        cos(2*x[1])*cos(2*x[3]) + cos(2*x[2])*cos(2*x[3]))
    u = sin(x[1])*cos(x[2])*cos(x[3])
    v = -cos(x[1])*sin(x[2])*cos(x[3])

    return [1.0, u, v, 0.0,  p/(f.γ-1.0) + 0.5*ρ*(u^2 + v^2)]
end