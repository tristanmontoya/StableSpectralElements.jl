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

    function EulerEquations{d}(γ::Float64=1.4, 
        source_term::AbstractGridFunction{d}=NoSourceTerm{d}()) where {d}
        return new{d}(γ,source_term, d+2)
    end
end

struct NavierStokesEquations{d} <: AbstractConservationLaw{d,SecondOrder} 
    γ::Float64
    source_term::AbstractGridFunction{d}
    N_c::Int

    function NavierStokesEquations{d}(γ::Float64=1.4, 
        source_term::AbstractGridFunction{d}=NoSourceTerm{d}()) where {d}
        @error "Navier-Stokes not implemented."
        return new{d}(γ,source_term, d+2)
    end
end

const EulerType{d} = Union{EulerEquations{d}, NavierStokesEquations{d}}

"""
Evaluate the flux for the Euler equations
"""
function physical_flux!(f::AbstractArray{Float64,3},    
    conservation_law::EulerType{d}, 
    u::AbstractMatrix{Float64}) where {d}

    @inbounds for i in axes(u,1)
        V = SVector{d}(u[i,m+1] / u[i,1] for m in 1:d)
        p = (conservation_law.γ-1) * (u[i,end] - (0.5/u[i,1]) * 
         (sum(u[i,m+1]^2 for m in 1:d)))
        f[i,1,:] .= u[i,2:end-1]
        f[i,2:end-1,:] .= SMatrix{d,d}(u[i,m+1]*V[n] + I[m,n]*p 
            for m in 1:d, n in 1:d)
        f[i,end,:] .= (u[i,end] + p)*V
    end
end

@inline function entropy(conservation_law::EulerType{d}, 
    u::AbstractVector{Float64}) where {d}
    p = (conservation_law.γ-1) * (u[end] - (0.5/u[1]) * 
        (sum(u[m+1]^2 for m in 1:d)))
    return -u[1]*log(p/u[1]^conservation_law.γ)/(conservation_law.γ-1)
end

@inline function conservative_to_primitive(conservation_law::EulerType{d},
    u::AbstractVector{Float64}) where {d}
    return vcat(u[1], SVector{d}(u[m+1] / u[1] for m in 1:d),
        (conservation_law.γ-1) * (u[end] - (0.5/u[1]) * 
        (sum(u[m+1]^2 for m in 1:d))))
end

@inline function conservative_to_entropy(conservation_law::EulerType{d}, 
    u::AbstractVector{Float64}) where {d}
    (; γ) = conservation_law
    k = (0.5/u[1]) * (sum(u[m+1]^2 for m in 1:d))
    p = (γ-1) * (u[end] - k)
    return vcat((γ-log(p/(u[1]^γ)))/(γ-1) - k/p,
        SVector{d}(u[m+1] / p for m in 1:d), -u[1]/p)
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
    
    V_L = SVector{d}(u_L[m+1] / u_L[1] for m in 1:d)
    p_L = (conservation_law.γ-1) * (u_L[end] - (0.5/u_L[1]) * 
     (sum(u_L[m+1]^2 for m in 1:d)))
    V_R = SVector{d}(u_R[m+1] / u_R[1] for m in 1:d)
    p_R = (conservation_law.γ-1) * (u_R[end] - (0.5/u_R[1]) * 
     (sum(u_R[m+1]^2 for m in 1:d)))

     return 0.5*
        (vcat(SMatrix{1,d}(u_L[2:end-1]),
            SMatrix{d,d}(u_L[m+1]*V_L[n] + I[m,n]*p_L 
                for m in 1:d, n in 1:d),
            SMatrix{1,d}((u_L[end] + p_L)*V_L)) +
        vcat(SMatrix{1,d}(u_R[2:end-1]),
            SMatrix{d,d}(u_R[m+1]*V_R[n] + I[m,n]*p_R
                for m in 1:d, n in 1:d),
            SMatrix{1,d}((u_R[end] + p_R)*V_R)))
end

"""
Isentropic vortex problem, taken verbatim from the Trixi.jl examples (https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_euler_vortex.jl).

Domain should be [-10,10] × [-10,10].
"""
struct IsentropicVortex <: AbstractGridFunction{2} 
    γ::Float64
    strength::Float64
    N_c::Int
    function IsentropicVortex(conservation_law::EulerEquations{2}, 
        strength::Float64=5.0)
        return new(conservation_law.γ,strength,4)
    end
end

function evaluate(f::IsentropicVortex, x::NTuple{2,Float64},t::Float64=0.0)
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
    return [ρ, fill(ρ,d)..., 1.0/(1.0-f.γ) + 0.5*ρ*d]
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

    return [1.0, u, v, 0.0,  p/(1.0-f.γ) + 0.5*ρ*(u^2 + v^2)]
end