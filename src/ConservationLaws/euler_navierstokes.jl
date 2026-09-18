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
The specific heat ratio is specified as a parameter $\gamma$, which must be greater than unity.
"""
struct EulerEquations{d, N_c} <: AbstractConservationLaw{d, FirstOrder, N_c}
    γ::Float64
    γ_minus_1::Float64
    inv_γ_minus_1::Float64
    source_term::AbstractGridFunction{d}

    function EulerEquations{d}(γ::Float64 = 1.4,
            source_term::AbstractGridFunction{d} = NoSourceTerm{d}()) where {
            d,
    }
        return new{d, d + 2}(γ, γ - 1, 1 / (γ - 1), source_term)
    end
end

struct NavierStokesEquations{d, N_c} <: AbstractConservationLaw{d, SecondOrder, N_c}
    γ::Float64
    γ_minus_1::Float64
    inv_γ_minus_1::Float64
    R:: Float64 # Ideal gas constant
    cv:: Float64
    cp:: Float64 
    Tref:: Float64
    μref:: Float64
    Sμref:: Float64
    source_term::AbstractGridFunction{d}

    # reference values taken at standard sea level values (pg 937 Anderson fundamentals of aerodynamics)
    function NavierStokesEquations{d}(γ::Float64 = 1.4, R:: Float64 = 287.0, Tref:: Float64 = 288.16, μref:: Float64 = 1.7894e-5, Sμref:: Float64 = 110.4,
            source_term::AbstractGridFunction{d} = NoSourceTerm{d}()) where {
            d,
    }
        @error "Navier-Stokes in progress... not yet implemented."
        return new{d, N_c}(γ, γ - 1, 1 / (γ - 1), R, R / (γ - 1), γ * cv, Tref, μref, Sμref, source_term)
    end
end

const EulerType{d} = Union{EulerEquations{d}, NavierStokesEquations{d}}

# Function to evaluate viscous fluxes assume in the form du/dt + sum(d/dxi F_inviscid) + sum(d/dxi F_viscous) = 0
@inline function viscous_flux(conservation_law::NavierStokesEquations{d},
        u::AbstractVector{Float64},
        ∇u::AbstractMatrix{Float64}) where {d}
    
    (; γ_minus_1, cv, cp, Tref, μref, Sμref) = conservation_law

    V = SVector{d}(u[m + 1] / u[1] for m in 1:d) # get the velocities
    ∇V = SVector{d}(∇u[m + 1, n] / u[1] - u[m + 1] * ∇u[1, n] / (u[1]^2) for m in 1:d, n in 1:d) # velocity gradients. 4 components in 2D, 9 in 3D (d velocities gradients in d directions)
    p = γ_minus_1 * (u[end] - 0.5 * (sum(u[m + 1] * V[m] for m in 1:d))) # pressure = (γ-1)*(e-1/(2*ρ)(sum of (ρV_i)))
    T = p / (u[1] * R)   # compute temperature T
    ∇p = SVector{d}(γ_minus_1 * -0.5 * (sum(2 * u[1] * u[m + 1] * ∇u[m + 1, n] - u[m + 1]^2 * ∇u[1, n]) for m in 1:d) / (u[1]^2) + ∇u[end, n] for n in 1:d) # pressure gradients
    ∇T = SVector{d}((u[1] * ∇p[m] - p * ∇u[1])/(u[1]^2 * R) for m in 1:d) # temperature gradients

    # compute dynamic viscosity μ using Sutherland's law
    μ = μref * (T/Tref)^(3/2) * (Tref + Sμref) / (T + Sμref)
    
    # compute thermal conductivity κ 
    κ = μ * cp / 0.71 # maybe do it using sutherlands law...

    # compute viscous stress tensor τ
    τ = SMatrix{d, d}(μ * (∇V[m, n] + ∇V[n, m] - (2/3) * (m == n ? sum(∇V[k, k] for k in 1:d) : 0.0)) for m in 1:d, n in 1:d) 

    # compute heat flux vector q
    q = SVector{d}(κ*∇T[m] for m in 1:d)
    
    return -1 * (vcat(Smatrix{1, d}(0.0 for n in 1:d),
        τ,
        SMatrix{1, d}(sum(τ[m, n] * V[n] for n in 1:d) + q[m] for m in 1:d)))
end


# Evaluate the flux for the Euler equations
@inline function physical_flux(conservation_law::EulerType{d},
        u::AbstractVector{Float64}) where {d}
    (; γ_minus_1) = conservation_law

    V = SVector{d}(u[m + 1] / u[1] for m in 1:d)
    p = γ_minus_1 * (u[end] - 0.5 * (sum(u[m + 1] * V[m] for m in 1:d)))
    h_t = u[end] + p
    return vcat(SMatrix{1, d}(u[n + 1] for n in 1:d),
        SMatrix{d, d}(u[m + 1] * V[n] + I[m, n] * p for m in 1:d, n in 1:d),
        SMatrix{1, d}(h_t * V[n] for n in 1:d))
end

@inline function physical_flux(conservation_law::EulerType{d}, # facet flux 
        u::AbstractVector{Float64},
        n_f::AbstractVector{Float64}) where {d}
    (; γ_minus_1) = conservation_law

    V = SVector{d}(u[m + 1] / u[1] for m in 1:d)
    Vₙ = sum(V[m] * n_f[m] for m in 1:d)

    p = γ_minus_1 * (u[end] - 0.5 * (sum(u[m + 1] * V[m] for m in 1:d)))

    return SVector{d + 2}(u[1] * Vₙ,
        (u[m + 1] * Vₙ + p * n_f[m] for m in 1:d)...,
        (u[end] + p) * Vₙ)
end

@inline @views function physical_flux!(f::AbstractArray{Float64, 3},
        conservation_law::EulerType{d},
        u::AbstractMatrix{Float64}) where {d}
    @inbounds for i in axes(u, 1)
        f[i, :, :] .= physical_flux(conservation_law, u[i, :])
    end
end

@inline function entropy(conservation_law::EulerType{d},
        u::AbstractVector{Float64}) where {d}
    (; γ, γ_minus_1, inv_γ_minus_1) = conservation_law
    p = γ_minus_1 * (u[end] - (0.5 / u[1]) * (sum(u[m + 1]^2 for m in 1:d)))
    return -u[1] * log(p / (u[1]^γ)) * inv_γ_minus_1
end

@inline function conservative_to_entropy!(
    w::AbstractVector{Float64},
    conservation_law::EulerType{d},
    u::AbstractVector{Float64},
) where {d}

    (; γ, γ_minus_1, inv_γ_minus_1) = conservation_law

    ρ = u[1]

    ke_num = 0.0
    @inbounds for m in 1:d
        um = u[m + 1]
        ke_num += um * um
    end

    k = 0.5 * ke_num / ρ

    p = γ_minus_1 * (u[d + 2] - k)
    inv_p = 1.0 / p

    logp = log(p)
    logρ = log(ρ)

    w[1] = inv_γ_minus_1 * (γ - (logp - γ * logρ)) - k * inv_p

    @inbounds for m in 1:d
        w[m + 1] = u[m + 1] * inv_p
    end

    w[d + 2] = -ρ * inv_p

    return
end
@inline function entropy_to_conservative!(
    u::AbstractVector{Float64},
    conservation_law::EulerType{d},
    w::AbstractVector{Float64},
) where {d}

    (; γ, γ_minus_1, inv_γ_minus_1) = conservation_law

    k = γ_minus_1 * sum(w[m + 1]^2 for m in 1:d) / (2 * w[d + 2])

    s = γ - γ_minus_1 * w[1] + k

    wd2 = γ_minus_1 * w[d + 2]

    ρe = exp(
        (log(γ_minus_1) - γ * log(-wd2) - s) * inv_γ_minus_1
    )

    u[1] = -wd2 * ρe

    @inbounds for m in 1:d
        u[m + 1] = γ_minus_1 * w[m + 1] * ρe
    end

    u[d + 2] = ρe * (1 - k)

    return
end

@inline function wave_speed(conservation_law::EulerType{d},
        u_in::AbstractVector{Float64},
        u_out::AbstractVector{Float64},
        n_f) where {d}
    (; γ, γ_minus_1) = conservation_law

    V_in = SVector{d}(u_in[m + 1] / u_in[1] for m in 1:d)
    p_in = γ_minus_1 * (u_in[end] - (0.5 / u_in[1]) * (sum(u_in[m + 1]^2 for m in 1:d)))
    V_out = SVector{d}(u_out[m + 1] / u_out[1] for m in 1:d)
    p_out = γ_minus_1 * (u_out[end] - (0.5 / u_out[1]) * (sum(u_out[m + 1]^2 for m in 1:d)))
    Vn_in = sum(V_in[m] * n_f[m] for m in 1:d)
    Vn_out = sum(V_out[m] * n_f[m] for m in 1:d)

    c_in = sqrt(γ * p_in / u_in[1])
    c_out = sqrt(γ * p_out / u_out[1])

    return max(abs(Vn_in), abs(Vn_out)) + max(c_in, c_out)
end

@inline function compute_two_point_flux(conservation_law::EulerType{d},
        ::ConservativeFlux,
        u_L::AbstractVector{Float64},
        u_R::AbstractVector{Float64}) where {d}
    return 0.5 *
           (physical_flux(conservation_law, u_L) .+ physical_flux(conservation_law, u_R))
end

@inline function compute_two_point_flux(conservation_law::EulerType{d},
        ::ConservativeFlux,
        u_L::AbstractVector{Float64},
        u_R::AbstractVector{Float64},
        n_f::AbstractVector{Float64}) where {d}
    return 0.5 * (physical_flux(conservation_law, u_L, n_f) .+
            physical_flux(conservation_law, u_R, n_f))
end

# Entropy-conservative, kinetic-energy-preserving, and pressure-equilibrium-preserving  
# numerical flux from Ranocha (see his 2018 PhD thesis)
@inline function compute_two_point_flux(conservation_law::EulerType{d},
        ::EntropyConservativeFlux,
        u_L::AbstractVector{Float64},
        u_R::AbstractVector{Float64}) where {d}
    (; γ_minus_1, inv_γ_minus_1) = conservation_law

    # velocities and pressures
    V_L = SVector{d}(u_L[m + 1] / u_L[1] for m in 1:d)
    V_R = SVector{d}(u_R[m + 1] / u_R[1] for m in 1:d)
    p_L = γ_minus_1 * (u_L[end] - 0.5 * u_L[1] * (sum(V_L[m]^2 for m in 1:d)))
    p_R = γ_minus_1 * (u_R[end] - 0.5 * u_R[1] * (sum(V_R[m]^2 for m in 1:d)))

    # mean quantities
    ρ_avg = logmean(u_L[1], u_R[1])
    V_avg = 0.5 * (V_L + V_R)
    p_avg = 0.5 * (p_L + p_R)
    C = 0.5 * sum(V_L[m] * V_R[m] for m in 1:d) +
        inv_γ_minus_1 * inv_logmean(u_L[1] / p_L, u_R[1] / p_R)

    # flux tensor
    f_ρ = SMatrix{1, d}(ρ_avg * V_avg[n] for n in 1:d)
    f_ρV = SMatrix{d, d}(f_ρ[m] * V_avg[n] + I[m, n] * p_avg for m in 1:d, n in 1:d)
    f_E = SMatrix{1, d}(f_ρ[n] * C + 0.5 * (p_L * V_R[n] + p_R * V_L[n]) for n in 1:d)
    return vcat(f_ρ, f_ρV, f_E)
end

@inline function compute_two_point_flux(conservation_law::EulerType{d},
        ::EntropyConservativeFlux,
        u_L::AbstractVector{Float64},
        u_R::AbstractVector{Float64},
        n::AbstractVector{Float64}) where {d}
    (; γ_minus_1, inv_γ_minus_1) = conservation_law

    # velocities and pressures
    V_L = SVector{d}(u_L[m + 1] / u_L[1] for m in 1:d)
    V_R = SVector{d}(u_R[m + 1] / u_R[1] for m in 1:d)
    Vn_L = sum(V_L[m] * n[m] for m in 1:d)
    Vn_R = sum(V_R[m] * n[m] for m in 1:d)
    p_L = γ_minus_1 * (u_L[end] - 0.5 * u_L[1] * (sum(V_L[m]^2 for m in 1:d)))
    p_R = γ_minus_1 * (u_R[end] - 0.5 * u_R[1] * (sum(V_R[m]^2 for m in 1:d)))

    # mean quantities
    ρ_avg = logmean(u_L[1], u_R[1])
    V_avg = 0.5 * (V_L + V_R)
    Vn_avg = 0.5 * (Vn_L + Vn_R)
    p_avg = 0.5 * (p_L + p_R)
    C = 0.5 * sum(V_L[m] * V_R[m] for m in 1:d) +
        inv_γ_minus_1 * inv_logmean(u_L[1] / p_L, u_R[1] / p_R)

    # flux vector
    f_ρ = ρ_avg * Vn_avg
    return SVector{d + 2}(f_ρ,
        (f_ρ * V_avg[m] + p_avg * n[m] for m in 1:d)...,
        f_ρ * C + 0.5 * (p_L * Vn_R + p_R * Vn_L))
end

@doc raw"""
    IsentropicVortex(conservation_law::EulerEquations{2};
                     Ma = 0.4, θ = π/4, R = 1.0, β = 1.0,
                     σ = 1.0, x_0 = (0.0, 0.0))

Isentropic vortex test case for `EulerEquations{2}`.
"""
struct IsentropicVortex <: AbstractGridFunction{2}
    γ::Float64
    Ma::Float64
    θ::Float64
    R::Float64
    β::Float64
    σ²::Float64
    x_0::NTuple{2, Float64}
    N_c::Int

    function IsentropicVortex(conservation_law::EulerEquations{2};
            Ma::Float64 = 0.4,
            θ::Float64 = π / 4,
            R::Float64 = 1.0,
            β::Float64 = 1.0,
            σ::Float64 = 1.0,
            x_0::NTuple{2, Float64} = (0.0, 0.0))
        return new(conservation_law.γ, Ma, θ, R, β, σ^2, x_0, 4)
    end
end

#= @inline function evaluate(f::IsentropicVortex, x::NTuple{2, Float64}, t::Float64 = 0.0)
    (; γ, Ma, θ, R, β, σ², x_0) = f
    x_rel = ((x[1] - x_0[1]) / R, (x[2] - x_0[2]) / R)
    Ω = β * exp(-0.5 / σ² * (x_rel[1]^2 + x_rel[2]^2))
    dv = (-x_rel[2] * Ω, x_rel[1] * Ω)
    dT = -0.5 * (γ - 1) * Ω^2
    ρ = (1 + dT)^(1 / (γ - 1))
    v = (Ma * cos(θ) + dv[1], Ma * sin(θ) + dv[2])
    p = (ρ^γ) / γ
    E = p / (γ - 1) + 0.5 * ρ * (v[1]^2 + v[2]^2)
    return SVector{4}(ρ, ρ * v[1], ρ * v[2], E)
end =#

@inline function evaluate(f::IsentropicVortex,x::NTuple{2, Float64},t::Float64)

    (; γ, Ma, θ, R, β, σ², x_0) = f
    # Freestream convection velocity
    u∞ = (Ma * cos(θ), Ma * sin(θ))
    # Moving vortex center
    xc = (x_0[1] + u∞[1] * t, x_0[2] + u∞[2] * t)
    # Relative coordinates
    x_rel = ((x[1] - xc[1]) / R, (x[2] - xc[2]) / R)
    Ω = β * exp(-0.5 / σ² * (x_rel[1]^2 + x_rel[2]^2))
    dv = (-x_rel[2] * Ω, x_rel[1] * Ω)
    dT = -0.5 * (γ - 1) * Ω^2
    ρ = (1 + dT)^(1 / (γ - 1))
    v = (u∞[1] + dv[1],u∞[2] + dv[2])
    p = (ρ^γ) / γ
    E = p / (γ - 1) + 0.5 * ρ * (v[1]^2 + v[2]^2)

    return SVector{4}(ρ, ρ*v[1], ρ*v[2], E)
end

# similar to the euler periodic but a gaussian. Note this problem is not periodic 
struct EulerGaussianPulse{d} <: AbstractGridFunction{d}
    γ::Float64
    amplitude::Float64
    σ::Float64
    x₀::NTuple{d,Float64}
    N_c::Int

    function EulerGaussianPulse(
            conservation_law::EulerEquations{d},
            amplitude::Float64 = 0.5,
            σ::Float64 = 0.05,
            x₀::NTuple{d,Float64} = ntuple(_ -> 0.5, d)) where {d}

        return new{d}(conservation_law.γ, amplitude, σ, x₀, d + 2)
    end
end    

@inline function evaluate(f::EulerGaussianPulse{d},
    x::NTuple{d,Float64},
    t::Float64 = 0.0) where {d}

    r² = 0.0

    @inbounds for m in 1:d
        ξ = x[m] - (f.x₀[m] + t)
        r² += ξ^2
    end

    ρ = 1.0 + f.amplitude * exp(-r² / f.σ^2)

    return SVector{d+2}(ρ, fill(ρ,d)..., 1.0/(f.γ - 1.0) + 0.5*d*ρ )
end

@doc raw"""
    EulerPeriodicTest(conservation_law::EulerEquations{d},
                      strength = 0.2, L = 2.0)

Periodic wave test case for `EulerEquations{d}` on the domain $\Omega = [0,L]^d$, based on the following paper:
- G.-S. Jiang and C.-W. Shu (1996). Efficient implementation of weighted ENO schemes. 
  *Journal of Computational Physics* 126(1): 202-228.
"""
struct EulerPeriodicTest{d} <: AbstractGridFunction{d}
    γ::Float64
    strength::Float64
    L::Float64
    N_c::Int

    function EulerPeriodicTest(conservation_law::EulerEquations{d},
            strength::Float64 = 0.2,
            L::Float64 = 2.0) where {d}
        return new{d}(conservation_law.γ, strength, L, d + 2)
    end
end

@inline function evaluate(f::EulerPeriodicTest{d},
        x::NTuple{d, Float64},
        t::Float64 = 0.0) where {d}
    ρ = 1.0 + f.strength * sin(2π * sum(x[m] for m in 1:d) / f.L)
    return SVector{d + 2}(ρ, fill(ρ, d)..., 1.0 / (f.γ - 1.0) + 0.5 * ρ * d)
end

#= @inline function evaluate(f::EulerPeriodicTest{d},
    x::NTuple{d,Float64},
    t::Float64 = 0.0) where {d}

    phase = sum(x) - d*t

    ρ = 1.0 + f.strength * sin(2π * phase / f.L)

    return SVector{d+2}(ρ, fill(ρ,d)..., 1.0/(f.γ - 1.0) + 0.5*ρ*d)
end =#

@doc raw"""
    TaylorGreenVortex(conservation_law::EulerEquations{3}, Ma = 0.1)

Inviscid Taylor-Green vortex initial condition for `EulerEquations{3}` on the domain
$\Omega = [0, 2\pi]^3$.
"""
struct TaylorGreenVortex <: AbstractGridFunction{3}
    γ::Float64
    Ma::Float64
    N_c::Int
    function TaylorGreenVortex(conservation_law::EulerEquations{3}, Ma::Float64 = 0.1)
        return new(conservation_law.γ, Ma, 5)
    end
end

@inline function evaluate(f::TaylorGreenVortex, x::NTuple{3, Float64}, t::Float64 = 0.0)
    p = (1 / (f.Ma^2 * f.γ)) +
        0.0625 * (2 * cos(2 * x[1]) +
         2 * cos(2 * x[2]) +
         cos(2 * x[1]) * cos(2 * x[3]) +
         cos(2 * x[2]) * cos(2 * x[3]))
    u = sin(x[1]) * cos(x[2]) * cos(x[3])
    v = -cos(x[1]) * sin(x[2]) * cos(x[3])
    return SVector{5}(1.0, u, v, 0.0, p / (f.γ - 1) + 0.5 * (u^2 + v^2))
end

@doc raw"""
    KelvinHelmholtzInstability(conservation_law::EulerEquations{2}, ρ_0 = 0.5)

Kelvin-Helmholtz instability initial condition for `EulerEquations{2}` on the domain
$\Omega = [0,2]^2$
"""
struct KelvinHelmholtzInstability <: AbstractGridFunction{2}
    γ::Float64
    ρ_0::Float64
    N_c::Int
    function KelvinHelmholtzInstability(conservation_law::EulerEquations{2},
            ρ_0::Float64 = 0.5)
        return new(conservation_law.γ, ρ_0, 4)
    end
end

@inline function evaluate(f::KelvinHelmholtzInstability,
        x::NTuple{2, Float64},
        t::Float64 = 0.0)
    x_rel = (x[1] - 1, x[2] - 1)
    B = tanh(15 * x_rel[2] + 7.5) - tanh(15 * x_rel[2] - 7.5)
    ρ = f.ρ_0 + 0.75 * B
    p = 1.0
    u = 0.5 * (B - 1)
    v = 0.1 * sin(2π * x_rel[1])
    return SVector{4}(ρ, ρ * u, ρ * v, p / (f.γ - 1) + 0.5 * ρ * (u^2 + v^2))
end

struct EulerCosineWave{d} <: AbstractGridFunction{d}
    γ::Float64
    amplitude::Float64
    L::Float64
    N_c::Int

    function EulerCosineWave(
            conservation_law::EulerEquations{d},
            amplitude::Float64 = 0.2,
            L::Float64 = 1.0
        ) where {d}

        new{d}(conservation_law.γ,
               amplitude,
               L,
               d + 2)
    end
end

@inline function evaluate(f::EulerCosineWave{d},
    x::NTuple{d,Float64},
    t::Float64 = 0.0) where {d}

ϕ = 1.0

@inbounds for m in 1:d
    ϕ *= cos(2*π * (x[m] - t) / f.L)
end

ρ = 1.0 + f.amplitude * ϕ

return SVector{d+2}(
    ρ,
    fill(ρ,d)...,
    1.0/(f.γ - 1.0) + 0.5*d*ρ
)
end