@doc raw"""
    InviscidBurgersEquation(a::NTuple{d,Float64}) where {d}

Define an inviscid Burgers' equation of the form
```math
\partial_t U(\bm{x},t) + \bm{\nabla}_{\bm{x}} \cdot \big(\tfrac{1}{2}\bm{a} U(\bm{x},t)^2 \big) = 0,
```
where $\bm{a} \in \R^d$. A specialized constructor `InviscidBurgersEquation()` is provided for the one-dimensional case with `a = (1.0,)`.
"""
struct InviscidBurgersEquation{d} <: AbstractConservationLaw{d, FirstOrder, 1}
    a::NTuple{d, Float64}
    source_term::AbstractGridFunction{d}

    function InviscidBurgersEquation(a::NTuple{d, Float64},
            source_term::AbstractGridFunction{d} = NoSourceTerm{d}()) where {
            d,
    }
        return new{d}(a, source_term)
    end
end

@doc raw"""
    ViscousBurgersEquation(a::NTuple{d,Float64}, b::Float64) where {d}

Define a viscous Burgers' equation of the form
```math
\partial_t U(\bm{x},t) + \bm{\nabla}_{\bm{x}} \cdot \big(\tfrac{1}{2}\bm{a} U(\bm{x},t)^2 - b \bm{\nabla} U(\bm{x},t)\big) = 0,
```
where $\bm{a} \in \R^d$ and $b \in \R^+$. A specialized constructor `ViscousBurgersEquation(b::Float64)` is provided for the one-dimensional case with `a = (1.0,)`.
"""
struct ViscousBurgersEquation{d} <: AbstractConservationLaw{d, SecondOrder, 1}
    a::NTuple{d, Float64}
    b::Float64
    source_term::AbstractGridFunction{d}

    function ViscousBurgersEquation(a::NTuple{d, Float64},
            b::Float64,
            source_term::AbstractGridFunction{d} = NoSourceTerm{d}()) where {
            d,
    }
        return new{d}(a, b, source_term)
    end
end

const BurgersType{d} = Union{InviscidBurgersEquation{d}, ViscousBurgersEquation{d}}

# Default constructors
InviscidBurgersEquation() = InviscidBurgersEquation((1.0,))
ViscousBurgersEquation(b::Float64) = ViscousBurgersEquation((1.0,), b)

# Evaluate the flux F(u) = a u^2/2 for the inviscid Burgers' equation
function physical_flux!(f::AbstractArray{Float64, 3},
        conservation_law::InviscidBurgersEquation{d},
        u::AbstractMatrix{Float64}) where {d}
    @inbounds for m in 1:d
        f[:, :, m] .= 0.5 * conservation_law.a[m] * u .^ 2
    end
end

# Evaluate the flux F(u,q) = a u^2/2 - bq for the viscous Burgers' equation 
function physical_flux!(f::AbstractArray{Float64, 3},
        conservation_law::ViscousBurgersEquation{d},
        u::AbstractMatrix{Float64},
        q::AbstractArray{Float64, 3}) where {d}
    @inbounds for m in 1:d
        f[:, :, m] .= 0.5 * conservation_law.a[m] * u .^ 2 .-
                      conservation_law.b * q[:, :, m]
    end
end

# Evaluate the interface normal solution U*(u⁻, u⁺, n) = ½(u⁻ + u⁺)n 
# for the viscous Burgers' equation using the BR1 approach 
@inline function numerical_flux!(u_nstar::AbstractArray{Float64, 3},
        ::ViscousBurgersEquation{d},
        ::BR1,
        u_in::AbstractMatrix{Float64},
        u_out::AbstractMatrix{Float64},
        n_f::AbstractMatrix{Float64}) where {d}
    u_avg = 0.5 * (u_in .+ u_out)

    @inbounds for m in 1:d
        u_nstar[:, :, m] .= u_avg .* n_f[m, :]
    end
end

# Evaluate the numerical flux for the viscous Burgers' equation using the BR1 approach
# F*(u⁻, u⁺, q⁻, q⁺, n) = ½(F²(u⁻,q⁻) + F²(u⁺, q⁺))⋅n
@inline function numerical_flux!(f_star::AbstractMatrix{Float64},
        conservation_law::ViscousBurgersEquation{d},
        ::BR1,
        u_in::AbstractMatrix{Float64},
        u_out::AbstractMatrix{Float64},
        q_in::AbstractArray{Float64, 3},
        q_out::AbstractArray{Float64, 3},
        n_f::AbstractMatrix{Float64}) where {d}
    minus_q_avg = -0.5 * (q_in .+ q_out)
    f_star .+= sum(conservation_law.b * minus_q_avg[:, :, m] .* n_f[m, :] for m in 1:d)
end

@inline entropy(::BurgersType, u) = 0.5 * u[1]^2

@inline function wave_speed(conservation_law::BurgersType{d},
        u_in::AbstractVector{Float64},
        u_out::AbstractVector{Float64},
        n_f) where {d}
    a_n = sum(conservation_law.a[m] * n_f[m] for m in 1:d)
    return max(abs(a_n * u_in[1]), abs(a_n * u_out[1]))
end

@inline function compute_two_point_flux(conservation_law::BurgersType{d},
        ::EntropyConservativeFlux,
        u_L::AbstractVector{Float64},
        u_R::AbstractVector{Float64}) where {d}
    flux_1d = (u_L[1]^2 + u_L[1] * u_R[1] + u_R[1]^2) / 6
    return SMatrix{1, d}(conservation_law.a[m] * flux_1d for m in 1:d)
end

@inline function compute_two_point_flux(conservation_law::BurgersType{d},
        ::EntropyConservativeFlux,
        u_L::AbstractVector{Float64},
        u_R::AbstractVector{Float64},
        n::NTuple{d, Float64}) where {d}
    flux_1d = (u_L[1]^2 + u_L[1] * u_R[1] + u_R[1]^2) / 6
    return SVector{1}(n[m] * conservation_law.a[m] * flux_1d for m in 1:d)
end

@inline function compute_two_point_flux(conservation_law::BurgersType{d},
        ::ConservativeFlux,
        u_L::AbstractVector{Float64},
        u_R::AbstractVector{Float64}) where {d}
    flux_1d = (u_L[1]^2 + u_R[1]^2) * 0.25
    return SMatrix{1, d}(conservation_law.a[m] * flux_1d for m in 1:d)
end

@inline function compute_two_point_flux(conservation_law::BurgersType{d},
        ::ConservativeFlux,
        u_L::AbstractVector{Float64},
        u_R::AbstractVector{Float64},
        n::NTuple{d, Float64}) where {d}
    flux_1d = (u_L[1]^2 + u_R[1]^2) * 0.25
    return SVector{1}(n[m] * conservation_law.a[m] * flux_1d for m in 1:d)
end

function evaluate(
        exact_solution::ExactSolution{1,
            InviscidBurgersEquation{1},
            InitialDataGassner,
            SourceTermGassner},
        x::NTuple{1, Float64},
        t::Float64 = 0.0)
    if !exact_solution.periodic
        z = x[1] - t
    else
        z = x[1]
    end
    return [sin(exact_solution.initial_data.k * z) + exact_solution.initial_data.ϵ]
end
