@doc raw"""
    InviscidBurgersEquation(a::NTuple{d,Float64}) where {d}

Define an inviscid Burgers' equation of the form
```math
\partial_t U(\bm{x},t) + \bm{\nabla} \cdot \big(\tfrac{1}{2}\bm{a} U(\bm{x},t)^2 \big) = 0,
```
where $\bm{a} \in \R^d$. A specialized constructor `InviscidBurgersEquation()` is provided for the one-dimensional case with `a = (1.0,)`.
"""
struct InviscidBurgersEquation{d} <: AbstractConservationLaw{d,FirstOrder}
    a::NTuple{d,Float64} 
    source_term::AbstractGridFunction{d}
    N_c::Int

    function InviscidBurgersEquation(a::NTuple{d,Float64}, 
        source_term::AbstractGridFunction{d}=NoSourceTerm()) where {d}
        return new{d}(a, source_term, 1)
    end
end

@doc raw"""
    ViscousBurgersEquation(a::NTuple{d,Float64}, b::Float64) where {d}

Define a viscous Burgers' equation of the form
```math
\partial_t U(\bm{x},t) + \bm{\nabla} \cdot \big(\tfrac{1}{2}\bm{a} U(\bm{x},t)^2 - b \bm{\nabla} U(\bm{x},t)\big) = 0,
```
where $\bm{a} \in \R^d$ and $b \in \R^+$. A specialized constructor `ViscousBurgersEquation(b::Float64)` is provided for the one-dimensional case with `a = (1.0,)`.
"""
struct ViscousBurgersEquation{d} <: AbstractConservationLaw{d,SecondOrder}
    a::NTuple{d,Float64}
    b::Float64
    source_term::AbstractGridFunction{d}
    N_c::Int

    function ViscousBurgersEquation(a::NTuple{d,Float64}, b::Float64, 
        source_term::AbstractGridFunction{d}=NoSourceTerm()) where {d}
        return new{d}(a, b, source_term, 1)
    end
end

const BurgersType{d} = Union{InviscidBurgersEquation{d}, ViscousBurgersEquation{d}}

InviscidBurgersEquation() = InviscidBurgersEquation((1.0,))
ViscousBurgersEquation(b::Float64) = ViscousBurgersEquation((1.0,), b)

"""
Evaluate the flux for the inviscid Burgers' equation

`F(u) = a ½u^2`
"""
function physical_flux!(f::AbstractArray{Float64,3},
    conservation_law::BurgersType{d}, 
    u::AbstractMatrix{Float64}) where {d}
    @inbounds for m in 1:d
        f[:,:,m] .= 0.5* conservation_law.a[m] * u.^2
    end
end

"""
Evaluate the flux for the viscous Burgers' equation

`F(u,q) = a ½u^2 - bq`
"""
function physical_flux!(f::AbstractArray{Float64,3},
    conservation_law::ViscousBurgersEquation{d},
    u::AbstractMatrix{Float64}, q::NTuple{d,AbstractMatrix{Float64}}) where {d}
    @inbounds for m in 1:d
        f[:,:,m] .= 0.5*conservation_law.a[m] * u.^2 .- 
            conservation_law.b .* q[:,:,m]
    end
end

"""
Evaluate the Lax-Friedrichs flux for Burgers' equation
`F*(u⁻, u⁺, n) = ½a⋅n(½(u⁻)² + ½(u⁺)²) + ½λ max(|au⁻⋅n|,|au⁺⋅n|)(u⁺ - u⁻)`
"""
function numerical_flux!(
    f_star::AbstractMatrix{Float64},
    conservation_law::BurgersType{d}, numerical_flux::LaxFriedrichsNumericalFlux, 
    u_in::AbstractMatrix{Float64}, u_out::AbstractMatrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    a_n = sum(conservation_law.a[m].*n[m] for m in 1:d)
    f_star .= 0.25*a_n.*(u_in.^2 .+ u_out.^2) .- 
        numerical_flux.λ*0.5 * 
        max.(abs.(a_n*u_out),abs.(a_n*u_in)) .* (u_out .- u_in)
end

"""
Evaluate the interface normal solution for the viscous Burgers' equation using the BR1 approach

`U*(u⁻, u⁺, n) = ½(u⁻ + u⁺)n`
"""
function numerical_flux!(u_nstar::AbstractArray{Float64,3},
    ::ViscousBurgersEquation{d},
    ::BR1, u_in::AbstractMatrix{Float64}, u_out::AbstractMatrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    u_avg = 0.5*(u_in .+ u_out)

    @inbounds for m in 1:d
        u_nstar[:,:,m] .= u_avg.*n[m]
    end
end

"""
Evaluate the numerical flux for the viscous Burgers' equation using the BR1 approach

F*(u⁻, u⁺, q⁻, q⁺, n) = ½(F²(u⁻,q⁻) + F²(u⁺, q⁺))⋅n
"""
function numerical_flux!(f_star::AbstractMatrix{Float64},
    conservation_law::ViscousBurgersEquation{d},
    ::BR1, u_in::AbstractMatrix{Float64}, u_out::AbstractMatrix{Float64}, 
    q_in::AbstractArray{Float64,3}, q_out::AbstractArray{Float64,3}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    # average both sides
    minus_q_avg = -0.5*(q_in .+ q_out)
    f_star .+= sum(conservation_law.b * minus_q_avg[:,:,m] .* n[m] for m in 1:d)
end

function evaluate(
    exact_solution::ExactSolution{1,InviscidBurgersEquation{1},InitialDataGassner,SourceTermGassner},
    x::NTuple{1,Float64},t::Float64=0.0)
    if !exact_solution.periodic
        z = x[1] - t
    else
        z = x[1]
    end
    return [sin(exact_solution.initial_data.k*z) +
        exact_solution.initial_data.ϵ]
end