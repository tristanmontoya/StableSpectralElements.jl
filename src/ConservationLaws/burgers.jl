@doc raw"""
    InviscidBurgersEquation(a::NTuple{d,Float64}) where {d}

Define an inviscid burgers equation of the form
```math
\partial_t U(\bm{x},t) + \bm{\nabla} \cdot \big(\tfrac{1}{2}\bm{a} U(\bm{x},t)^2 \big) = 0,
```
where $\bm{a} \in \R^d$. A specialized constructor `InviscidBurgersEquation()` is provided for the one-dimensional case, wherein the PDE becomes
```math
\partial_t U(x,t) + \partial_x \big(\tfrac{1}{2}U(x,t)^2 \big) = 0.
```
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
    ViscousBurgersEquation(a::NTuple{d,Float64}, 
    b::Float64) where {d}

Define a viscous burgers equation of the form
```math
\partial_t U(\bm{x},t) + \bm{\nabla} \cdot \big(\tfrac{1}{2}\bm{a} U(\bm{x},t)^2 - b \bm{\nabla} U(\bm{x},t)\big) = 0,
```
where $\bm{a} \in \R^d$ and $b \in \R^+$.
"""
struct ViscousBurgersEquation{d} <: AbstractConservationLaw{d,SecondOrder}
    a::NTuple{d,Float64}
    b::Float64
    source_term::AbstractGridFunction{d}
    N_c::Int

    function ViscousBurgersEquation(a::NTuple{d,Float64}, 
        b::Float64, 
        source_term::AbstractGridFunction{d}=NoSourceTerm()) where {d}
        return new{d}(a, b, source_term, 1)
    end
end

const BurgersType{d} = Union{InviscidBurgersEquation{d}, ViscousBurgersEquation{d}}

function InviscidBurgersEquation()
    return InviscidBurgersEquation((1.0,))
end

"""
Evaluate the flux for the inviscid Burgers' equation

`F(u) = a ½u^2`
"""
@inline function physical_flux(conservation_law::BurgersType{d}, 
    u::AbstractMatrix{Float64}) where {d}
    return Tuple(conservation_law.a[m] * 0.5.*u.^2 for m in 1:d)
end

"""
Evaluate the flux for the viscous Burgers' equation

`F(u,q) = a ½u^2 - bq`
"""
@inline function physical_flux(conservation_law::ViscousBurgersEquation{d},
    u::AbstractMatrix{Float64}, q::NTuple{d,AbstractMatrix{Float64}}) where {d}
    return Tuple(conservation_law.a[m]*0.5.*u.^2 - 
        conservation_law.b*q[m] for m in 1:d)
end

"""
Evaluate the Lax-Friedrichs flux for Burgers' equation
`F*(u⁻, u⁺, n) = ½a⋅n(½(u⁻)² + ½(u⁺)²) + ½λ max(|au⁻⋅n|,|au⁺⋅n|)(u⁺ - u⁻)`
"""
@inline function numerical_flux(
    conservation_law::BurgersType{d}, numerical_flux::LaxFriedrichsNumericalFlux, 
    u_in::AbstractMatrix{Float64}, u_out::AbstractMatrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    a_n = sum(conservation_law.a[m].*n[m] for m in 1:d)
    return (0.25*a_n.*(u_in.^2 + u_out.^2) - 
        numerical_flux.λ*0.5*max.(abs.(a_n.*u_out),abs.(a_n.*u_in)).*(u_out - u_in))
end

"""
Evaluate the interface normal solution for the viscous Burgers' equation using the BR1 approach

`U*(u⁻, u⁺, n) = ½(u⁻ + u⁺)n`
"""
@inline function numerical_flux(::ViscousBurgersEquation{d},
    ::BR1,u_in::AbstractMatrix{Float64}, u_out::AbstractMatrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    # average both sides
    u_avg = 0.5*(u_in + u_out)

    return Tuple(u_avg.*n[m] for m in 1:d)
end

"""
Evaluate the numerical flux for the viscous Burgers' equation using the BR1 approach

F*(u⁻, u⁺, q⁻, q⁺, n) = ½(F²(u⁻,q⁻) + F²(u⁺, q⁺))⋅n
"""
@inline function numerical_flux(conservation_law::ViscousBurgersEquation{d},
    ::BR1, u_in::AbstractMatrix{Float64}, u_out::AbstractMatrix{Float64}, 
    q_in::NTuple{d,AbstractMatrix{Float64}}, q_out::NTuple{d,AbstractMatrix{Float64}}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    # average both sides
    q_avg = Tuple(0.5*(q_in[m] + q_out[m]) for m in 1:d)

    # Note that if you give it scaled normal nJf, 
    # the flux will be appropriately scaled by Jacobian too
    # returns vector of length N_f 
    return -1.0*sum(conservation_law.b*q_avg[m] .* n[m] for m in 1:d)
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