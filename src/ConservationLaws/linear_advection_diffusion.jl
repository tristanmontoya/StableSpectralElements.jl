@doc raw"""
    LinearAdvectionEquation(a::NTuple{d,Float64}) where {d}

Define a linear advection equation of the form
```math
\partial_t U(\bm{x},t) + \bm{\nabla} \cdot \big( \bm{a} U(\bm{x},t) \big) = 0,
```
with a constant advection velocity $\bm{a} \in \R^d$. A specialized constructor `LinearAdvectionEquation(a::Float64)` is provided for the one-dimensional case.
"""
struct LinearAdvectionEquation{d} <: AbstractConservationLaw{d,FirstOrder}
    a::NTuple{d,Float64} 
    source_term::AbstractGridFunction{d}
    N_c::Int

    function LinearAdvectionEquation(a::NTuple{d,Float64}, 
        source_term::AbstractGridFunction{d}) where {d}
        return new{d}(a, source_term, 1)
    end
end

@doc raw"""
    LinearAdvectionDiffusionEquation(a::NTuple{d,Float64}, b::Float64) where {d}

Define a linear advection-diffusion equation of the form
```math
\partial_t U(\bm{x},t) + \bm{\nabla} \cdot \big( \bm{a} U(\bm{x},t) - b \bm{\nabla} U(\bm{x},t)\big) = 0,
```
with a constant advection velocity $\bm{a} \in \R^d$ and diffusion coefficient $b \in \R^+$. A specialized constructor `LinearAdvectionEquation(a::Float64, b::Float64)` is provided for the one-dimensional case.
"""
struct LinearAdvectionDiffusionEquation{d} <: AbstractConservationLaw{d,SecondOrder}
    a::NTuple{d,Float64}
    b::Float64
    source_term::AbstractGridFunction{d}
    N_c::Int

    function LinearAdvectionDiffusionEquation(a::NTuple{d,Float64}, 
        b::Float64, source_term::AbstractGridFunction{d}) where {d}
        return new{d}(a, b, source_term, 1)
    end
end

const AdvectionType{d} = Union{LinearAdvectionEquation{d}, LinearAdvectionDiffusionEquation{d}}

function LinearAdvectionEquation(a::NTuple{d,Float64}) where {d}
    return LinearAdvectionEquation(a,NoSourceTerm{d}())
end

function LinearAdvectionEquation(a::Float64)
    return LinearAdvectionEquation((a,),NoSourceTerm{1}())
end

function LinearAdvectionDiffusionEquation(a::NTuple{d,Float64}, 
    b::Float64) where {d}
    return LinearAdvectionDiffusionEquation(a,b,NoSourceTerm{d}())
end

function LinearAdvectionDiffusionEquation(a::Float64, b::Float64)
    return LinearAdvectionDiffusionEquation((a,),b,NoSourceTerm{1}())
end

"""
Evaluate the flux for the linear advection equation 1D linear advection equation

`F(u) = au`
"""
@inline function physical_flux(conservation_law::LinearAdvectionEquation{d}, 
    u::AbstractMatrix{Float64}) where {d}
    # returns d-tuple of matrices of size N_q x N_c
    return Tuple(conservation_law.a[m] * u for m in 1:d)
end

"""
Evaluate the flux for the linear advection-diffusion equation

`F(u,q) = au - bq`
"""
@inline function physical_flux(
    conservation_law::LinearAdvectionDiffusionEquation{d},
    u::AbstractMatrix{Float64}, q::NTuple{d,AbstractMatrix{Float64}}) where {d}
    # returns d-tuple of matrices of size N_q x N_c
    return Tuple(conservation_law.a[m]*u .- 
        conservation_law.b*q[m] for m in 1:d)
end

"""
Evaluate the upwind/blended/central advective numerical flux
`F*(u⁻, u⁺, n) = ½a⋅n(u⁻ + u⁺) + ½λ|a⋅n|(u⁺ - u⁻)`
"""
@inline function numerical_flux(conservation_law::AdvectionType{d},
    numerical_flux::LaxFriedrichsNumericalFlux,
    u_in::AbstractMatrix{Float64}, u_out::AbstractMatrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    # Note that if you give it scaled normal nJf, 
    # the flux will be appropriately scaled by Jacobian too 
    a_n = sum(conservation_law.a[m].*n[m] for m in 1:d)
    
    # returns vector of length N_f
    return 0.5*a_n.*(u_in .+ u_out) .- 
        0.5*numerical_flux.λ*abs.(a_n).*(u_out .- u_in)
end

"""
Evaluate the central advective numerical flux

`F*(u⁻, u⁺, n) = ½a⋅n(u⁻ + u⁺)`
"""
@inline function numerical_flux(conservation_law::AdvectionType{d},
    ::EntropyConservativeFlux,
    u_in::AbstractMatrix{Float64}, u_out::AbstractMatrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    a_n = sum(conservation_law.a[m].*n[m] for m in 1:d)
    return 0.5*a_n.*(u_in .+ u_out)
end

"""
Evaluate the interface normal solution for the (advection-)diffusion equation using the BR1 approach

`U*(u⁻, u⁺, n) = ½(u⁻ + u⁺)n`
"""
@inline function numerical_flux(::LinearAdvectionDiffusionEquation{d},
    ::BR1,u_in::AbstractMatrix{Float64}, u_out::AbstractMatrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    # average both sides
    u_avg = 0.5*(u_in .+ u_out)

    return Tuple(u_avg.*n[m] for m in 1:d)
end

"""
Evaluate the numerical flux for the (advection-)diffusion equation using the BR1 approach

`F*(u⁻, u⁺, q⁻, q⁺, n) = ½(F²(u⁻,q⁻) + F²(u⁺, q⁺))⋅n`
"""

@inline function numerical_flux(conservation_law::LinearAdvectionDiffusionEquation{d},
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
    exact_solution::ExactSolution{d,LinearAdvectionEquation{d}, <:AbstractGridFunction{d},NoSourceTerm{d}},
    x::NTuple{d,Float64},t::Float64=0.0) where {d}
    @unpack initial_data, conservation_law = exact_solution
    
    if !exact_solution.periodic
        z = Tuple(x[m] - conservation_law.a[m]*t for m in 1:d)
    else
        z = x
    end

    return evaluate(initial_data,z)
end

function evaluate(
    exact_solution::ExactSolution{d,LinearAdvectionDiffusionEquation{d}, InitialDataGaussian{d},NoSourceTerm{d}},
    x::NTuple{d,Float64},t::Float64=0.0) where {d}
    @unpack A, σ, x₀ = exact_solution.initial_data
    @unpack a, b = exact_solution.conservation_law

    if !exact_solution.periodic
        z = Tuple(x[m] - a[m]*t for m in 1:d)
    else
        z = x
    end

    r² = sum((z[m] - x₀[m]).^2 for m in 1:d)
    t₀ = σ^2/(2.0*b)
    return [A*(t₀/(t+t₀))^(0.5*d)*exp.(-r²/(4.0*b*(t₀ + t)))]
end

function evaluate(
    exact_solution::ExactSolution{d,LinearAdvectionDiffusionEquation{d}, InitialDataSine{d},NoSourceTerm{d}},
    x::NTuple{d,Float64},t::Float64=0.0) where {d}
    @unpack A, k = exact_solution.initial_data
    @unpack a, b = exact_solution.conservation_law

    if !exact_solution.periodic
        z = Tuple(x[m] - a[m]*t for m in 1:d)
    else
        z = x
    end

    return evaluate(exact_solution.initial_data,z) * 
        exp(-b*sum(k[m]^2 for m in 1:d)*t)
end
