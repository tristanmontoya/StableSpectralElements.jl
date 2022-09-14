"""
Inviscid Burgers' equation

`∂ₜu + ∇⋅(a ½u²) = s`
"""
struct InviscidBurgersEquation{d} <: AbstractConservationLaw{d,Hyperbolic}
    a::NTuple{d,Float64} 
    source_term::AbstractParametrizedFunction{d}
end

num_equations(::InviscidBurgersEquation) = 1

"""
Viscous Burgers' equation (1D)

`∂ₜu + ∇⋅(a ½u² - b∇u) = s`
"""
struct ViscousBurgersEquation{d} <: AbstractConservationLaw{d,Mixed}
    a::NTuple{d,Float64}
    b::Float64
    source_term::AbstractParametrizedFunction{d}
end

num_equations(::ViscousBurgersEquation) = 1

const BurgersType{d} = Union{InviscidBurgersEquation{d}, ViscousBurgersEquation{d}}

struct BurgersSolution{InitialData,SourceTerm} <: AbstractParametrizedFunction{1}
    initial_data::InitialData
    source_term::SourceTerm
    N_eq::Int
end

"""
Evaluate the flux for the inviscid Burgers' equation

`F(u) = a ½u^2`
"""
function physical_flux(conservation_law::BurgersType{d}, 
    u::Matrix{Float64}) where {d}
    return Tuple(conservation_law.a[m] * 0.5.*u.^2 for m in 1:d)
end

"""
Evaluate the flux for the viscous Burgers' equation

`F(u,q) = a ½u^2 - bq`
"""
function physical_flux(conservation_law::ViscousBurgersEquation{d},
    u::Matrix{Float64}, q::NTuple{d,Matrix{Float64}}) where {d}
    @unpack a, b = conservation_law
    return Tuple(a[m]*0.5.*u.^2 - b*q[m] for m in 1:d)
end

"""
Evaluate the Lax-Friedrichs flux for Burgers' equation
`F*(u⁻, u⁺, n) = ½a⋅n(½(u⁻)² + ½(u⁺)²) + ½λ max(|au⁻⋅n|,|au⁺⋅n|)(u⁺ - u⁻)`
"""
function numerical_flux(
    conservation_law::BurgersType{d}, numerical_flux::LaxFriedrichsNumericalFlux, 
    u_in::Matrix{Float64}, u_out::Matrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    a_n = sum(conservation_law.a[m].*n[m] for m in 1:d)
    return (0.25*a_n.*(u_in.^2 + u_out.^2) - 
        numerical_flux.λ*0.5*max.(abs.(a_n.*u_out),abs.(a_n.*u_in)).*(u_out - u_in))
end

"""
Evaluate the interface normal solution for the viscous Burgers' equation using the BR1 approach

`U*(u⁻, u⁺, n) = ½(u⁻ + u⁺)n`
"""
function numerical_flux(::ViscousBurgersEquation{d},
    ::BR1,u_in::Matrix{Float64}, u_out::Matrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    # average both sides
    u_avg = 0.5*(u_in + u_out)

    return Tuple(u_avg.*n[m] for m in 1:d)
end

"""
Evaluate the numerical flux for the viscous Burgers' equation using the BR1 approach

F*(u⁻, u⁺, q⁻, q⁺, n) = ½(F²(u⁻,q⁻) + F²(u⁺, q⁺))⋅n
"""
function numerical_flux(conservation_law::ViscousBurgersEquation{d},
    ::BR1, u_in::Matrix{Float64}, u_out::Matrix{Float64}, 
    q_in::NTuple{d,Matrix{Float64}}, q_out::NTuple{d,Matrix{Float64}}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    @unpack b = conservation_law

    # average both sides
    q_avg = Tuple(0.5*(q_in[m] + q_out[m]) for m in 1:d)

    # Note that if you give it scaled normal nJf, 
    # the flux will be appropriately scaled by Jacobian too
    # returns vector of length N_f 
    return -1.0*sum(b*q_avg[m] .* n[m] for m in 1:d)
end

function BurgersSolution(initial_data::AbstractParametrizedFunction{1},
    source_term::AbstractParametrizedFunction{1})
    return BurgersSolution(initial_data, source_term, 1)
end

function evaluate(s::BurgersSolution{InitialDataGassner,SourceTermGassner}, 
    x::NTuple{1,Float64},t::Float64=0.0)
    return [sin(s.initial_data.k*(x[1]-t))+s.initial_data.eps]
end