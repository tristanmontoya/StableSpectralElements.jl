"""
Inviscid Burgers' equation

`∂ₜu + ∇⋅(a ½u²) = s`
"""
struct InviscidBurgersEquation{d} <: AbstractConservationLaw{d,1,Hyperbolic}
    a::NTuple{d,Float64} 
    source_term::AbstractParametrizedFunction{d}
end

"""
Viscous Burgers' equation (1D)

`∂ₜu + ∇⋅(a ½u² - b∇u) = s`
"""
struct ViscousBurgersEquation{d} <: AbstractConservationLaw{d,1,Hyperbolic}
    a::NTuple{d,Float64}
    b::Float64
    source_term::AbstractParametrizedFunction{d}
end

"""
    function physical_flux(conservation_law::InviscidBurgersEquation, u::Matrix{Float64})

Evaluate the flux for 1D Burgers' equation

`F(u) = a ½u^2`
"""
function physical_flux(conservation_law::LinearAdvectionEquation{d}, 
    u::Matrix{Float64}) where {d}
    # returns d-tuple of matrices of size N_q x N_eq
    return Tuple(conservation_law.a[m] * 0.5.*u.^2 for m in 1:d)
end

"""
    function numerical_flux(conservation_law::Union{InviscidBurgersEquation{d},ViscousBurgersEquation{d}}, numerical_flux::LaxFriedrichsNumericalFlux, u_in::Matrix{Float64}, u_out::Matrix{Float64}, n::NTuple{d, Vector{Float64}})

Evaluate the Lax-Friedrichs flux for Burgers' equation

`F*(u⁻, u⁺, n) = ½a⋅n(½(u⁻)² + ½(u⁺)²) + ½λ max(|au⁻⋅n|,|au⁺⋅n|)(u⁺ - u⁻)`
"""
function numerical_flux(
    conservation_law::Union{InviscidBurgersEquation{d},ViscousBurgersEquation{d}}, numerical_flux::LaxFriedrichsNumericalFlux, 
    u_in::Matrix{Float64}, u_out::Matrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    a_n = sum(conservation_law.a[m].*n[m] for m in 1:d)
    return (0.25*a_n.*(u_in.^2 + u_out.^2).*n[1] - 
        numerical_flux.λ*0.5*max.(abs.(a_n.*u_out),abs.(a_n.*u_in)).*(u_out - u_in))
end

"""
    function numerical_flux(::ViscousBurgersEquation{d}, ::BR1,u_in::Matrix{Float64}, u_out::Matrix{Float64}, n::NTuple{d, Vector{Float64}}

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