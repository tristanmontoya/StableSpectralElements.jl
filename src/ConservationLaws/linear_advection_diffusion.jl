
struct LinearAdvectionEquation{d} <: AbstractConservationLaw{d,1,Hyperbolic}
    a::NTuple{d,Float64}
    source_term::Union{AbstractParametrizedFunction{d}, Nothing}
end

struct LinearAdvectionDiffusionEquation{d} <: AbstractConservationLaw{d,1,Mixed}
    a::NTuple{d,Float64}
    b::Float64
    source_term::Union{AbstractParametrizedFunction{d}, Nothing}
end

function LinearAdvectionEquation(a::NTuple{d,Float64}) where {d}
    return LinearAdvectionEquation{d}(a,nothing)
end

function LinearAdvectionEquation(a::Float64)
    return LinearAdvectionEquation{1}((a,),nothing)
end

function LinearAdvectionDiffusionEquation(a::NTuple{d,Float64}, 
    b::Float64) where {d}
    return LinearAdvectionDiffusionEquation{d}(a,b,nothing)
end

function LinearAdvectionDiffusionEquation(a::Float64, b::Float64) where {d}
    return LinearAdvectionDiffusionEquation{d}((a,),b,nothing)
end

"""
F(U) = aU
"""
function physical_flux(conservation_law::LinearAdvectionEquation{d}, 
    u::Matrix{Float64}) where {d}
    # returns d-tuple of matrices of size N_q x N_eq
    return Tuple(conservation_law.a[m] * u for m in 1:d)
end

"""
F(U,Q) = aU - bQ
"""
function physical_flux(conservation_law::LinearAdvectionDiffusionEquation{d},
    u::Matrix{Float64}, q::Tuple{d,Matrix{Float64}}) where {d}
    # returns d-tuple of matrices of size N_q x N_eq
    return Tuple(conservation_law.a[m] * u - b*q[m] for m in 1:d)
end

struct LinearAdvectionNumericalFlux <: AbstractFirstOrderNumericalFlux
    λ::Float64
end

struct BR1{d} <: AbstractSecondOrderNumericalFlux end

"""
Standard advective numerical flux

F*(U⁻, U⁺, n) = a⋅n(U⁻+U⁺)/2 + λ|a⋅n|(U⁺-U⁻)/2 
"""
function numerical_flux(conservation_law::LinearAdvectionEquation{d},
    numerical_flux::LinearAdvectionNumericalFlux,
    u_in::Matrix{Float64}, u_out::Matrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    # Note that if you give it scaled normal nJf, 
    # the flux will be appropriately scaled by Jacobian too 
    a_n = sum(conservation_law.a[m].*n[m] for m in 1:d)
    
    # returns vector of length N_zeta
    return 0.5*a_n.*(u_in + u_out) - 
        0.5*numerical_flux.λ*abs.(a_n).*(u_out - u_in)
end