
struct LinearAdvectionEquation{d} <: AbstractConservationLaw{d,1,Hyperbolic}
    a::NTuple{d,Float64} 
    source_term::AbstractParametrizedFunction{d}
end

struct LinearAdvectionDiffusionEquation{d} <: AbstractConservationLaw{d,1,Mixed}
    a::NTuple{d,Float64}
    b::Float64
    source_term::AbstractParametrizedFunction{d}
end

function LinearAdvectionEquation(a::NTuple{d,Float64}) where {d}
    return LinearAdvectionEquation{d}(a,NoSourceTerm{d}())
end

function LinearAdvectionEquation(a::Float64)
    return LinearAdvectionEquation{1}((a,),NoSourceTerm{1}())
end

function LinearAdvectionDiffusionEquation(a::NTuple{d,Float64}, 
    b::Float64) where {d}
    return LinearAdvectionDiffusionEquation{d}(a,b,NoSourceTerm{d}())
end

function LinearAdvectionDiffusionEquation(a::Float64, b::Float64) where {d}
    return LinearAdvectionDiffusionEquation{d}((a,),b,NoSourceTerm{d}())
end

"""
    function physical_flux(conservation_law::LinearAdvectionEquation{d}, u::Matrix{Float64})

Evaluate the flux for 1D linear advection-diffusion equation 1D linear advection equation

F(U) = aU
"""
function physical_flux(conservation_law::LinearAdvectionEquation{d}, 
    u::Matrix{Float64}) where {d}
    # returns d-tuple of matrices of size N_q x N_eq
    return Tuple(conservation_law.a[m] * u for m in 1:d)
end

"""
    function physical_flux(conservation_law::LinearAdvectionDiffusionEquation{d}u::Matrix{Float64}, q::Tuple{d,Matrix{Float64}})

Evaluate the flux for 1D linear advection-diffusion equation

F(U,Q) = aU - bQ
"""
function physical_flux(conservation_law::LinearAdvectionDiffusionEquation{d},
    u::Matrix{Float64}, q::Array{Float64,4}) where {d}
    # returns d-tuple of matrices of size N_q x N_eq
    return Tuple(conservation_law.a[m] * u - b*q[:,:,m] for m in 1:d)
end

struct LinearAdvectionNumericalFlux <: AbstractFirstOrderNumericalFlux
    λ::Float64
end

struct BR1{d} <: AbstractSecondOrderNumericalFlux end

"""
    numerical_flux(conservation_law::LinearAdvectionEquation{d},numerical_flux::LinearAdvectionNumericalFlux, u_in::Matrix{Float64}, u_out::Matrix{Float64}, n::NTuple{d, Vector{Float64}})

Evaluate the standard advective numerical flux

F*(U⁻, U⁺, n) = 1/2 a⋅n(U⁻ + U⁺) + λ/2 |a⋅n|(U⁺ - U⁻)
"""
function numerical_flux(conservation_law::LinearAdvectionEquation{d},
    numerical_flux::LinearAdvectionNumericalFlux,
    u_in::Matrix{Float64}, u_out::Matrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    # Note that if you give it scaled normal nJf, 
    # the flux will be appropriately scaled by Jacobian too 
    a_n = sum(conservation_law.a[m].*n[m] for m in 1:d)
    
    # returns vector of length N_f
    return 0.5*a_n.*(u_in + u_out) - 
        0.5*numerical_flux.λ*abs.(a_n).*(u_out - u_in)
end

"""
    function numerical_flux(::LinearAdvectionDiffusionEquation{d}, ::BR1,u_in::Matrix{Float64}, u_out::Matrix{Float64}, n::NTuple{d, Vector{Float64}}

Evaluate the interface normal solution for the (advection-)diffusion equation using the BR1 approach

U*(U⁻, U⁺, n) = 1/2 (U⁻ + U⁺)n
"""

function numerical_flux(::LinearAdvectionDiffusionEquation{d},
    ::BR1,u_in::Matrix{Float64}, u_out::Matrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    # average both sides
    u_avg = 0.5*(u_in + u_out)

    # Note that if you give it scaled normal nJf, 
    # the flux will be appropriately scaled by Jacobian too
    # returns tuple of vectors of length N_f 
    return Tuple(u_avg.*n[m] for m in 1:d)
end

"""
    function numerical_flux(::LinearAdvectionDiffusionEquation{d}, ::BR1,u_in::Matrix{Float64}, u_out::Matrix{Float64}, n::NTuple{d, Vector{Float64}}

Evaluate the numerical flux for the (advection-)diffusion equation using the BR1 approach

F*(U⁻, U⁺, Q⁻, Q⁺, n) = 1/2 (F²(U⁻, Q⁻) + F²(U⁺, Q⁺))⋅n
"""

function numerical_flux(::LinearAdvectionDiffusionEquation{d},
    ::BR1, u_in::Matrix{Float64}, u_out::Matrix{Float64}, 
    q_in::Array{3,Float64}, q_out::Array{3,Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    # average both sides
    q_avg = 0.5*(q_in + q_out)

    # Note that if you give it scaled normal nJf, 
    # the flux will be appropriately scaled by Jacobian too
    # returns vector of length N_f 
    return sum(b*q_avg[:,:,m] * n[m] for m in 1:d)
end