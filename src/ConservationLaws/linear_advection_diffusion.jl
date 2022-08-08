"""
Linear advection equation

`∂ₜu + ∇⋅(au) = s`
"""
struct LinearAdvectionEquation{d} <: AbstractConservationLaw{d,1,Hyperbolic}
    a::NTuple{d,Float64} 
    source_term::AbstractParametrizedFunction{d}
end

"""
Linear advection-diffusion equation

`∂ₜu + ∇⋅(au - b∇u) = s`
"""
struct LinearAdvectionDiffusionEquation{d} <: AbstractConservationLaw{d,1,Mixed}
    a::NTuple{d,Float64}
    b::Float64
    source_term::AbstractParametrizedFunction{d}
end


struct DiffusionSolution{InitialData} <: AbstractParametrizedFunction{1}
    conservation_law::LinearAdvectionDiffusionEquation
    initial_data::InitialData
    N_eq::Int 
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

function LinearAdvectionDiffusionEquation(a::Float64, b::Float64)
    return LinearAdvectionDiffusionEquation{1}((a,),b,NoSourceTerm{1}())
end

function DiffusionSolution(conservation_law::LinearAdvectionDiffusionEquation, initial_data::AbstractParametrizedFunction)
    return DiffusionSolution(conservation_law,initial_data,1)
end

"""
Evaluate the flux for the linear advection equation 1D linear advection equation

`F(u) = au`
"""
function physical_flux(conservation_law::LinearAdvectionEquation{d}, 
    u::Matrix{Float64}) where {d}
    # returns d-tuple of matrices of size N_q x N_eq
    return Tuple(conservation_law.a[m] * u for m in 1:d)
end

"""
Evaluate the flux for the linear advection-diffusion equation

`F(u,q) = au - bq`
"""
function physical_flux(conservation_law::LinearAdvectionDiffusionEquation{d},
    u::Matrix{Float64}, q::NTuple{d,Matrix{Float64}}) where {d}
    @unpack a, b = conservation_law
    # returns d-tuple of matrices of size N_q x N_eq
    return Tuple(a[m]*u - b*q[m] for m in 1:d)
end

"""
Evaluate the upwind/blended/central advective numerical flux

`F*(u⁻, u⁺, n) = ½a⋅n(u⁻ + u⁺) + ½λ|a⋅n|(u⁺ - u⁻)`
"""
function numerical_flux(conservation_law::Union{LinearAdvectionEquation{d},LinearAdvectionDiffusionEquation{d}},
    numerical_flux::LaxFriedrichsNumericalFlux,
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
Evaluate the central advective numerical flux

`F*(u⁻, u⁺, n) = ½a⋅n(u⁻ + u⁺)`
"""
function numerical_flux(conservation_law::Union{LinearAdvectionEquation{d},LinearAdvectionDiffusionEquation{d}},
    ::EntropyConservativeFlux,
    u_in::Matrix{Float64}, u_out::Matrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    a_n = sum(conservation_law.a[m].*n[m] for m in 1:d)
    return 0.5*a_n.*(u_in + u_out)
end

"""
Evaluate the interface normal solution for the (advection-)diffusion equation using the BR1 approach

`U*(u⁻, u⁺, n) = ½(u⁻ + u⁺)n`
"""
function numerical_flux(::LinearAdvectionDiffusionEquation{d},
    ::BR1,u_in::Matrix{Float64}, u_out::Matrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    # average both sides
    u_avg = 0.5*(u_in + u_out)

    return Tuple(u_avg.*n[m] for m in 1:d)
end

"""
Evaluate the numerical flux for the (advection-)diffusion equation using the BR1 approach

`F*(u⁻, u⁺, q⁻, q⁺, n) = ½(F²(u⁻,q⁻) + F²(u⁺, q⁺))⋅n`
"""

function numerical_flux(conservation_law::LinearAdvectionDiffusionEquation{d},
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

function evaluate(s::DiffusionSolution{InitialDataGaussian{d}}, 
    x::NTuple{d,Float64},t::Float64=0.0) where {d}
    @unpack A, k, x_0 = s.initial_data
    @unpack b = s.conservation_law
    # this seems to be right but maybe plug into equation to check
    r² = sum((x[m] - x_0[m]).^2 for m in 1:d)
    t_0 = k^2/(2.0*b)
    C = A*(t_0/(t+t_0))^(0.5*d)
    return [C*exp.(-r²/(4.0*b*(t_0 + t)))]
end
