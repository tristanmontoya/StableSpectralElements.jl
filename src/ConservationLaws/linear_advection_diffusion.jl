"""
Linear advection equation

`∂ₜu + ∇⋅(au) = s`
"""
struct LinearAdvectionEquation{d} <: AbstractConservationLaw{d,FirstOrder}
    a::NTuple{d,Float64} 
    source_term::AbstractGridFunction{d}
    N_eq::Int

    function LinearAdvectionEquation(a::NTuple{d,Float64}, 
        source_term::AbstractGridFunction{d}) where {d}
        return new{d}(a, source_term, 1)
    end
end

"""
Linear advection-diffusion equation

`∂ₜu + ∇⋅(au - b∇u) = s`
"""
struct LinearAdvectionDiffusionEquation{d} <: AbstractConservationLaw{d,SecondOrder}
    a::NTuple{d,Float64}
    b::Float64
    source_term::AbstractGridFunction{d}
    N_eq::Int

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
function numerical_flux(conservation_law::AdvectionType{d},
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
function numerical_flux(conservation_law::AdvectionType{d},
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
