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
        source_term::AbstractGridFunction{d}=NoSourceTerm{d}()) where {d}
        return new{d}(a, source_term, 1)
    end
end

@doc raw"""
    LinearAdvectionDiffusionEquation(a::NTuple{d,Float64}, b::Float64) where {d}

Define a linear advection-diffusion equation of the form
```math
\partial_t U(\bm{x},t) + \bm{\nabla} \cdot \big( \bm{a} U(\bm{x},t) - b \bm{\nabla} U(\bm{x},t)\big) = 0,
```
with a constant advection velocity $\bm{a} \in \R^d$ and diffusion coefficient $b \in \R^+$. A specialized constructor `LinearAdvectionDiffusionEquation(a::Float64, b::Float64)` is provided for the one-dimensional case.
"""
struct LinearAdvectionDiffusionEquation{d} <: AbstractConservationLaw{d,SecondOrder}
    a::NTuple{d,Float64}
    b::Float64
    source_term::AbstractGridFunction{d}
    N_c::Int

    function LinearAdvectionDiffusionEquation(
        a::NTuple{d,Float64}, b::Float64, 
        source_term::AbstractGridFunction{d}=NoSourceTerm{d}()) where {d}
        return new{d}(a, b, source_term, 1)
    end
end

const AdvectionType{d} = Union{LinearAdvectionEquation{d}, LinearAdvectionDiffusionEquation{d}}

LinearAdvectionEquation(a::Float64) = LinearAdvectionEquation((a,))

LinearAdvectionDiffusionEquation(a::Float64, b::Float64) = LinearAdvectionDiffusionEquation((a,),b)


"""
Evaluate the flux for the linear advection equation 1D linear advection equation

`F(u) = au`
"""
function physical_flux!(
    f::AbstractArray{Float64,3},
    conservation_law::LinearAdvectionEquation{d}, 
    u::AbstractMatrix{Float64}) where {d}

    @inbounds for m in 1:d
        f[:,:,m] .= conservation_law.a[m] .* u
    end
    return f
end

"""
Evaluate the flux for the linear advection-diffusion equation

`F(u,q) = au - bq`
"""
function physical_flux!(f::AbstractArray{Float64,3},
    conservation_law::LinearAdvectionDiffusionEquation{d},
    u::AbstractMatrix{Float64}, q::AbstractArray{Float64,3}) where {d}

    @inbounds for m in 1:d
        f[:,:,m] .= conservation_law.a[m] .* u .- conservation_law.b .* q[:,:,m]
    end
end

"""
Evaluate the upwind/blended/central advective numerical flux
`F*(u⁻, u⁺, n) = ½a⋅n(u⁻ + u⁺) + ½λ|a⋅n|(u⁺ - u⁻)`
"""
function numerical_flux!(f_star::AbstractMatrix{Float64},
    conservation_law::AdvectionType{d},
    numerical_flux::LaxFriedrichsNumericalFlux,
    u_in::AbstractMatrix{Float64}, u_out::AbstractMatrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    a_n = sum(conservation_law.a[m].*n[m] for m in 1:d)
    
    f_star .= 0.5.*(a_n.*(u_in .+ u_out) .- 
        numerical_flux.λ.*abs.(a_n).*(u_out .- u_in))
end

"""
Evaluate the central advective numerical flux

`F*(u⁻, u⁺, n) = ½a⋅n(u⁻ + u⁺)`
"""
function numerical_flux!(
    f_star::AbstractMatrix{Float64},
    conservation_law::AdvectionType{d},
    ::EntropyConservativeFlux,
    u_in::AbstractMatrix{Float64}, u_out::AbstractMatrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    f_star .= 0.5*sum(conservation_law.a[m].*n[m] for m in 1:d)*(u_in .+ u_out)
end

"""
Evaluate the interface normal solution for the (advection-)diffusion equation using the BR1 approach

`U*(u⁻, u⁺, n) = ½(u⁻ + u⁺)n`
"""
function numerical_flux!(u_nstar::AbstractArray{Float64,3},
    ::LinearAdvectionDiffusionEquation{d},
    ::BR1, u_in::AbstractMatrix{Float64}, u_out::AbstractMatrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    u_avg = 0.5*(u_in .+ u_out)

    @inbounds for m in 1:d
        u_nstar[:,:,m] .= u_avg.*n[m]
    end
end

"""
Evaluate the numerical flux for the (advection-)diffusion equation using the BR1 approach (note that this gets added to the f_star from the advective flux)

`F*(u⁻, u⁺, q⁻, q⁺, n) = ½(F²(u⁻,q⁻) + F²(u⁺, q⁺))⋅n`
"""
function numerical_flux!(f_star::AbstractMatrix{Float64},
    conservation_law::LinearAdvectionDiffusionEquation{d},
    ::BR1, u_in::AbstractMatrix{Float64}, u_out::AbstractMatrix{Float64}, 
    q_in::AbstractArray{Float64,3}, q_out::AbstractArray{Float64,3}, 
    n::NTuple{d, Vector{Float64}}) where {d}

    # average both sides
    minus_q_avg = -0.5*(q_in .+ q_out)
    f_star .+= sum(conservation_law.b * minus_q_avg[:,:,m] .* n[m] for m in 1:d)
end

function evaluate(
    exact_solution::ExactSolution{d,LinearAdvectionEquation{d}, <:AbstractGridFunction{d},NoSourceTerm{d}},
    x::NTuple{d,Float64},t::Float64=0.0) where {d}

    (; initial_data, conservation_law) = exact_solution
    
    if !exact_solution.periodic 
        z = Tuple(x[m] - conservation_law.a[m]*t for m in 1:d)
    else z = x end

    return evaluate(initial_data,z)
end

function evaluate(
    exact_solution::ExactSolution{d,LinearAdvectionDiffusionEquation{d}, InitialDataGaussian{d},NoSourceTerm{d}},
    x::NTuple{d,Float64},t::Float64=0.0) where {d}

    (; A, σ, x₀) = exact_solution.initial_data
    (; a, b) = exact_solution.conservation_law

    if !exact_solution.periodic z = Tuple(x[m] - a[m]*t for m in 1:d)
    else z = x end

    r² = sum((z[m] - x₀[m]).^2 for m in 1:d)
    t₀ = σ^2/(2.0*b)
    return [A*(t₀/(t+t₀))^(0.5*d)*exp.(-r²/(4.0*b*(t₀ + t)))]
end

function evaluate(
    exact_solution::ExactSolution{d,LinearAdvectionDiffusionEquation{d}, InitialDataSine{d},NoSourceTerm{d}},
    x::NTuple{d,Float64},t::Float64=0.0) where {d}
    
    (; A, k) = exact_solution.initial_data
    (; a, b) = exact_solution.conservation_law

    if !exact_solution.periodic z = Tuple(x[m] - a[m]*t for m in 1:d)
    else z = x end

    return evaluate(exact_solution.initial_data,z) * 
        exp(-b*sum(k[m]^2 for m in 1:d)*t)
end
