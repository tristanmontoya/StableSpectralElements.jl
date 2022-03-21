struct ConstantLinearAdvectionFlux{d} <: AbstractFirstOrderFlux{d,1}
    a::NTuple{d,Float64} # advection velocity
end

struct ConstantLinearAdvectionNumericalFlux{d} <: AbstractFirstOrderNumericalFlux{ConstantLinearAdvectionFlux{d}}
    a::NTuple{d,Float64}
    λ::Float64
end

function linear_advection_equation(a::Float64; λ=1.0)
    return ConservationLaw{1,1}(ConstantLinearAdvectionFlux((a,)), 
        nothing, 
        ConstantLinearAdvectionNumericalFlux((a,), λ), 
        nothing,nothing,nothing)
end 

function linear_advection_equation(a::NTuple{d,Float64}; λ=1.0) where {d}
    return ConservationLaw{d,1}(ConstantLinearAdvectionFlux(a), 
        nothing,
        ConstantLinearAdvectionNumericalFlux(a, λ),
        nothing, nothing,nothing)
end

function physical_flux(flux::ConstantLinearAdvectionFlux{d}, 
    u::Matrix{Float64}) where {d}
    # returns d-tuple of matrices of size N_q x N_eq
    return Tuple(flux.a[m] * u for m in 1:d)
end

function numerical_flux(flux::ConstantLinearAdvectionNumericalFlux{d}, 
    u_in::Matrix{Float64}, u_out::Matrix{Float64}, 
    n::NTuple{d, Vector{Float64}}) where {d}
    # Note that if you give it scaled normal nJf, 
    # the flux will be appropriately scaled by Jacobian too 
    a_n = sum(flux.a[m].*n[m] for m in 1:d) 
    
    # returns vector of length N_zeta 
    return 0.5*a_n.*(u_in + u_out) - 0.5*flux.λ*abs.(a_n).*(u_out - u_in)
end
