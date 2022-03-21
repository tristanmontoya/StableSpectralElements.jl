struct BurgersFlux{d} <: AbstractFirstOrderFlux{d,1} end

function burgers_lax_friedrichs_flux(λ::Float64=1.0)
    return LaxFriedrichsNumericalFlux{BurgersFlux{1}}(λ)
end

function burgers_equation(source_term=nothing,  
    numerical_flux=burgers_lax_friedrichs_flux(1.0),
    two_point_flux=EntropyConservativeFlux{BurgersFlux{1}}())
    return ConservationLaw{1,1}(
        BurgersFlux{1}(), 
        nothing, 
        numerical_flux, 
        nothing,
        source_term,
        two_point_flux)
end 

function physical_flux(::BurgersFlux{d}, u::Matrix{Float64}) where {d}
    # returns d-tuple of matrices of size N_q x N_eq
    return Tuple(0.5*u.^2 for m in 1:d)
end

function two_point_flux(::ConservativeFlux{BurgersFlux{d}}, 
    u::Matrix{Float64}) where {d}
    e = ones(size(u,1))
    u_L = u[:,1]*e'
    u_R = e*(u[:,1])'
    return Tuple(0.25*((u_L .* u_L) + (u_R .* u_R)) for m in 1:d)
end

function two_point_flux(::EntropyConservativeFlux{BurgersFlux{d}}, 
    u::Matrix{Float64}) where {d}
    e = ones(size(u,1))
    u_L = u[:,1]*e'
    u_R = e*(u[:,1])'
    return Tuple((1.0/6.0)*((u_L .* u_L) + (u_L .* u_R) + (u_R .* u_R)) for m in 1:d)
end

function numerical_flux(flux::LaxFriedrichsNumericalFlux{BurgersFlux{1}}, 
    u_in::Matrix{Float64}, u_out::Matrix{Float64}, 
    n::NTuple{1, Vector{Float64}})

    # returns vector of length N_zeta 
    return (0.25*(u_in.^2 + u_out.^2).*n[1] - 
    flux.λ*0.5*max.(abs.(u_out),abs.(u_in)).*(u_out - u_in))
end