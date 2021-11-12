struct ConstantLinearAdvectionFlux <: AbstractFirstOrderFlux
    a::Union{Float64,Vector{Float64}} # advection velocity
end

function linear_advection_equation(a::Float64)
    return ConservationLaw(1, 1, ConstantLinearAdvectionFlux(a), nothing)
end 

function linear_advection_equation(a::Vector{Float64})
    return ConservationLaw(
        size(a,1), 1, ConstantLinearAdvectionFlux(a), nothing)
end

physical_flux(flux::ConstantLinearAdvectionFlux, u) = flux.a * u