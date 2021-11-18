struct ConstantLinearAdvectionFlux{d} <: AbstractFirstOrderFlux{d,1}
    a::NTuple{d,Float64} # advection velocity
end

function linear_advection_equation(a::Float64)
    return ConservationLaw{1,1}(ConstantLinearAdvectionFlux{1}((a,)), nothing)
end 

function linear_advection_equation(a::NTuple{d,Float64}) where {d}
    return ConservationLaw{d,1}(ConstantLinearAdvectionFlux{d}(a), nothing)
end

function physical_flux(flux::ConstantLinearAdvectionFlux{d}, u) where {d}
    return (flux.a[m] * u for m in 1:d)
end