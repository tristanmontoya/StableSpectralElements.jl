struct ConstantCoefficientLinearAdvectionEquation <: UnsteadyConservationLaw 
    d::Int64
    n_eq::Int64
    a::Array{Float64}
end

function ConstantCoefficientLinearAdvectionEquation(a::Array{Float64})
    d = length(a)
    return ConstantCoefficientLinearAdvectionEquation(d,1,a)
end

function first_order_flux(conservation_law::ConstantCoefficientLinearAdvectionEquation)
    return u ->  u .* conservation_law.a
end