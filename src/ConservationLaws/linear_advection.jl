abstract type AbstractConstantLinearAdvectionEquation <: AbstractConservationLaw end

abstract type AbstractVariableLinearAdvectionEquation <: AbstractConservationLaw end

struct ConstantLinearAdvectionEquation1D <: AbstractConstantLinearAdvectionEquation 
    d::Int # spatial dimension
    N_eq::Int # number of equations
    a::Float64 # advection velocity
end

struct ConstantLinearAdvectionEquation2D <: AbstractConstantLinearAdvectionEquation 
    d::Int # spatial dimension
    N_eq::Int # number of equations
    a::Vector{Float64} # advection velocity
end


function ConstantLinearAdvectionEquation1D(a::Float64)
    return ConstantLinearAdvectionEquation1D(1,1,a)
end 


function ConstantLinearAdvectionEquation2D(a::Vector{Float64})
    return ConstantLinearAdvectionEquation2D(2,1,a)
end


function first_order_flux(cl::AbstractConstantLinearAdvectionEquation)
    return u -> u .* cl.a
end