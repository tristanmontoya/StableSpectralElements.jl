module ConservationLaws

    export AbstractFirstOrderFlux, AbstractSecondOrderFlux, ConservationLaw

    abstract type AbstractFirstOrderFlux end
    abstract type AbstractSecondOrderFlux end

    struct ConservationLaw
        d::Int # spatial dimension
        N_eq::Int # number of equations
        first_order_flux::Union{AbstractFirstOrderFlux, Nothing}
        second_order_flux::Union{AbstractSecondOrderFlux, Nothing}
    end

    export ConstantLinearAdvectionFlux, 
        linear_advection_equation,
        physical_flux
    include("linear_advection.jl")
end