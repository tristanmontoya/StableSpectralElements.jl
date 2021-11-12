module ConservationLaws

    export AbstractFirstOrderFlux, 
        AbstractSecondOrderFlux, 
        ConservationLaw

    abstract type AbstractFirstOrderFlux end
    abstract type AbstractSecondOrderFlux end

    struct ConservationLaw
        d::Int # spatial dimension
        N_eq::Int # number of equations
        first_order_flux::Union{AbstractFirstOrderFlux, Nothing}
        second_order_flux::Union{AbstractSecondOrderFlux, Nothing}
    end

    # linear advection equation
    include("linear_advection.jl")

    export ConstantLinearAdvectionFlux, 
        linear_advection_equation,
        physical_flux

end