module ConservationLaws

    export AbstractFirstOrderFlux, AbstractSecondOrderFlux, ConservationLaw

    abstract type AbstractFirstOrderFlux{d, N_eq} end
    abstract type AbstractSecondOrderFlux{d, N_eq} end

    struct ConservationLaw{d, N_eq}
        first_order_flux::Union{AbstractFirstOrderFlux, Nothing}
        second_order_flux::Union{AbstractSecondOrderFlux, Nothing}
    end

    export ConstantLinearAdvectionFlux, 
        linear_advection_equation,
        physical_flux
    include("linear_advection.jl")
end