module ConservationLaws

    export AbstractFirstOrderFlux, AbstractSecondOrderFlux, AbstractFirstOrderNumericalFlux, AbstractSecondOrderNumericalFlux,ConservationLaw, physical_flux, numerical_flux

    abstract type AbstractFirstOrderFlux{d, N_eq} end
    abstract type AbstractSecondOrderFlux{d, N_eq} end
    abstract type AbstractFirstOrderNumericalFlux{d,N_eq} end
    abstract type AbstractSecondOrderNumericalFlux{d,N_eq} end
    
    struct ConservationLaw{d, N_eq}
        first_order_flux::Union{AbstractFirstOrderFlux,Nothing}
        second_order_flux::Union{AbstractSecondOrderFlux,Nothing}

        first_order_numerical_flux::Union{AbstractFirstOrderNumericalFlux, Nothing}
        second_order_numerical_flux::Union{AbstractSecondOrderNumericalFlux,Nothing}
    end

    export ConstantLinearAdvectionFlux, ConstantLinearAdvectionNumericalFlux,
        linear_advection_equation
    include("linear_advection.jl")
end