module ConservationLaws

    abstract type AbstractConservationLaw end

    include("linear_advection.jl")

end