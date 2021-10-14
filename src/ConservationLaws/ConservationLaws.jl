module ConservationLaws

    export ConstantCoefficientLinearAdvectionEquation

    abstract type AbstractConservationLaw end
    abstract type SteadyConservationLaw  <: AbstractConservationLaw end
    abstract type UnsteadyConservationLaw  <: AbstractConservationLaw end

    include("linear_advection.jl")

end