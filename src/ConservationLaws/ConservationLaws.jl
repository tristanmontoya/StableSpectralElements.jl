module ConservationLaws

    export AbstractConservationLaw

    abstract type AbstractConservationLaw end

    # linear advection equation
    include("linear_advection.jl")
    export AbstractConstantLinearAdvectionEquation, AbstractVariableLinearAdvectionEquation, ConstantLinearAdvectionEquation1D, ConstantLinearAdvectionEquation2D

end