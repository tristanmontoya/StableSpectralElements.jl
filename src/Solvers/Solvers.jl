module Solvers

    using OrdinaryDiffEq: ODEProblem
    using ..ConservationLaws: ConservationLaw
    using ..SpatialDiscretizations: SpatialDiscretization
    using ..InitialConditions: AbstractInitialCondition, evaluate_initial_condition
    
    export compute_residual, solver

    function compute_residual(
        conservation_law::ConservationLaw,
        spatial_discretization::SpatialDiscretization,
        u::Vector{Float64})

        return nothing
    end

    function solver(
        conservation_law::ConservationLaw,spatial_discretization::SpatialDiscretization,
        initial_condition::AbstractInitialCondition, 
        tspan::NTuple{2,Float64})

        u0 = evaluate_initial_condition(
            initial_condition,
            conservation_law,
            spatial_discretization)

        R(u) = compute_residual(conservation_law, 
            spatial_discretization, u)

        return ODEProblem(R, u0, tspan)
    end

end