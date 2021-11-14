module Solvers

    using OrdinaryDiffEq: ODEProblem
    using ..SpatialDiscretizations
    
    export compute_residual, make_ode_problem

    function compute_residual(
        spatial_discretization::SpatialDiscretization,
        u::Vector{Float64})

        # use if statement for form

        return nothing
    end

    function make_ode_problem(spatial_discretization::SpatialDiscretization,
        initial_condition::AbstractInitialCondition)

        u0 = evaluate_initial_condition(
            initial_condition,
            spatial_discretization)

        R(u) = compute_residual(spatial_discretization, u)

        return ODEProblem(R, u0)
    end

    include("initial_conditions.jl")
    export AbstractInitialCondition, InitialConditionSine


end