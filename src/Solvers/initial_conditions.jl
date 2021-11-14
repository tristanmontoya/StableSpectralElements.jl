function evaluate_initial_condition(initial_condition::InitialConditionSine,
    spatial_discretization::SpatialDiscretization)
    # broadcast to quadrature points, then project to solution DOF

    return nothing
end

abstract type AbstractInitialCondition end

struct InitialConditionSine <: AbstractInitialCondition
    k::Float64  # wave number
end