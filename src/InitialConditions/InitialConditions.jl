module InitialConditions

    using ..ConservationLaws: ConservationLaw
    using ..SpatialDiscretizations: SpatialDiscretization

    export AbstractInitialCondition, InitialConditionSine, evaluate_initial_condition
    
    abstract type AbstractInitialCondition end

    struct InitialConditionSine <: AbstractInitialCondition
        k::Float64  # wave number
    end

    function evaluate_initial_condition(initial_condition::InitialConditionSine, 
        conservation_law::ConservationLaw,
        spatial_discretization::SpatialDiscretization)
        # broadcast to quadrature points, then project to solution DOF
        return nothing
        
    end

end