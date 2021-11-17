module InitialConditions

    using ..ConservationLaws: ConservationLaw
    using ..SpatialDiscretizations: SpatialDiscretization, l2_projection

    export AbstractInitialData, InitialDataSine, initial_condition
    
    abstract type AbstractInitialData end

    struct InitialDataSine{d} <: AbstractInitialData
        A::Float64  # amplitude
        k::NTuple{d,Float64}  # wave number in each direction
    end

    function InitialDataSine(A::Float64, k::Float64)
        return InitialDataSine{1}(A,(k,))
    end

    function initial_condition(initial_data::InitialDataSine{d}) where {d}
        return x -> initial_data.A*prod((sin(initial_data.k*x[m]) for m in 1:d))
    end

    function initialize(initial_data::AbstractInitialData, 
        conservation_law::ConservationLaw,
        spatial_discretization::SpatialDiscretization)
        
        u0 = initial_condition(initial_data)

        return l2_projection(spatial_discretization, u0)
        
    end

end