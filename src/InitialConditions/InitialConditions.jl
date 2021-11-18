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

    function initial_condition(initial_data::InitialDataSine{d},
        conservation_law::ConservationLaw{d,N_eq}) where {d, N_eq}
        return x -> Tuple(initial_data.A*prod(Tuple(sin.(initial_data.k[m]*x[m])
            for m in 1:d)) 
            for e in 1:N_eq)
    end

    function initialize(initial_data::AbstractInitialData,
        conservation_law::ConservationLaw,
        spatial_discretization::SpatialDiscretization)
        
        # make initial data into function returninng tuple of length N_eq
        u0 = initial_condition(initial_data, conservation_law)

        # project onto solution DOF
        return l2_projection(spatial_discretization, u0)
        
    end

end