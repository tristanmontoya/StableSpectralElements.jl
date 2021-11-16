module InitialConditions

    using ..ConservationLaws: ConservationLaw
    using ..SpatialDiscretizations: SpatialDiscretization, l2_projection

    export AbstractInitialData, InitialDataSine, initial_condition
    
    abstract type AbstractInitialData end

    struct InitialDataSine <: AbstractInitialData
        A::Float64  # amplitude
        k::Union{Float64,Vector{Float64}}  # wave number
    end

    function initial_condition(conservation_law::ConservationLaw,
        initial_data::InitialDataSine)
        if conservation_law.d == 1
            return x -> initial_data.A*sin(initial_data.k*x)
        else
            function u0_multi(x)
                u0 = initial_data.A
                for m in 1:d 
                    u0 *= sin(initial_data.k[m]*x[m]) 
                end
                return u0
            end

            return x -> u0_multi(x)
        end
    end

    function initialize(initial_data::AbstractInitialData, 
        conservation_law::ConservationLaw,
        spatial_discretization::SpatialDiscretization)
        
        u0 = initial_condition(conservation_law, initial_data)

        return l2_projection(spatial_discretization, u0)
        
    end

end