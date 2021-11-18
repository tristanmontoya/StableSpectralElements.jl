module Solvers

    using OrdinaryDiffEq: ODEProblem
    using ..ConservationLaws: ConservationLaw
    using ..SpatialDiscretizations: SpatialDiscretization
    using ..InitialConditions: AbstractInitialData, initial_condition, initialize
    
    export compute_residual, solver

    function compute_residual(
        conservation_law::ConservationLaw{d, N_eq},
        spatial_discretization::SpatialDiscretization{d},
        u::Array{Float64,3}) where {d, N_eq}

        return nothing
    end

    function solver(
        conservation_law::ConservationLaw,spatial_discretization::SpatialDiscretization,
        initial_data::AbstractInitialData, 
        tspan::NTuple{2,Float64})

        initialize(
            initial_data,
            conservation_law,
            spatial_discretization)

        # acts on multi-d array of shape N_p x N_eq x N_el
        # maybe at some point make ragged to handle varying N_p
        R(u) = compute_residual(conservation_law, 
            spatial_discretization, u)

        # OrdinaryDiffEq supports multi-d operators
        return ODEProblem(R, 0, tspan)
    end
end