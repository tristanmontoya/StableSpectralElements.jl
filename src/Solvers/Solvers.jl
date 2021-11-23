module Solvers

    using OrdinaryDiffEq: ODEProblem
    using LinearMaps: LinearMap
    using ..ConservationLaws: ConservationLaw
    using ..SpatialDiscretizations: SpatialDiscretization, apply_to_all_nodes, apply_to_all_dof
    using ..InitialConditions: AbstractInitialData, initial_condition
    
    export compute_residual, solver, initialize

    function compute_residual(
        conservation_law::ConservationLaw{d, N_eq},
        spatial_discretization::SpatialDiscretization{d},
        u::Array{Float64,3}) where {d, N_eq}
        return nothing
    end

    function initialize(initial_data::AbstractInitialData,
        conservation_law::ConservationLaw{d,N_eq},
        spatial_discretization::SpatialDiscretization{d}) where {d, N_eq}

        f = initial_condition(initial_data, conservation_law)
        
        return apply_to_all_dof(spatial_discretization.projection,
            apply_to_all_nodes(f,
            spatial_discretization.mesh.xyzq, N_eq))
        
    end

    function solver(
        conservation_law::ConservationLaw,spatial_discretization::SpatialDiscretization,
        initial_data::AbstractInitialData, 
        tspan::NTuple{2,Float64})

        u0 = initialize(
            initial_data,
            conservation_law,
            spatial_discretization)

        # acts on multi-d array of shape N_p x N_eq x N_el
        # maybe at some point make ragged to handle varying N_p
        R(u) = compute_residual(conservation_law, 
            spatial_discretization, u)

        # OrdinaryDiffEq supports multi-d operators
        return ODEProblem(R, u0, tspan)
    end
end