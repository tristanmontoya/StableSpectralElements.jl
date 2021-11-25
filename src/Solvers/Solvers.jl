module Solvers

    using OrdinaryDiffEq: ODEProblem
    using LinearAlgebra: diagm 
    using LinearMaps: LinearMap
    using StaticArrays: SMatrix
    using ..ConservationLaws: ConservationLaw
    using ..SpatialDiscretizations: SpatialDiscretization, apply_to_all_nodes, apply_to_all_dof
    using ..InitialConditions: AbstractInitialData, initial_condition
    
    export AbstractResidualForm, initialize, solver, element_residual

    abstract type AbstractResidualForm end
    abstract type AbstractOperatorStorage end
    struct ReferenceOperatorStorage <: AbstractOperatorStorage end
    struct PhysicalOperatorStorage <: AbstractOperatorStorage end

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
        tspan::NTuple{2,Float64};
        form::AbstractResidualForm=StrongConservationForm(),
        operator_storage::AbstractOperatorStorage=PhysicalOperatorStorage())

        u0 = initialize(
            initial_data,
            conservation_law,
            spatial_discretization)

        # local residual acts on array of shape N_p x N_eq x N_el (global)
        # maybe at some point make ragged to handle varying N_p
        # MPI parallelization would involve only sharing part of this array
        # shared-memory would have all with access to u
        # returns a N_p x N_eq array (local)
        res = element_residual(conservation_law, 
        spatial_discretization, form, operator_storage)

        # we then assemble the global residual
        R(u) = map(k -> res(u,k), 1:spatial_discretization.N_el)

        # OrdinaryDiffEq supports multi-d operators
        return ODEProblem(R, u0, tspan)
    end

    export StrongConservationForm
    include("strong_conservation_form.jl")
end