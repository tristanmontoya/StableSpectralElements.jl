module Solvers

    using OrdinaryDiffEq: ODEProblem
    using UnPack
    using LinearAlgebra: Diagonal, inv
    using LinearMaps: LinearMap
    using StaticArrays: SMatrix

    using ..ConservationLaws: ConservationLaw, physical_flux, numerical_flux
    using ..SpatialDiscretizations: SpatialDiscretization, apply_to_all_nodes, apply_to_all_dof
    using ..InitialConditions: AbstractInitialData, initial_condition
    
    export AbstractResidualForm, Solver, AbstractPhysicalOperators, PhysicalOperatorsLinear, initialize, semidiscretize, rhs!

    abstract type AbstractResidualForm end
    abstract type AbstractPhysicalOperators{d} end
    
    function initialize(initial_data::AbstractInitialData,
        conservation_law::ConservationLaw{d,N_eq},
        spatial_discretization::SpatialDiscretization{d}) where {d, N_eq}

        f = initial_condition(initial_data, conservation_law)
        
        return apply_to_all_dof(spatial_discretization.physical_projection,
            apply_to_all_nodes(f,
            spatial_discretization.mesh.xyzq, N_eq))
    end

    struct Solver{ResidualForm,d,N_eq}
        conservation_law::ConservationLaw{d,N_eq}
        operators::Vector{<:AbstractPhysicalOperators}
        connectivity::Matrix{Int}
        form::ResidualForm
    end

    struct PhysicalOperatorsLinear{d} <: AbstractPhysicalOperators{d}
        VOL::NTuple{d,LinearMap}
        FAC::LinearMap
        EXTRAPOLATE_SOLUTION::LinearMap
        NORMAL_TRACE::NTuple{d,LinearMap}  # only needed for strong form
        scaled_normal::NTuple{d, Vector{Float64}}
    end

    export StrongConservationForm
    include("strong_conservation_form.jl")
end