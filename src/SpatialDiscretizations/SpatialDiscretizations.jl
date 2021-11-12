module SpatialDiscretizations

    using StartUpDG
    using LinearAlgebra
    using LinearMaps
    using ..ConservationLaws

    export AbstractApproximationType, AbstractResidualForm, StrongConservationForm, WeakConservationForm, SpatialDiscretization
    
    abstract type AbstractApproximationType end
    abstract type AbstractResidualForm end

    struct StrongConservationForm <: AbstractResidualForm end
    struct WeakConservationForm <: AbstractResidualForm end

    struct SpatialDiscretization
        conservation_law::ConservationLaw
        mesh::MeshData
        reference_element::RefElemData
        approx_type::AbstractApproximationType
        form::AbstractResidualForm
        volume_operator::Union{Tuple{Vararg{Vector{LinearMap{Float64}}}},
            Nothing}
        facet_operator::Union{Vector{LinearMap{Float64}},Nothing}
        solution_to_volume_nodes::Union{LinearMap{Float64},
            LinearMaps.UniformScalingMap}
        solution_to_facet_nodes::LinearMap{Float64}
    end

    # collocated discretizations
    include("collocated.jl")
    export AbstractCollocatedApproximation, DGSEM, DGMulti

end