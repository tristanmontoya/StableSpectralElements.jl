module SpatialDiscretizations

    using LinearAlgebra: I
    using LinearMaps: LinearMap, UniformScalingMap
    using Reexport
    @reexport using StartUpDG: MeshData, RefElemData
    using ..ConservationLaws

    export AbstractApproximationType, AbstractResidualForm, StrongConservationForm, WeakConservationForm, ReferenceOperators, GeometricFactors, SpatialDiscretization
    
    abstract type AbstractApproximationType end
    abstract type AbstractResidualForm end

    struct StrongConservationForm <: AbstractResidualForm end
    struct WeakConservationForm <: AbstractResidualForm end

    # define in dimension-independent way
    struct ReferenceOperators end
    struct GeometricFactors end

    struct SpatialDiscretization

        conservation_law::ConservationLaw
        mesh::MeshData
        form::AbstractResidualForm
        reference_operators::ReferenceOperators
        geometric_factors::GeometricFactors
        
    end

    # collocated discretizations
    include("collocated.jl")
    export AbstractCollocatedApproximation, DGSEM, DGMulti

end