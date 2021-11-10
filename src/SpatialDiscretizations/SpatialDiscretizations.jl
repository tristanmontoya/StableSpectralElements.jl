module SpatialDiscretizations

    using StartUpDG
    using ..ConservationLaws

    export AbstractSpatialDiscretization, AbstractApproximationType, AbstractResidualForm, StrongConservationForm, WeakConservationForm, SpatialDiscretization1D
    
    abstract type AbstractSpatialDiscretization end
    abstract type AbstractApproximationType end
    abstract type AbstractResidualForm end
    struct StrongConservationForm <: AbstractResidualForm end
    struct WeakConservationForm <: AbstractResidualForm end
    
    struct SpatialDiscretization1D <: AbstractSpatialDiscretization
        conservation_law::AbstractConservationLaw
        mesh::MeshData
        reference_element::RefElemData
        approx_type::AbstractApproximationType
        form::AbstractResidualForm
        operators::Nothing #placeholder
    end

    # collocated discretizations
    include("collocated.jl")
    export CollocatedLG, CollocatedLGL

end