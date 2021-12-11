module Analysis

    using LinearMaps: LinearMap
    using LinearAlgebra: Diagonal, dot
    using UnPack
    using StartUpDG: MeshData

    using ..ConservationLaws: ConservationLaw
    using ..Mesh: uniform_periodic_mesh
    using ..SpatialDiscretizations: SpatialDiscretization, ReferenceApproximation
    using ..InitialConditions: AbstractInitialData
    using ..Solvers: AbstractResidualForm, AbstractStrategy, semidiscretize

    export AbstractAnalysis, analyze

    abstract type AbstractAnalysis{d} end
    
    export ErrorAnalysis, AbstractNorm, QuadratureL2, error_analysis
    include("error.jl")

    include("koopman.jl")

end