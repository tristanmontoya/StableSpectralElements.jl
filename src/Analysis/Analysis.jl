module Analysis

    using LinearMaps: LinearMap
    using LinearAlgebra: Diagonal
    using UnPack

    using ..ConservationLaws
    using ..Mesh
    using ..SpatialDiscretizations
    using ..InitialConditions
    using ..Solvers
    using ..IO

    export AbstractAnalysis, AbstractNorm, QuadratureL2, RMS, l∞, ErrorAnalysis

    abstract type AbstractAnalysis{d} end
    abstract type AbstractNorm end

    struct QuadratureL2 <: AbstractNorm 
        WJ_err::Vector{AbstractMatrix}
    end

    struct RMS <: AbstractNorm end
    struct l∞ <: AbstractNorm end

    struct ErrorAnalysis{d} <: AbstractAnalysis{d}
        norm::AbstractNorm
        N_el::Int
        V_err::LinearMap
        x_err::NTuple{d, Matrix{Float64}}
    end

    function ErrorAnalysis(
        spatial_discretization::SpatialDiscretization{d}) where {d}
        @unpack reference_element, V = 
            spatial_discretization.reference_approximation
        @unpack geometric_factors, mesh, N_el = spatial_discretization
        
        return ErrorAnalysis(QuadratureL2([Diagonal(reference_element.wq) *
            Diagonal(geometric_factors.J[:,k]) for k in 1:N_el]), 
            N_el, V, mesh.xyzq)
    end
        
    export grid_refine
    include("grid_refine.jl")

    include("koopman.jl")

end