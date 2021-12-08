module Analysis

    using LinearMaps: LinearMap
    using LinearAlgebra: Diagonal, dot
    using UnPack
    using StartUpDG: MeshData
    using OrdinaryDiffEq: OrdinaryDiffEqAlgorithm, solve

    using ..ConservationLaws: ConservationLaw
    using ..Mesh: uniform_periodic_mesh
    using ..SpatialDiscretizations: SpatialDiscretization, ReferenceApproximation
    using ..InitialConditions: AbstractInitialData
    using ..Solvers: AbstractResidualForm, AbstractStrategy, semidiscretize

    export AbstractAnalysis, AbstractNorm, QuadratureL2, RMS, l∞, ErrorAnalysis, calculate_error

    abstract type AbstractAnalysis{d} end
    abstract type AbstractNorm end

    struct QuadratureL2 <: AbstractNorm 
        WJ::Vector{AbstractMatrix}
    end

    struct RMS <: AbstractNorm end
    struct l∞ <: AbstractNorm end

    struct ErrorAnalysis{NormType, d} <: AbstractAnalysis{d}
        norm::NormType
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

    function calculate_error(error_analysis::ErrorAnalysis{QuadratureL2, d}, 
        sol::Array{Float64,3}, exact_solution::Function; e::Int=1) where {d}
        @unpack norm, N_el, V_err, x_err = error_analysis 
        err = exact_solution(x_err)[e] - convert(Matrix, V_err * sol[:,e,:])
        return sqrt(sum(dot(err[:,k], norm.WJ[k]*err[:,k]) 
            for k in 1:error_analysis.N_el ))
    end
    
    export grid_refine
    include("grid_refine.jl")

    include("koopman.jl")

end