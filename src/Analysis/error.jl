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
    results_path::String
end

function ErrorAnalysis(spatial_discretization::SpatialDiscretization{d},
    results_path::String) where {d}
    @unpack reference_element, V = 
        spatial_discretization.reference_approximation
    @unpack geometric_factors, mesh, N_el = spatial_discretization
    
    norm = QuadratureL2([Diagonal(reference_element.wq) *
    Diagonal(geometric_factors.J[:,k]) for k in 1:N_el])

    return ErrorAnalysis(norm, N_el, V, mesh.xyzq, results_path)
end

function analyze(error_analysis::ErrorAnalysis{QuadratureL2, d}, 
    sol::Array{Float64,3}, exact_solution::Function; e::Int=1) where {d}
    @unpack norm, N_el, V_err, x_err = error_analysis 
    nodal_error = exact_solution(x_err)[e] - convert(Matrix, V_err * sol[:,e,:])
    error_norm = sqrt(sum(dot(nodal_error[:,k], norm.WJ[k]*nodal_error[:,k]) 
        for k in 1:error_analysis.N_el ))
    # TODO: write error to file 
    return error_norm
end