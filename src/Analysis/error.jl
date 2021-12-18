abstract type AbstractNorm end

struct QuadratureL2 <: AbstractNorm 
    WJ::Vector{AbstractMatrix}
end

struct RMS <: AbstractNorm end
struct l∞ <: AbstractNorm end

struct ErrorAnalysis{NormType, d} <: AbstractAnalysis
    norm::NormType
    N_eq::Int
    N_el::Int
    V_err::LinearMap
    x_err::NTuple{d, Matrix{Float64}}
    results_path::String
end

function ErrorAnalysis(::ConservationLaw{d,N_eq},
    spatial_discretization::SpatialDiscretization{d},
    results_path::String) where {d, N_eq}
    @unpack reference_element, V = 
        spatial_discretization.reference_approximation
    @unpack geometric_factors, mesh, N_el = spatial_discretization
    
    norm = QuadratureL2([Diagonal(reference_element.wq) *
    Diagonal(geometric_factors.J[:,k]) for k in 1:N_el])

    return ErrorAnalysis(norm, N_eq, N_el, V, mesh.xyzq, results_path)
end

function analyze(error_analysis::ErrorAnalysis{QuadratureL2, d}, 
    sol::Array{Float64,3}, exact_solution::Function) where {d}
    @unpack norm, N_eq, N_el, V_err, x_err = error_analysis 
    nodal_error = Tuple(exact_solution(x_err)[e] - convert(Matrix, V_err * sol[:,e,:]) for e in 1:N_eq)

    #return nodal_error
    return [sqrt(sum(dot(nodal_error[e][:,k], norm.WJ[k]*nodal_error[e][:,k]) 
        for k in 1:error_analysis.N_el )) for e in 1:N_eq]
end

function save_analysis(error_analysis::ErrorAnalysis, error::Vector{Float64})
    save(string(results_path, "error.jld2"), 
        Dict("error_analysis" => error_analysis,
            "error" => error))
end