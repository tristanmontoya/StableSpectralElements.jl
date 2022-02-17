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

function ErrorAnalysis(results_path::String, ::ConservationLaw{d,N_eq},
    spatial_discretization::SpatialDiscretization{d}) where {d, N_eq}

    @unpack reference_element, V = 
        spatial_discretization.reference_approximation
    @unpack geometric_factors, mesh, N_el = spatial_discretization
    
    norm = QuadratureL2([Diagonal(reference_element.wq) *
        Diagonal(geometric_factors.J_q[:,k]) for k in 1:N_el])

    return ErrorAnalysis(norm, N_eq, N_el, V, mesh.xyzq, results_path)
end

function analyze(analysis::ErrorAnalysis{QuadratureL2, d}, 
    sol::Array{Float64,3}, exact_solution::AbstractInitialData{d}) where {d}

    @unpack norm, N_eq, N_el, V_err, x_err, results_path = analysis 

    u_exact = evaluate(exact_solution, x_err)
    nodal_error = Tuple(u_exact[:,e,:] - convert(Matrix, V_err * sol[:,e,:])
        for e in 1:N_eq)

    error = [sqrt(sum(dot(nodal_error[e][:,k], norm.WJ[k]*nodal_error[e][:,
        k]) for k in 1:N_el )) for e in 1:N_eq]

    save(string(results_path, "error.jld2"), 
        Dict("error_analysis" => analysis,
            "error" => error))
    
    return error
end