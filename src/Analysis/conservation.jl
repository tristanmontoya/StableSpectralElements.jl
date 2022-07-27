abstract type ConservationAnalysis <: AbstractAnalysis end

struct PrimaryConservationAnalysis{d} <: ConservationAnalysis
    WJ::Vector{<:AbstractMatrix}
    N_eq::Int
    N_el::Int
    V::LinearMap
    results_path::String
    analysis_path::String
    dict_name::String
end

struct EnergyConservationAnalysis{d} <: ConservationAnalysis
    WJ::Vector{<:AbstractMatrix}
    N_eq::Int
    N_el::Int
    V::LinearMap
    results_path::String
    analysis_path::String
    dict_name::String
end

function PrimaryConservationAnalysis(results_path::String,
    conservation_law::AbstractConservationLaw, 
    spatial_discretization::SpatialDiscretization{d}, name="primary_conservation_analysis") where {d}

    analysis_path = new_path(string(results_path, name, "/"))
    _, N_eq, N_el = get_dof(spatial_discretization, conservation_law)
  
    @unpack W, V =  spatial_discretization.reference_approximation
    @unpack geometric_factors, mesh, N_el = spatial_discretization
    
    WJ = [Matrix(W) * Diagonal(geometric_factors.J_q[:,k]) for k in 1:N_el]

    return PrimaryConservationAnalysis{d}(
        WJ, N_eq, N_el, V, results_path, analysis_path, "conservation.jld2")
end

function EnergyConservationAnalysis(results_path::String,
    conservation_law::AbstractConservationLaw,
    spatial_discretization::SpatialDiscretization{d},
    name="energy_conservation_analysis") where {d}

    analysis_path = new_path(string(results_path, name, "/"))
    _, N_eq, N_el = get_dof(spatial_discretization, conservation_law)

    @unpack W, V =  spatial_discretization.reference_approximation
    @unpack geometric_factors, mesh, N_el = spatial_discretization
    
    WJ = [Matrix(W) * Diagonal(geometric_factors.J_q[:,k]) for k in 1:N_el]

    return EnergyConservationAnalysis{d}(
        WJ, N_eq, N_el, V ,results_path, analysis_path, "energy.jld2")
end

function evaluate_conservation(
    analysis::PrimaryConservationAnalysis, 
    sol::Array{Float64,3})
    @unpack WJ, N_eq, N_el, V = analysis 

    return [sum(sum(WJ[k]*V*sol[:,e,k]) 
        for k in 1:N_el) for e in 1:N_eq]
end

function evaluate_conservation(
    analysis::EnergyConservationAnalysis, 
    sol::Array{Float64,3})
    @unpack WJ, N_eq, N_el, V = analysis 

    return [0.5*sum(sol[:,e,k]'*V'*WJ[k]*V*sol[:,e,k] 
        for k in 1:N_el) for e in 1:N_eq]
end

function analyze(analysis::ConservationAnalysis, 
    initial_time_step::Union{Int,String}=0, 
    final_time_step::Union{Int,String}="final")
    
    @unpack results_path, analysis_path, dict_name = analysis

    u_0, t_0 = load_solution(results_path, initial_time_step)
    u_f, t_f = load_solution(results_path, final_time_step)

    initial = evaluate_conservation(analysis,u_0)
    final = evaluate_conservation(analysis,u_f)
    difference = final .- initial

    save(string(analysis_path, dict_name), 
        Dict("analysis" => analysis,
            "initial" => initial,
            "final" => final,
            "difference" => difference,
            "t_0" => t_0,
            "t_f" => t_f))

    return initial, final, difference
end