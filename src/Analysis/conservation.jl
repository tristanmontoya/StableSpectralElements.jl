abstract type ConservationAnalysis <: AbstractAnalysis end

"""
Evaluate change in ∫udx  
"""
struct PrimaryConservationAnalysis <: ConservationAnalysis
    WJ::Vector{<:AbstractMatrix}
    N_eq::Int
    N_el::Int
    V::LinearMap
    results_path::String
    analysis_path::String
    dict_name::String
end

"""
Evaluate change in  ∫½u²dx  
"""
struct EnergyConservationAnalysis <: ConservationAnalysis
    WJ::Vector{<:AbstractMatrix}
    N_eq::Int
    N_el::Int
    V::LinearMap
    results_path::String
    analysis_path::String
    dict_name::String
end

struct ConservationAnalysisResults <:AbstractAnalysisResults
    t::Vector{Float64}
    E::Matrix{Float64}
end

function PrimaryConservationAnalysis(results_path::String,
    conservation_law::AbstractConservationLaw, 
    spatial_discretization::SpatialDiscretization{d}, name="primary_conservation_analysis") where {d}

    analysis_path = new_path(string(results_path, name, "/"))
    _, N_eq, N_el = get_dof(spatial_discretization, conservation_law)
  
    @unpack W, V =  spatial_discretization.reference_approximation
    @unpack geometric_factors, mesh, N_el = spatial_discretization
    
    WJ = [Matrix(W) * Diagonal(geometric_factors.J_q[:,k]) for k in 1:N_el]

    return PrimaryConservationAnalysis(
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

    return EnergyConservationAnalysis(
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

function analyze(analysis::ConservationAnalysis,
    time_steps::Vector{Int})

    @unpack results_path, N_eq, dict_name = analysis
    N_t = length(time_steps)
    t = Vector{Float64}(undef,N_t)
    E = Matrix{Float64}(undef,N_t, N_eq)
    for i in 1:N_t
        u, t[i] = load_solution(results_path, time_steps[i])
        E[i,:] = evaluate_conservation(analysis, u)
    end

    results = ConservationAnalysisResults(t,E)

    save(string(results_path, dict_name), 
    Dict("conservation_analysis" => analysis,
        "conservation_results" => results))

    return ConservationAnalysisResults(t,E)
end

function analyze(analysis::ConservationAnalysis,    
    dynamical_model::DynamicalAnalysisResults,
    time_steps::Vector{Int}, start::Int=1)

    @unpack results_path, N_eq, dict_name = analysis
    N_t = length(time_steps)
    t = Vector{Float64}(undef,N_t)
    E = Matrix{Float64}(undef,N_t, N_eq)
    E_modeled = Matrix{Float64}(undef,N_t-start+1, N_eq)

    u0, _ = load_solution(results_path, time_steps[1])
    for i in 1:N_t
        u, t[i] = load_solution(results_path, time_steps[i])
        E[i,:] = evaluate_conservation(analysis, u)
    end

    (N_p,N_eq,N_el) = size(u0)

    for i in start:N_t
        u = reshape(real.(evolve_forward(
            dynamical_model, t[i]-t[start]))[1:N_p*N_eq*N_el],
            (N_p,N_eq,N_el))
        E_modeled[i-start+1,:] = evaluate_conservation(analysis, u)
    end

    results = ConservationAnalysisResults(t,E)
    modeled_results = ConservationAnalysisResults(t[start:end], E_modeled)

    save(string(results_path, dict_name), 
    Dict("conservation_analysis" => analysis,
        "conservation_results" => results,
        "modeled_conservation_results" => modeled_results))  

    return results, modeled_results
end
    
function plot_evolution(analysis::ConservationAnalysis, 
    results::ConservationAnalysisResults, title::String; legend::Bool=false,
    ylabel::String="Energy", e::Int=1)
    p = plot(results.t, results.E[:,e], 
        legend=legend, xlabel="\$t\$", ylabel=ylabel)
    savefig(p, string(analysis.analysis_path, title))
    return p
end

function plot_evolution(analysis::ConservationAnalysis, 
    results::Vector{ConservationAnalysisResults}, title::String; 
    labels::Vector{String}=["Actual", "Predicted"],
    ylabel::String="Energy", e::Int=1)
    p = plot()
    N = length(results)
    for i in 1:N
        plot!(p, results[i].t, results[i].E[:,e], xlabel="\$t\$",   
        ylabel=ylabel, labels=labels[i])
    end
    savefig(p, string(analysis.analysis_path, title))
    return p
end