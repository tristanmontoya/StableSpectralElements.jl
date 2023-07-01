abstract type ConservationAnalysis <: AbstractAnalysis end
abstract type AbstractConservationAnalysisResults <: AbstractAnalysisResults end

"""
Evaluate change in ∫udx  
"""
struct PrimaryConservationAnalysis <: ConservationAnalysis
    WJ::Vector{Diagonal}
    N_c::Int
    N_e::Int
    V::LinearMap
    results_path::String
    dict_name::String
end

"""
Evaluate change in  ∫½u²dx  
"""
struct EnergyConservationAnalysis <: ConservationAnalysis
    mass_solver::AbstractMassMatrixSolver
    N_c::Int
    N_e::Int
    V::LinearMap
    results_path::String
    dict_name::String
end

struct ConservationAnalysisResults <:AbstractConservationAnalysisResults
    t::Vector{Float64}
    E::Matrix{Float64}
end

struct ConservationAnalysisResultsWithDerivative <:AbstractConservationAnalysisResults
    t::Vector{Float64}
    E::Matrix{Float64}
    dEdt::Matrix{Float64}
end

function PrimaryConservationAnalysis(results_path::String,
    conservation_law::AbstractConservationLaw, 
    spatial_discretization::SpatialDiscretization{d}) where {d}

    _, N_c, N_e = get_dof(spatial_discretization, conservation_law)
  
    (; W, V) =  spatial_discretization.reference_approximation
    (; geometric_factors, mesh, N_e) = spatial_discretization
    
    WJ = [Diagonal(W .* geometric_factors.J_q[:,k]) for k in 1:N_e]

    return PrimaryConservationAnalysis(
        WJ, N_c, N_e, V, results_path, "conservation.jld2")
end

function EnergyConservationAnalysis(results_path::String,
    conservation_law::AbstractConservationLaw,
    spatial_discretization::SpatialDiscretization{d},
    mass_solver=WeightAdjustedSolver(spatial_discretization)) where {d}

    _, N_c, N_e = get_dof(spatial_discretization, conservation_law)

    (; W, V) =  spatial_discretization.reference_approximation
    (; mesh, N_e) = spatial_discretization
    (; J_q) = spatial_discretization.geometric_factors

    return EnergyConservationAnalysis(
        mass_solver, N_c, N_e, V ,results_path, "energy.jld2")
end

function PrimaryConservationAnalysis(       
    conservation_law::AbstractConservationLaw, 
    spatial_discretization::SpatialDiscretization)
    return PrimaryConservationAnalysis("./", conservation_law, 
        spatial_discretization)
end

function EnergyConservationAnalysis(       
    conservation_law::AbstractConservationLaw, 
    spatial_discretization::SpatialDiscretization,
    mass_solver=WeightAdjustedSolver(spatial_discretization))
    return EnergyConservationAnalysis("./", conservation_law, 
        spatial_discretization, mass_solver)
end

function evaluate_conservation(
    analysis::PrimaryConservationAnalysis, 
    u::Array{Float64,3})
    (; WJ, N_c, N_e, V) = analysis 

    return [sum(sum(WJ[k]*V*u[:,e,k]) 
        for k in 1:N_e) for e in 1:N_c]
end

function evaluate_conservation(
    analysis::EnergyConservationAnalysis, 
    u::Array{Float64,3})
    (; mass_solver, N_c, N_e, V) = analysis 
    E = zeros(N_c)
    N_p = size(V,2)
    M = Matrix{Float64}(undef, N_p, N_p)
    for k in 1:N_e
        M .= mass_matrix(mass_solver, k)
        for e in 1:N_c
            E[e] += 0.5*u[:,e,k]'*M*u[:,e,k] 
        end
    end
    return E
end

function evaluate_conservation_residual(
    analysis::PrimaryConservationAnalysis, 
    ::Array{Float64,3},
    dudt::Array{Float64,3})
    (; WJ, N_c, N_e, V) = analysis 

    return [sum(ones(size(WJ[k],1))'*WJ[k]*V*dudt[:,e,k]
        for k in 1:N_e) for e in 1:N_c]
end

function evaluate_conservation_residual(
    analysis::EnergyConservationAnalysis, 
    u::Array{Float64,3},
    dudt::Array{Float64,3})
    (; mass_solver, N_c, N_e, V) = analysis 

    dEdt = zeros(N_c)
    for k in 1:N_e
        M = mass_matrix(mass_solver, k)
        for e in 1:N_c
            dEdt[e] += u[:,e,k]'*M*dudt[:,e,k] 
        end
    end
    return dEdt
end

function analyze(analysis::ConservationAnalysis, 
    initial_time_step::Union{Int,String}=0, 
    final_time_step::Union{Int,String}="final")
    
    (; results_path, dict_name) = analysis

    u_0, _ = load_solution(results_path, initial_time_step)
    u_f, _ = load_solution(results_path, final_time_step)

    initial = evaluate_conservation(analysis,u_0)
    final = evaluate_conservation(analysis,u_f)
    difference = final .- initial

    return initial, final, difference
end

function analyze(analysis::ConservationAnalysis,
    time_steps::Vector{Int})

    (; results_path, N_c, dict_name) = analysis
    N_t = length(time_steps)
    t = Vector{Float64}(undef,N_t)
    E = Matrix{Float64}(undef,N_t, N_c)
    dEdt = Matrix{Float64}(undef,N_t, N_c)

    for i in 1:N_t
        u, dudt, t[i] = load_solution(results_path, time_steps[i], load_du=true)
        E[i,:] = evaluate_conservation(analysis, u)
        dEdt[i,:] = evaluate_conservation_residual(analysis, u, dudt)
    end

    results = ConservationAnalysisResults(t,E)

    save(string(results_path, dict_name), 
    Dict("conservation_analysis" => analysis,
        "conservation_results" => results))

    return ConservationAnalysisResultsWithDerivative(t,E, dEdt)
end

function analyze(analysis::ConservationAnalysis,    
    model::DynamicalAnalysisResults,
    time_steps::Vector{Int}, Δt::Float64, start::Int=1,
    resolution=100;  n=1, window_size=nothing, new_projection=false)

    (; results_path, N_c, dict_name) = analysis
    N_t = length(time_steps)
    t = Vector{Float64}(undef,N_t)
    E = Matrix{Float64}(undef,N_t, N_c)
    
    t_modeled = Vector{Float64}(undef,resolution+1)
    E_modeled = Matrix{Float64}(undef,resolution+1, N_c)

    u0, t0 = load_solution(results_path, time_steps[start])
    for i in 1:N_t
        u, t[i] = load_solution(results_path, time_steps[i])
        E[i,:] = evaluate_conservation(analysis, u)
    end

    (N_p,N_c,N_e) = size(u0)
    N = N_p*N_c*N_e

    dt = Δt/resolution
    if new_projection
        c = pinv(model.Z[1:N,:]) * vec(u0)
    elseif !isnothing(window_size)
        c = model.c[:,1]
        t0 = t[max(start-window_size+1,1)]
    else
        c = model.c[:, (start-1)*n+1]
    end

    for i in 0:resolution
        u = reshape(real.(forecast(model, dt*i, c)[1:N]),(N_p,N_c,N_e))
        t_modeled[i+1] = t0+dt*i
        E_modeled[i+1,:] = evaluate_conservation(analysis, u)
    end

    results = ConservationAnalysisResults(t,E)
    modeled_results = ConservationAnalysisResults(t_modeled, E_modeled)

    save(string(results_path, dict_name), 
    Dict("conservation_analysis" => analysis,
        "conservation_results" => results,
        "modeled_conservation_results" => modeled_results))  

    return results, modeled_results
end

@recipe function plot(results::ConservationAnalysisResults, e::Int=1)

    xlabel --> latexstring("t")
    ylabel --> LaTeXString("Energy")
    legend --> false

    results.t, results.E[:,e]
end

@recipe function plot(results::ConservationAnalysisResultsWithDerivative, e::Int=1)

    xlabel --> latexstring("t")
    labels = [LaTeXString("Net change"), LaTeXString("Time derivative")]
    fontfamily --> "Computer Modern"

    @series begin
        linestyle --> :solid
        label --> labels[1]
        results.t, results.E[:,e] .- first(results.E[:,e])
    end
    @series begin
        linestyle --> :dot
        label --> labels[2]
        results.t, results.dEdt[:,e]
    end
   
end

function plot_evolution(analysis::ConservationAnalysis, 
    results::Vector{<:AbstractConservationAnalysisResults}, title::String; 
    labels::Vector{String}=["Actual", "Predicted"],
    ylabel::String="Energy", e::Int=1, t=nothing, xlims=nothing, ylims=nothing)

    p = plot(results[1].t, results[1].E[:,e], xlabel="\$t\$",
        ylabel=ylabel, labels=labels[1], xlims=xlims, ylims=ylims, 
        linewidth=2.0)
    N = length(results)
    
    for i in 2:N
        plot!(p, results[i].t, results[i].E[:,e], labels=labels[i], linestyle=:dash, linewidth=3.0, legend=:topright)
    end
    if !isnothing(t)
       vline!(p,[t], labels=nothing)
    end

    savefig(p, string(analysis.results_path, title))
    return p
end