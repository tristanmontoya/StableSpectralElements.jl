abstract type ConservationAnalysis <: AbstractAnalysis end
abstract type AbstractConservationAnalysisResults <: AbstractAnalysisResults end

# Evaluate change in ∫udx  
struct PrimaryConservationAnalysis{V_type} <: ConservationAnalysis
    WJ::Vector{Diagonal{Float64, Vector{Float64}}}
    N_c::Int
    N_e::Int
    V::V_type
    results_path::String
    dict_name::String
end

# Evaluate change in  ∫½u²dx  
struct EnergyConservationAnalysis{V_type, MassSolver} <: ConservationAnalysis
    mass_solver::MassSolver
    N_c::Int
    N_e::Int
    V::V_type
    results_path::String
    dict_name::String
end

struct EntropyConservationAnalysis{V_type, MassSolver, ConservationLaw} <:
       ConservationAnalysis
    mass_solver::MassSolver
    WJ::Vector{Diagonal{Float64, Vector{Float64}}}
    conservation_law::ConservationLaw
    N_c::Int
    N_e::Int
    V::V_type
    results_path::String
    dict_name::String
end

struct ConservationAnalysisResults <: AbstractConservationAnalysisResults
    t::Vector{Float64}
    E::Matrix{Float64}
end

struct ConservationAnalysisResultsWithDerivative <: AbstractConservationAnalysisResults
    t::Vector{Float64}
    E::Matrix{Float64}
    dEdt::Matrix{Float64}
end

function PrimaryConservationAnalysis(results_path::String,
        conservation_law::AbstractConservationLaw,
        spatial_discretization::SpatialDiscretization{d}) where {
        d,
}
    _, N_c, N_e = get_dof(spatial_discretization, conservation_law)

    (; W, V) = spatial_discretization.reference_approximation
    (; geometric_factors, N_e) = spatial_discretization

    WJ = [Diagonal(W .* geometric_factors.J_q[:, k]) for k in 1:N_e]

    return PrimaryConservationAnalysis(WJ, N_c, N_e, V, results_path, "conservation.jld2")
end

function EnergyConservationAnalysis(results_path::String,
        conservation_law::AbstractConservationLaw,
        spatial_discretization::SpatialDiscretization{d},
        mass_solver = WeightAdjustedSolver(spatial_discretization)) where {
        d,
}
    _, N_c, N_e = get_dof(spatial_discretization, conservation_law)

    (; V) = spatial_discretization.reference_approximation
    (; N_e) = spatial_discretization

    return EnergyConservationAnalysis(mass_solver, N_c, N_e, V, results_path, "energy.jld2")
end

function PrimaryConservationAnalysis(conservation_law::AbstractConservationLaw,
        spatial_discretization::SpatialDiscretization)
    return PrimaryConservationAnalysis("./", conservation_law, spatial_discretization)
end

function EnergyConservationAnalysis(conservation_law::AbstractConservationLaw,
        spatial_discretization::SpatialDiscretization,
        mass_solver = WeightAdjustedSolver(spatial_discretization))
    return EnergyConservationAnalysis("./",
        conservation_law,
        spatial_discretization,
        mass_solver)
end

function EntropyConservationAnalysis(results_path::String,
        conservation_law::AbstractConservationLaw,
        spatial_discretization::SpatialDiscretization{d},
        mass_solver = WeightAdjustedSolver(spatial_discretization)) where {
        d,
}
    _, N_c, N_e = get_dof(spatial_discretization, conservation_law)

    (; W, V) = spatial_discretization.reference_approximation
    (; geometric_factors, N_e) = spatial_discretization

    WJ = [Diagonal(W .* geometric_factors.J_q[:, k]) for k in 1:N_e]

    return EntropyConservationAnalysis(mass_solver,
        WJ,
        conservation_law,
        N_c,
        N_e,
        V,
        results_path,
        "entropy.jld2")
end

function evaluate_conservation(analysis::PrimaryConservationAnalysis, u::Array{Float64, 3})
    (; WJ, N_c, N_e, V) = analysis

    return [sum(sum(WJ[k] * V * u[:, e, k]) for k in 1:N_e) for e in 1:N_c]
end

function evaluate_conservation(analysis::EntropyConservationAnalysis, u::Array{Float64, 3})
    (; WJ, conservation_law, N_e, V) = analysis
    S = 0.0
    @views for k in 1:N_e
        u_q = Matrix(V * u[:, :, k])
        for i in axes(u_q, 1)
            S += WJ[k][i, i] * entropy(conservation_law, u_q[i, :])
        end
    end
    return [S]
end

function evaluate_conservation(analysis::EnergyConservationAnalysis, u::Array{Float64, 3})
    (; mass_solver, N_c, N_e, V) = analysis
    E = zeros(N_c)
    N_p = size(V, 2)
    M = Matrix{Float64}(undef, N_p, N_p)
    for k in 1:N_e
        M .= mass_matrix(mass_solver, k)
        for e in 1:N_c
            E[e] += 0.5 * u[:, e, k]' * M * u[:, e, k]
        end
    end
    return E
end

function evaluate_conservation_residual(analysis::PrimaryConservationAnalysis,
        ::Array{Float64, 3},
        dudt::Array{Float64, 3})
    (; WJ, N_c, N_e, V) = analysis

    return [sum(ones(size(WJ[k], 1))' * WJ[k] * V * dudt[:, e, k] for k in 1:N_e)
            for e in 1:N_c]
end

function evaluate_conservation_residual(analysis::EnergyConservationAnalysis,
        u::Array{Float64, 3},
        dudt::Array{Float64, 3})
    (; mass_solver, N_c, N_e) = analysis

    dEdt = zeros(N_c)
    for k in 1:N_e
        M = mass_matrix(mass_solver, k)
        for e in 1:N_c
            dEdt[e] += u[:, e, k]' * M * dudt[:, e, k]
        end
    end
    return dEdt
end

@inline @views function evaluate_conservation_residual(
        analysis::EntropyConservationAnalysis,
        u::Array{Float64, 3},
        dudt::Array{Float64, 3})
    (; mass_solver, conservation_law, N_c, N_e, V, WJ) = analysis

    u_q = Matrix{Float64}(undef, size(V, 1), N_c)
    w_q = Matrix{Float64}(undef, size(V, 1), N_c)
    dEdt = 0.0
    for k in 1:N_e
        M = mass_matrix(mass_solver, k)
        mul!(u_q, V, u[:, :, k])
        for i in axes(u_q, 1)
            conservative_to_entropy!(w_q[i, :], conservation_law, u_q[i, :])
        end
        P = mass_matrix_inverse(mass_solver, k) * V' * WJ[k]
        for e in 1:N_c
            dEdt += (P * w_q[:, e])' * M * dudt[:, e, k]
        end
    end
    return [dEdt]
end

function analyze(analysis::EntropyConservationAnalysis,
        time_steps::Vector{Int};
        normalize = false)
    (; results_path, dict_name) = analysis
    N_t = length(time_steps)
    t = Vector{Float64}(undef, N_t)
    E = Matrix{Float64}(undef, N_t, 1)
    dEdt = Matrix{Float64}(undef, N_t, 1)

    if normalize
        u_0, _ = load_solution(results_path, time_steps[1])
        factor = abs.(evaluate_conservation(analysis, u_0))
    else
        factor = 1.0
    end

    for i in 1:N_t
        u, dudt, t[i] = load_solution(results_path, time_steps[i], load_du = true)
        E[i, :] .= evaluate_conservation(analysis, u) ./ factor
        dEdt[i, :] .= evaluate_conservation_residual(analysis, u, dudt) ./ factor
    end

    results = ConservationAnalysisResults(t, E)

    save(string(results_path, dict_name),
        Dict("conservation_analysis" => analysis, "conservation_results" => results))

    return ConservationAnalysisResultsWithDerivative(t, E, dEdt)
end

function analyze(analysis::ConservationAnalysis,
        initial_time_step::Union{Int, String} = 0,
        final_time_step::Union{Int, String} = "final")
    (; results_path) = analysis

    u_0, _ = load_solution(results_path, initial_time_step)
    u_f, _ = load_solution(results_path, final_time_step)

    initial = evaluate_conservation(analysis, u_0)
    final = evaluate_conservation(analysis, u_f)
    difference = final .- initial

    return initial, final, difference
end

function analyze(analysis::ConservationAnalysis, time_steps::Vector{Int}; normalize = false)
    (; results_path, N_c, dict_name) = analysis
    N_t = length(time_steps)
    t = Vector{Float64}(undef, N_t)
    E = Matrix{Float64}(undef, N_t, N_c)
    dEdt = Matrix{Float64}(undef, N_t, N_c)

    if normalize
        u_0, _ = load_solution(results_path, time_steps[1])
        factor = abs.(evaluate_conservation(analysis, u_0))
    else
        factor = 1.0
    end

    for i in 1:N_t
        u, dudt, t[i] = load_solution(results_path, time_steps[i], load_du = true)
        E[i, :] = evaluate_conservation(analysis, u) ./ factor
        dEdt[i, :] = evaluate_conservation_residual(analysis, u, dudt) ./ factor
    end

    results = ConservationAnalysisResults(t, E)

    save(string(results_path, dict_name),
        Dict("conservation_analysis" => analysis, "conservation_results" => results))

    return ConservationAnalysisResultsWithDerivative(t, E, dEdt)
end

@recipe function plot(results::ConservationAnalysisResults, e::Int = 1)
    xlabel --> latexstring("t")
    ylabel --> LaTeXString("Energy")
    legend --> false

    results.t, results.E[:, e]
end

@recipe function plot(results::ConservationAnalysisResultsWithDerivative,
        e::Int = 1;
        net_change = true,
        derivative = true)
    xlabel --> latexstring("t")
    labels = [LaTeXString("Net change"), LaTeXString("Time derivative")]
    fontfamily --> "Computer Modern"

    if net_change
        @series begin
            linestyle --> :solid
            label --> labels[1]
            results.t, results.E[:, e] .- first(results.E[:, e])
        end
    end
    if derivative
        @series begin
            linestyle --> :dot
            label --> labels[2]
            results.t, results.dEdt[:, e]
        end
    end
end
