abstract type AbstractDynamicalAnalysis <: AbstractAnalysis end

struct DMDAnalysis <: AbstractDynamicalAnalysis
    path::String
    time_steps::Vector{Int64}
    N_t::Int
    N_p::Int
    N_eq::Int
    N_el::Int
    dt::Float64
    rank::Int
end

function DMDAnalysis(results_path::String;
     num_snapshots::Int=0, rank::Int=5)
    
    full_time_steps = load_time_steps(results_path)
    
    if num_snapshots > 1
        time_steps = full_time_steps[1:num_snapshots]
    else
        time_steps=full_time_steps
    end

    N_t = length(time_steps)
    first, t0 = load_solution(results_path, time_steps[1])
    _ , tf = load_solution(results_path, last(time_steps))
    dt = (tf - t0)/(N_t - 1)
    N_p = size(first,1)
    N_eq = size(first,2)
    N_el = size(first,3)

    path = new_path(string(results_path, "dmd/"))

    return DMDAnalysis(path, time_steps,
         N_t, N_p, N_eq, N_el, dt, rank)
end

function analyze(dmd::DMDAnalysis, X::Matrix{Float64}, 
    Y::Matrix{Float64})

    # SVD (i.e. POD) of initial states
    U_full, S_full, V_full = svd(X)
    
    U = U_full[:,1:dmd.rank]
    S = S_full[1:dmd.rank]
    V = V_full[:,1:dmd.rank]

    # eigendecomposition of reduced DMD matrix (projected onto singular vectors)
    F = eigen((U') * Y * V * inv(Diagonal(S)))
    # eigenvectors and eigenvalues of A = Y*pinv(X)
    ϕ = Y*V*inv(Diagonal(S))*F.vectors
    σ = F.values

    # continuous-time eigenvalues such that σ = exp(λdt)
    λ = log.(σ)/dmd.dt
    
    return σ, λ, ϕ

end

function save_analysis(dmd_analysis::DMDAnalysis, 
    σ::Vector{ComplexF64}, 
    λ::Vector{ComplexF64}, 
    ϕ::Matrix{ComplexF64})

    save(string(dmd_analysis.path, "dmd.jld2"), 
        Dict("dmd_analysis" => dmd_analysis,
            "discrete_eigvals" => σ,
            "continuous_eigvals" => λ,
            "modes" => ϕ))
end

function plot_spectrum(analysis::AbstractDynamicalAnalysis, 
    eigs::Vector{ComplexF64}, symbol::String="\\lambda")

    p = scatter(eigs, xlabel=latexstring(string("\\mathrm{Re}(", symbol, ")")), 
        ylabel=latexstring(string("\\mathrm{Im}(", symbol, ")")), legend=false, 
        series_annotations = text.(1:length(eigs), :bottom))

    savefig(p, string(analysis.path, "spectrum.pdf"))

    return p
end

function plot_modes(analysis::AbstractDynamicalAnalysis, 
    plotter::Plotter{1}, ϕ::Matrix{Float64}, e::Int)

    @unpack N_p, N_eq, N_el = analysis
    @unpack x_plot, V_plot, directory_name = plotter

    n_modes = size(ϕ,2)
    p = plot()
    for j in 1:n_modes
        sol = reshape(ϕ[:,j],(N_p, N_eq, N_el))
        linelabel = string("\\mathrm{Mode} \\, \\,", j)
        u = convert(Matrix, V_plot * sol[:,e,:])
        plot!(p,vec(vcat(x_plot[1],fill(NaN,1,N_el))), 
            vec(vcat(u,fill(NaN,1,N_el))), 
            label=latexstring(linelabel), xlabel=latexstring("x"))
    end

    savefig(p, string(directory_name, "modes.pdf")) 
    return p
end