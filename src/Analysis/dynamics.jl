abstract type AbstractDynamicalAnalysis <: AbstractAnalysis end

struct DynamicalAnalysisResults 
    σ::Vector{ComplexF64}
    λ::Vector{ComplexF64}
    ϕ::Matrix{ComplexF64}
    conjugate_pairs::Vector{Int}
    c::Union{Matrix{ComplexF64},Nothing}
    projection::Union{Matrix{ComplexF64},Nothing}
    E::Union{Matrix{Float64},Nothing}
end

struct LinearAnalysis <: AbstractDynamicalAnalysis
    path::String
    r::Int
    tol::Float64
    N_p::Int
    N_eq::Int
    N_el::Int
    dt::Float64
    M::AbstractMatrix
    plotter::Plotter
    L::LinearMap
    X::Union{Matrix{Float64},Nothing}
end

struct DMDAnalysis <: AbstractDynamicalAnalysis
    path::String
    r::Int
    tol::Float64
    time_steps::Vector{Int64}
    N_t::Int
    N_p::Int
    N_eq::Int
    N_el::Int
    dt::Float64
    M::AbstractMatrix
    plotter::Plotter
    X::Matrix{Float64}
end

function LinearAnalysis(results_path::String; dt=1.0, r=4, 
    tol=1.0e-12, X=nothing, name="linear_analysis")

    # create path and get discretization information
    path = new_path(string(results_path, name, "/"))
    (conservation_law, spatial_discretization, 
        _, form, _, strategy) = load_project(results_path)
    L = LinearResidual(Solver(
        conservation_law,spatial_discretization,form,strategy))

    N_p, N_eq, N_el = get_dof(spatial_discretization, conservation_law)

    # create a mass matrix for the state space as a Hilbert space 
    M = blockdiag((
        kron(Diagonal(ones(N_eq)),sparse(spatial_discretization.M[k]))
        for k in 1:N_el)...)
            
    return LinearAnalysis(path, r, tol, N_p, N_eq, N_el, dt, M, 
        Plotter(spatial_discretization, path), L, X)
end

function analyze(analysis::LinearAnalysis)

    @unpack M, L, dt, r, tol, X = analysis
    values, ϕ = eigs(L, nev=r, which=:LM)

    if isnothing(X)
        DynamicalAnalysisResults(exp.(dt*values), 
        values, ϕ, nothing, nothing, 
        find_conjugate_pairs(values), nothing)
    else
        # project onto eigenvectors
        c = (ϕ'*M*ϕ) \ ϕ'*M*X

        # calculate energy
        E = real([dot(c[i,j]*ϕ[:,i], M * (c[i,j]*ϕ[:,i]))
            for i in 1:r, j in 1:size(c,2)])
        
        # sort modes by decreasing energy in initial data
        inds_no_cutoff = sortperm(-E[:,1])
        inds = inds_no_cutoff[E[inds_no_cutoff,1] .> tol]
        λ = values[inds]

        # find conjugate pair indices
        conjugate_pairs = find_conjugate_pairs(λ)

        # discrete-time eigenvalues
        σ = exp.(dt*λ)

        return DynamicalAnalysisResults(σ, 
            λ, ϕ[:,inds],  conjugate_pairs, 
            c[inds,:], ϕ*c, E[inds,:])
    end
end

function DMDAnalysis(results_path::String; 
    num_snapshots=0, r=4, tol=1.0e-12, name="dmd")
    
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
    X = load_snapshots(results_path, time_steps)

    # create path and get discretization information
    path = new_path(string(results_path, name, "/"))
    conservation_law, spatial_discretization = load_project(results_path);

    # create a mass matrix for the state space as a Hilbert space 
    M = blockdiag((
        kron(Diagonal(ones(N_eq)),sparse(spatial_discretization.M[k]))
        for k in 1:N_el)...)

    return DMDAnalysis(path, r, tol, time_steps,
         N_t, N_p, N_eq, N_el, dt, M,
         Plotter(spatial_discretization, path), X)
end

function analyze(analysis::DMDAnalysis)

    @unpack r, tol, M, X, dt = analysis
    
    if r > 0
        # SVD (i.e. POD) of initial states
        U_full, S_full, V_full = svd(X[:,1:end-1])

        U = U_full[:,1:r]
        S = S_full[1:r]
        V = V_full[:,1:r]

        # eigendecomposition of reduced DMD matrix 
        # (projected onto singular vectors)
        F = eigen((U') * X[:,2:end] * V * inv(Diagonal(S)))

        # map eigenvectors back up into full space
        ϕ = X[:,2:end]*V*inv(Diagonal(S))*F.vectors
    else
        # compute full eigendecomposition w/ Moore-Penrose pseudo-inverse
        F = eigen(X[2:end]*pinv(X[1:end-1]))
        ϕ = F.vectors
    end
    
    # project onto eigenvectors with respect to the inner product of the scheme
    c = (ϕ'*M*ϕ) \ ϕ'*M*X

    # calculate energy
    E = real([dot(c[i,j]*ϕ[:,i], M * (c[i,j]*ϕ[:,i]))
        for i in 1:r, j in 1:size(X,2)])
    
    # sort modes by decreasing energy in initial data
    inds_no_cutoff = sortperm(-E[:,1])
    inds = inds_no_cutoff[E[inds_no_cutoff,1] .> tol]
    σ = F.values[inds]

    # find conjugate pair indices
    conjugate_pairs = find_conjugate_pairs(σ)

    # continuous-time eigenvalues such that σ = exp(λdt)
    λ = log.(σ)/dt

    return DynamicalAnalysisResults(σ, λ,
        ϕ[:,inds], conjugate_pairs, c[inds,:],
        ϕ*c, E[inds,:])
end

function save_analysis(analysis::AbstractDynamicalAnalysis,
    results::DynamicalAnalysisResults)
    save(string(analysis.path, "analysis.jld2"), 
        Dict("analysis" => analysis,
        "results" => results))
end

function plot_analysis(analysis::AbstractDynamicalAnalysis,
    results::DynamicalAnalysisResults; e=1, i=1)
    l = @layout [a{0.5w} b; c]
    return plot(plot_spectrum(analysis,results.λ, 
        label="\\tilde{\\lambda}_j", unit_circle=false, 
        xlims=(minimum(real.(results.λ))-15,maximum(real.(results.λ))+15),
        ylims=(minimum(imag.(results.λ))-15,maximum(imag.(results.λ))+15),
        title="continuous_time.pdf", xscale=-0.03, yscale=0.03), 
        plot_spectrum(analysis,results.σ, 
        label="\\exp(\\tilde{\\lambda}_j t_  {\\mathrm{s}})",
        unit_circle=true, xlims=(-1.25,1.25), ylims=(-1.25,1.25),
        title="discrete_time.pdf"), 
        plot_modes(analysis,results.ϕ::Matrix{ComplexF64}; e=e, 
        coeffs=results.c[:,i], conjugate_pairs=results.conjugate_pairs), layout=l, framestyle=:box)
end

function plot_spectrum(analysis::AbstractDynamicalAnalysis, 
    eigs::Vector{ComplexF64}; label="\\exp(\\tilde{\\lambda}_j t_{\\mathrm{s}})", unit_circle=true, exact=nothing, xlims=(-1.25,1.25), ylims=(-1.25,1.25),
    xscale=0.02, yscale=0.07, title="spectrum.pdf")

    if unit_circle
        t=collect(LinRange(0.0, 2.0*π,100))
        p = plot(cos.(t), sin.(t), aspect_ratio=:equal, linecolor="black")
    else
        p = plot()
    end

    if !isnothing(exact)
        plot!(p, [0.0, real(exp(im*exact*analysis.dt))],[0.0, imag(exp(im*exact*analysis.dt))], linecolor="black", linestyle=:dash)
        plot!(p, [0.0, real(exp(im*exact*analysis.dt))],[0.0, -imag(exp(im*exact*analysis.dt))], linecolor="black", linestyle=:dash)
    end

    plot!(p, eigs, xlabel=latexstring(string("\\mathrm{Re}\\,(", label, ")")), 
        ylabel=latexstring(string("\\mathrm{Im}\\,(", label, ")")), 
        xlims=xlims, ylims=ylims,legend=false,
        seriestype=:scatter)

    annotate!(real(eigs) .+ xscale*(xlims[2]-xlims[1]), 
        imag(eigs) + sign.(imag(eigs) .+ 1.0e-15)*yscale*(ylims[2]-ylims[1]), text.(1:length(eigs), :right, 8))

    savefig(p, string(analysis.path, title))

    return p
end

function plot_modes(analysis::AbstractDynamicalAnalysis, 
    ϕ::Matrix{ComplexF64}; e=1, 
    coeffs=nothing, projection=nothing,
    conjugate_pairs=nothing)

    @unpack N_p, N_eq, N_el, plotter = analysis
    @unpack x_plot, V_plot = plotter

    n_modes = size(ϕ,2)
    p = plot()

    if isnothing(coeffs)
        coeffs = ones(n_modes)
    end

    if isnothing(conjugate_pairs)
        conjugate_pairs = zeros(Int64,n_modes)
    end

    skip = fill(false, n_modes)
    for j in 1:n_modes

        if skip[j]
            continue
        end

        sol = reshape(ϕ[:,j],(N_p, N_eq, N_el))
        u = convert(Matrix, V_plot * real(coeffs[j]*sol[:,e,:]))

        if conjugate_pairs[j] == 0
            linelabel = string(j)
            scale_factor = 1.0
        else
            linelabel = string(j, ",", conjugate_pairs[j])
            scale_factor = 2.0
            skip[conjugate_pairs[j]] = true
        end

        plot!(p,vec(vcat(x_plot[1],fill(NaN,1,N_el))), 
            vec(vcat(scale_factor*u,fill(NaN,1,N_el))), 
            label=latexstring(linelabel),linewidth=2, ylabel=latexstring("\\tilde{\\varphi}_j(x)"))
    end

    if !isnothing(projection)
        sol = reshape(projection,(N_p, N_eq, N_el))
        linelabel = string("\\mathrm{Projection}")
        u = convert(Matrix, V_plot * real(sol[:,e,:]))
        plot!(p,vec(vcat(x_plot[1],fill(NaN,1,N_el))), 
            vec(vcat(u,fill(NaN,1,N_el))), 
            label=latexstring(linelabel), xlabel=latexstring("x"), 
            linewidth=2, linestyle=:dash, linecolor="black")
    end

    savefig(p, string(analysis.path, "modes.pdf")) 
    return p
end

function find_conjugate_pairs(σ::Vector{ComplexF64}; tol=1.0e-8)

    N = size(σ,1)
    conjugate_pairs = zeros(Int64,N)
    for i in 1:N
        if conjugate_pairs[i] == 0
            for j in (i+1):N
                if abs(σ[j] - conj(σ[i])) < tol
                    conjugate_pairs[i] = j
                    conjugate_pairs[j] = i
                    break
                end
            end
        end
    end
    return conjugate_pairs

end

find_conjugate_pairs(σ::Vector{Float64}; tol=1.0e-8) = nothing