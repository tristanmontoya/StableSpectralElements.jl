abstract type AbstractDynamicalAnalysis{d} <: AbstractAnalysis end

struct DynamicalAnalysisResults 
    σ::Vector{ComplexF64}
    λ::Vector{ComplexF64}
    ϕ::Matrix{ComplexF64}
    conjugate_pairs::Vector{Int}
    c::Union{Matrix{ComplexF64},Nothing}
    projection::Union{Matrix{ComplexF64},Nothing}
    E::Union{Matrix{Float64},Nothing}
end

struct LinearAnalysis{d} <: AbstractDynamicalAnalysis{d}
    results_path::String
    analysis_path::String
    r::Int
    tol::Float64
    N_p::Int
    N_eq::Int
    N_el::Int
    M::AbstractMatrix
    plotter::Plotter{d}
    L::LinearMap
    use_data::Bool
end

struct DMDAnalysis{d} <: AbstractDynamicalAnalysis{d}
    results_path::String
    analysis_path::String
    r::Int
    tol::Float64
    N_p::Int
    N_eq::Int
    N_el::Int
    M::AbstractMatrix
    plotter::Plotter{d}
end

function LinearAnalysis(results_path::String,
    conservation_law::ConservationLaw, 
    spatial_discretization::SpatialDiscretization, 
    L::Union{LinearMap{Float64},AbstractMatrix{Float64}};
    r=4, tol=1.0e-12, 
    name="linear_analysis", 
    use_data=true)

    analysis_path = new_path(string(results_path, name, "/"))
    N_p, N_eq, N_el = get_dof(spatial_discretization, conservation_law)

    # define mass matrix for the state space as a Hilbert space 
    M = blockdiag((kron(Diagonal(ones(N_eq)),
        sparse(spatial_discretization.M[k])) for k in 1:N_el)...)
            
    return LinearAnalysis(results_path, analysis_path, 
        r, tol, N_p, N_eq, N_el, M, Plotter(spatial_discretization, analysis_path), L, use_data)
end

function analyze(analysis::LinearAnalysis)

    @unpack M, L, r, tol, use_data, results_path = analysis
    values, ϕ_unscaled = eigs(L, nev=r, which=:SI)
    ϕ = similar(ϕ_unscaled)

    # normalize eigenvectors
    for i in 1:r
        ϕ[:,i] = ϕ_unscaled[:,i] / sqrt(ϕ_unscaled[:,i]'*M*ϕ_unscaled[:,i])
    end

    if use_data
        #load snapshot data
        X, t_s = load_snapshots(results_path, load_time_steps(results_path))

        # project data onto eigenvectors to determine coeffients
        c = (ϕ'*M*ϕ) \ ϕ'*M*X

        # calculate energy in each mode
        E = real([dot(c[i,j]*ϕ[:,i], M * (c[i,j]*ϕ[:,i]))
            for i in 1:r, j in 1:size(c,2)])
        
        # sort modes by decreasing energy in initial data
        inds_no_cutoff = sortperm(-E[:,1])
        inds = inds_no_cutoff[E[inds_no_cutoff,1] .> tol]
        λ = values[inds]

        # find conjugate pair indices
        conjugate_pairs = find_conjugate_pairs(λ)

        # discrete-time eigenvalues
        σ = exp.(λ*t_s)

        return DynamicalAnalysisResults(σ, 
            λ, ϕ[:,inds],  conjugate_pairs, 
            c[inds,:], ϕ*c, E[inds,:]) 
    else 
        return DynamicalAnalysisResults(exp.(dt*values), values, ϕ, nothing,
            nothing, find_conjugate_pairs(values), nothing)
    end
end

function DMDAnalysis(results_path::String, 
    conservation_law::ConservationLaw,spatial_discretization::SpatialDiscretization; 
    r=4, tol=1.0e-12, name="dmd_analysis")
    
    # create path and get discretization information
    analysis_path = new_path(string(results_path, name, "/"))

    N_p, N_eq, N_el = get_dof(spatial_discretization, conservation_law)

    # define mass matrix for the state space as a Hilbert space 
    M = blockdiag((kron(Diagonal(ones(N_eq)),
        sparse(spatial_discretization.M[k])) for k in 1:N_el)...)

    return DMDAnalysis(results_path, analysis_path, r, tol,
        N_p, N_eq, N_el, M, Plotter(spatial_discretization, analysis_path))
end

function analyze(analysis::DMDAnalysis)

    @unpack r, tol, M, results_path = analysis

    X, t_s = load_snapshots(results_path, load_time_steps(results_path))
    
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
        ϕ_unscaled = X[:,2:end]*V*inv(Diagonal(S))*F.vectors
    else
        # compute full eigendecomposition w/ Moore-Penrose pseudo-inverse
        F = eigen(X[2:end]*pinv(X[1:end-1]))
        ϕ_unscaled = F.vectors
    end

    # normalize eigenvectors (although not orthogonal - this isn't POD)
    ϕ =ϕ_unscaled/Diagonal([ϕ_unscaled[:,i]'*M*ϕ_unscaled[:,i] for i in 1:r])
    
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
    λ = log.(σ)/t_s

    return DynamicalAnalysisResults(σ, λ,
        ϕ[:,inds], conjugate_pairs, c[inds,:],
        ϕ*c, E[inds,:])
end

function save_analysis(analysis::AbstractDynamicalAnalysis,
    results::DynamicalAnalysisResults)
    save(string(analysis.analysis_path, "analysis.jld2"), 
        Dict("analysis" => analysis,
        "results" => results))
end

function plot_analysis(analysis::AbstractDynamicalAnalysis,
    results::DynamicalAnalysisResults; e=1, i=1, scale=true, title="spectrum.pdf")
    l = @layout [a{0.5w} b; c]
    if scale
        coeffs=results.c[:,i]
    else
        coeffs=nothing
    end

    p = plot(plot_spectrum(analysis,results.λ, 
        label="\\tilde{\\lambda}_j", unit_circle=false, 
        xlims=(minimum(real.(results.λ))-15,maximum(real.(results.λ))+15),
        ylims=(minimum(imag.(results.λ))-15,maximum(imag.(results.λ))+15),
        title="continuous_time.pdf", xscale=-0.03, yscale=0.03), 
        plot_spectrum(analysis,results.σ, 
        label="\\exp(\\tilde{\\lambda}_j t_  {\\mathrm{s}})",
        unit_circle=true, xlims=(-1.25,1.25), ylims=(-1.25,1.25),
        title="discrete_time.pdf"), 
        plot_modes(analysis,results.ϕ::Matrix{ComplexF64}; e=e, 
        coeffs=coeffs, conjugate_pairs=results.conjugate_pairs), layout=l, framestyle=:box)
    
    savefig(p, string(analysis.analysis_path, title))
    return p
end

function plot_spectrum(analysis::AbstractDynamicalAnalysis, 
    eigs::Vector{ComplexF64}; label="\\exp(\\tilde{\\lambda}_j t_{\\mathrm{s}})", unit_circle=true, xlims=(-1.25,1.25), ylims=(-1.25,1.25),
    xscale=0.02, yscale=0.07, title="spectrum.pdf")

    if unit_circle
        t=collect(LinRange(0.0, 2.0*π,100))
        p = plot(cos.(t), sin.(t), aspect_ratio=:equal, linecolor="black")
    else
        p = plot()
    end

    #=
    if !isnothing(exact)
        plot!(p, [0.0, real(exp(im*exact*analysis.dt))],[0.0, imag(exp(im*exact*analysis.dt))], linecolor="black", linestyle=:dash)
        plot!(p, [0.0, real(exp(im*exact*analysis.dt))],[0.0, -imag(exp(im*exact*analysis.dt))], linecolor="black", linestyle=:dash)
    end
    =#

    plot!(p, eigs, xlabel=latexstring(string("\\mathrm{Re}\\,(", label, ")")), 
        ylabel=latexstring(string("\\mathrm{Im}\\,(", label, ")")), 
        xlims=xlims, ylims=ylims,legend=false,
        seriestype=:scatter)

    annotate!(real(eigs) .+ xscale*(xlims[2]-xlims[1]), 
        imag(eigs) + sign.(imag(eigs) .+ 1.0e-15)*yscale*(ylims[2]-ylims[1]), text.(1:length(eigs), :right, 8))

    savefig(p, string(analysis.analysis_path, title))

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
            label=latexstring(linelabel),linewidth=2, ylabel=latexstring("\\tilde{\\varphi}_j(x)"), legendfontsize=6)
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

    savefig(p, string(analysis.analysis_path, "modes.pdf")) 
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