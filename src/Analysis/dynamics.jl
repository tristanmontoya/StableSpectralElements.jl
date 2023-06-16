abstract type AbstractDynamicalAnalysis{d} <: AbstractAnalysis end

abstract type AbstractKoopmanAlgorithm end

abstract type AbstractSamplingStrategy end

struct DynamicalAnalysisResults <: AbstractAnalysisResults
    σ::Vector{ComplexF64}
    λ::Vector{ComplexF64}
    Z::Matrix{ComplexF64}
    conjugate_pairs::Union{Vector{Int},Nothing}
    c::Union{Matrix{ComplexF64},Nothing}
    projection::Union{Matrix{ComplexF64},Nothing}
    E::Union{Matrix{Float64},Nothing}
end

struct LinearAnalysis{d} <: AbstractDynamicalAnalysis{d}
    results_path::String
    r::Int
    tol::Float64
    N_p::Int
    N_c::Int
    N_e::Int
    M::AbstractMatrix
    plotter::Plotter{d}
    L::LinearMap
    use_data::Bool
end

struct KoopmanAnalysis{d} <: AbstractDynamicalAnalysis{d}
    results_path::String
    r::Int
    svd_tol::Float64
    proj_tol::Float64
    N_p::Int
    N_c::Int
    N_e::Int
    M::AbstractMatrix
    plotter::Plotter{d}
end

"""Tu et al. (2014) or Kutz et al. (2018)"""
struct StandardDMD <: AbstractKoopmanAlgorithm 
    basis::Vector{<:Function}
end

"""Williams et al. (2015aß)"""
struct ExtendedDMD <: AbstractKoopmanAlgorithm
    basis::Vector{<:Function}
end

"""Klus et al. (2020)"""
struct GeneratorDMD
    basis::Vector{<:Function}
    basis_derivatives::Vector{<:Function}
    f::Function
end

"""Williams et al. (2015b)"""
struct KernelDMD <: AbstractKoopmanAlgorithm 
    k::Function
    η::Float64
end

"""Colbrook and Townsend (2021)"""
struct KernelResDMD <: AbstractKoopmanAlgorithm
    k::Function
    ϵ::Float64
end

struct GaussianSampling <: AbstractSamplingStrategy
    σ::Float64  # width
    n::Int  # number of samples
end

function LinearAnalysis(results_path::String,
    conservation_law::AbstractConservationLaw, 
    spatial_discretization::SpatialDiscretization, 
    L::Union{LinearMap{Float64},AbstractMatrix{Float64}};
    r=4, tol=1.0e-12, name="linear_analysis", 
    use_data=true)

    N_p, N_c, N_e = get_dof(spatial_discretization, conservation_law)

    # define mass matrix for the state space as a Hilbert space 
    M = blockdiag((kron(Diagonal(ones(N_c)),
        sparse(spatial_discretization.M[k])) for k in 1:N_e)...)
            
    return LinearAnalysis(results_path, r, tol, N_p, N_c, N_e, M, 
        Plotter(spatial_discretization, results_path), L, use_data)
end

"""Koopman analysis"""
function KoopmanAnalysis(results_path::String, 
    conservation_law::AbstractConservationLaw,spatial_discretization::SpatialDiscretization; 
    r=4, svd_tol=1.0e-12, proj_tol=1.0e-12, 
    name="koopman_analysis")
    
    # get discretization information
    N_p, N_c, N_e = get_dof(spatial_discretization, conservation_law)

    # define mass matrix for the state space as a Hilbert space 
    M = blockdiag((kron(Diagonal(ones(N_c)),
        sparse(spatial_discretization.M[k])) for k in 1:N_e)...)

    return KoopmanAnalysis(results_path, 
        r, svd_tol, proj_tol, N_p, N_c, N_e, M, 
        Plotter(spatial_discretization, results_path))
end

"""Default constructors"""
StandardDMD() = StandardDMD([identity])
ExtendedDMD() = ExtendedDMD([identity])
KernelDMD() = KernelDMD((x,y) -> (1.0 + x'*y), 0.0)
KernelDMD(k::Function) = KernelDMD(k, 0.0)


"""Linear eigensolution analysis"""
function analyze(analysis::LinearAnalysis)

    (; M, L, r, tol, use_data, results_path) = analysis
    eigenvalues, eigenvectors = eigs(L, nev=r, which=:LM, maxiter=1000)

    # normalize eigenvectors
    Z = eigenvectors/Diagonal([eigenvectors[:,i]'*M*eigenvectors[:,i] 
        for i in 1:r])

    if use_data
        #load snapshot data
        X, t_s = load_snapshots(results_path, load_time_steps(results_path))

        # project data onto eigenvectors to determine coeffients
        c = (Z'*M*Z) \ Z'*M*X

        N_s = size(c,2)
        # calculate energy in each mode
        E = real([dot(c[i,j]*Z[:,i], M * (c[i,j]*Z[:,i]))
            for i in 1:r, j in 1:N_s])
        
        # sort modes by decreasing energy in initial data
        inds_no_cutoff = sortperm(-E[:,1])
        inds = inds_no_cutoff[E[inds_no_cutoff,1] .> tol]
        
        return DynamicalAnalysisResults(exp.(eigenvalues[inds]*t_s), 
            eigenvalues[inds], Z[:,inds],
            find_conjugate_pairs(eigenvalues[inds]), 
            c[inds,:], Z*c, E[inds,:]) 
    else 
        dt = 1.0
        return DynamicalAnalysisResults(exp.(dt.*eigenvalues), 
            eigenvalues, Z, find_conjugate_pairs(eigenvalues), nothing, nothing, nothing)
    end

end

"""Approximate the Koopman operator from simulation trajectory data"""
analyze(analysis::KoopmanAnalysis, range=nothing) = analyze(analysis, StandardDMD(), range)

function analyze(analysis::KoopmanAnalysis, 
    algorithm::AbstractKoopmanAlgorithm, 
    range=nothing, samples=nothing)

    (; svd_tol, proj_tol, r, results_path) = analysis

    # load time steps
    time_steps = load_time_steps(results_path)
    if !isnothing(range)
        time_steps = time_steps[range[1]:range[2]]
    end
    if r >= length(time_steps)
        r = length(time_steps)-1
    end

    # set up data matrices
    
    if !isnothing(samples)
        (X, Y, t_s) = samples
        c, Z, σ, r = dmd(X, Y, algorithm, r, svd_tol)
    else
        X, t_s = load_snapshots(results_path, time_steps)
        c, Z, σ, r = dmd(X[:,1:end-1],X[:,2:end], algorithm, r, svd_tol)
    end
    (r,N_s) = size(c)

    # calculate energy in each mode 
    E = real.([dot(c[i,j]*Z[:,i], (c[i,j]*Z[:,i]))
        for i in 1:r, j in 1:N_s])
         
    # sort modes by decreasing energy in initial data
    inds_no_cutoff = sortperm(-E[:,1])
    inds = inds_no_cutoff[E[inds_no_cutoff,1] .> proj_tol]
    
    return DynamicalAnalysisResults(σ[inds], 
        log.(σ[inds])./t_s, Z[:,inds], 
        find_conjugate_pairs(σ[inds]), c[inds,:],
        Z*c, E[inds,:])
end


"""Approximate the Koopman generator (requires known dynamics + simulation trajectory data)"""
function analyze(analysis::KoopmanAnalysis,
    algorithm::GeneratorDMD, range=nothing)

    (; r, svd_tol, proj_tol, results_path) = analysis
    (; basis, basis_derivatives, f) = algorithm

    # load time step and ensure rank is suitable
    time_steps = load_time_steps(results_path)
    if !isnothing(range)
        time_steps = time_steps[range[1]:range[2]]
    end
    if r >= length(time_steps)
        r = length(time_steps)-1
    end

    # set up data matrices
    U, t_s = load_snapshots(results_path, time_steps)
    X = vcat([ψ.(U) for ψ ∈ basis]...)
    N_s = size(X,2)
    dUdt = hcat([f(U[:,i]) for i in 1:N_s]...)
    Y = vcat([dψdu.(U) .* dUdt for dψdu ∈ basis_derivatives]...)

    # perform (standard) DMD    
    c, Z, λ, r = dmd(X,Y,StandardDMD(),r,svd_tol)

    # calculate energy
    E = real([dot(c[i,j]*Z[:,i], M * (c[i,j]*Z[:,i]))
    for i in 1:r, j in 1:N_s])

    # sort modes by decreasing energy in initial data
    inds_no_cutoff = sortperm(-E[:,1])
    inds = inds_no_cutoff[E[inds_no_cutoff,1] .> proj_tol]

    return DynamicalAnalysisResults(exp.(λ[inds].*t_s), 
        λ[inds], Z[:,inds], find_conjugate_pairs(λ[inds]), 
        c[inds,:], Z*c, E[inds,:])
end

function forecast(results::DynamicalAnalysisResults, Δt::Float64; starting_step::Int=0)
    (; c, λ, Z) = results
    n_modes = size(Z,2)
    if starting_step == 0
        c0 = c[:,end]
    else
        c0 = c[:,starting_step]
    end
    return sum(Z[:,j]*exp(λ[j]*Δt)*c0[j] for j in 1:n_modes)
end

function forecast(results::DynamicalAnalysisResults, Δt::Float64, c0::Vector{ComplexF64})
    (; λ, Z) = results
    n_modes = size(Z,2)
    return sum(Z[:,j]*exp(λ[j]*Δt)*c0[j] for j in 1:n_modes)
end

function analyze_running(analysis::KoopmanAnalysis,
    algorithm::AbstractKoopmanAlgorithm, 
    range::NTuple{2,Int64};
    integrator=nothing,
    sampling_strategy=nothing,
    window_size=nothing)

    (; results_path, N_p, N_c, N_e) = analysis

    n_s = range[2]-range[1] + 1
    model = Vector{DynamicalAnalysisResults}(undef, n_s - 1)

    time_steps = load_time_steps(results_path)
    U, t_s = load_snapshots(results_path, time_steps[range[1]:range[2]])
    println("t_s = ", t_s)
    time_steps = load_time_steps(results_path)

    if !isnothing(sampling_strategy)
        (; n) = sampling_strategy
    else
        n = 1
    end
    X = Matrix{Float64}(undef, N_p*N_c*N_e, (n_s-1)*n)
    Y = Matrix{Float64}(undef, N_p*N_c*N_e, (n_s-1)*n)
    
    for i in (range[1]+1):range[2]
        
        open(string(results_path,"screen.txt"), "a") do io
            println(io, "Koopman analysis of time step", time_steps[i], " of ", time_steps[end])
        end

        # set finite window if desired
        if isnothing(window_size) || (i - range[1]) < window_size
            window_size_new = i-range[1]
        else
            window_size_new = window_size
        end

        # sample around trajectory
        if !isnothing(sampling_strategy)
            u, _ = load_solution(results_path,time_steps[i-1])
            reinit!(integrator, u)
            sample_range = ((i-range[1]-1)*n+1, (i-range[1]-1)*n+n)

            X[:,sample_range[1]:sample_range[2]], Y[
                :,sample_range[1]:sample_range[2]] = generate_samples(
                sampling_strategy, integrator, t_s)

            samples = (X[:, (((i-range[1])-window_size_new)*n + 1) : sample_range[2]],
                Y[:,(((i-range[1])-window_size_new)*n + 1) : sample_range[2]], t_s)

        # use trajectory data only
        else
            samples=nothing
        end

        model[i-range[1]] = analyze(analysis, 
            algorithm, (i-window_size_new,i), samples)

        save_object(string(results_path, "model_", i, ".jld2"), 
            model[i-range[1]])
    end

    return model, X, Y
end

function forecast(analysis::KoopmanAnalysis, Δt::Float64, 
    range::NTuple{2,Int64}, forecast_name::String="forecast"; window_size=nothing, algorithm::AbstractKoopmanAlgorithm=StandardDMD(), new_projection=false)
    
    (; results_path, N_p, N_c, N_e) = analysis
    time_steps = load_time_steps(results_path)
    forecast_path = new_path(string(results_path, forecast_name, "/"),
        true,true)
    save_object(string(forecast_path, "time_steps.jld2"), time_steps)
    if koopman_generator
        solver = load_solver(results_path)
        f(u::Vector{Float64}) = vec(rhs!(similar(reshape(u,(N_p,N_c,N_e))),
            reshape(u,(N_p,N_c,N_e)),solver,0.0))  # assume time invariant
    end

    u = Array{Float64,3}[]
    t = Float64[]
    model = DynamicalAnalysisResults[]
    for i in (range[1]+1):range[2]
        if isnothing(window_size) || (i - range[1]) < window_size
            window_size_new = i-range[1]
        else
            window_size_new = window_size
        end
        if koopman_generator
            push!(model,analyze(analysis,algorithm,f,(i-window_size_new,i)))
        else
            push!(model,analyze(analysis, algorithm, (i-window_size_new,i)))
        end
        u0, t0 = load_solution(results_path, time_steps[i-1])
        if new_projection
            c = pinv(Z) * vec(u0)
            push!(u,reshape(real.(forecast(last(model), Δt, c)[1:N_p*N_c*N_e]), (N_p,N_c,N_e)))
        else
            push!(u,reshape(real.(forecast(last(model), Δt, last(model).c[:,end]))[1:N_p*N_c*N_e], (N_p,N_c,N_e)))
        end
        push!(t, t0 + Δt)
        save(string(forecast_path, "sol_", time_steps[i], ".jld2"),
            Dict("u" => last(u), "t" => last(t)))
    end
    return forecast_path, model
end

function monomial_basis(p::Int)
    return [u->u.^k for k in 1:p]
end

function monomial_derivatives(p::Int)
    return [u->k.*u.^(k-1) for k in 1:p]
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

find_conjugate_pairs(::Vector{Float64}; tol=1.0e-8) = nothing

function make_dmd_matrices(X::AbstractMatrix{Float64},Y::AbstractMatrix{Float64}, algorithm::ExtendedDMD)
    (; basis) = algorithm

    Φ_X= vcat([ϕ.(X) for ϕ ∈ basis]...)
    Φ_Y= vcat([ϕ.(Y) for ϕ ∈ basis]...)

    return Φ_X' * Φ_X, Φ_Y' * Φ_X, Φ_Y' * Φ_Y
end

function make_dmd_matrices(X::AbstractMatrix{Float64},Y::AbstractMatrix{Float64}, algorithm::KernelDMD)

    (; k, η) = algorithm
    N_s = size(X,2)
    G_hat = [k(X[:,i], X[:,j]) for i in 1:N_s, j in 1:N_s]

    return (G_hat + η*norm(G_hat)*I,
        [k(Y[:,i], X[:,j]) for i in 1:N_s, j in 1:N_s], 
        [k(Y[:,i], Y[:,j]) for i in 1:N_s, j in 1:N_s])
end

function dmd(X::Matrix{Float64},Y::Matrix{Float64}, algorithm::StandardDMD,
    r::Int=0, svd_tol=1.0e-10)

    (; basis) = algorithm

    Φ_X = vcat([ϕ.(X) for ϕ ∈ basis]...)
    Φ_Y = vcat([ϕ.(Y) for ϕ ∈ basis]...)
    
    if r > 0
        # SVD (i.e. POD) of initial states
        U_full, S_full, V_full = svd(Φ_X)

        U = U_full[:,1:r][:,S_full[1:r] .> svd_tol]
        S = S_full[1:r][S_full[1:r] .> svd_tol]
        V = V_full[:,1:r][:,S_full[1:r] .> svd_tol]

        # eigendecomposition of DMD matrix (projected onto singular vectors)
        K_hat_trans_decomp = eigen((U') * Φ_Y * V * inv(Diagonal(S)))

        # map eigenvectors back up into full space
        Z_unscaled = Φ_Y*V*inv(Diagonal(S))*K_hat_trans_decomp.vectors
        σ = K_hat_trans_decomp.values
        r = length(σ)
    else
        K_trans = Y*pinv(X)
        K_trans_decomp = eigen(K_trans)

        Z_unscaled = K_trans_decomp.vectors
        σ = K_trans_decomp.values
        r = length(σ)
    end

    Z = hcat([Z_unscaled[:,i] ./ norm(Z_unscaled[:,i]) for i in 1:r]...)
    c = pinv(Z)*X

    return c, Z, σ, r
end

function dmd(X::AbstractMatrix{Float64},Y::AbstractMatrix{Float64},
    algorithm::Union{ExtendedDMD,KernelDMD}, r::Int=0, svd_tol=1.0e-10)

    G_hat, A_hat, _ = make_dmd_matrices(X,Y,algorithm)

    # diagonalize the Gram matrix (method of snapshots)
    G_hat_decomp = eigen(G_hat)
    S_full = sqrt.(abs.(G_hat_decomp.values))

    # order by descending singular values
    sort!(S_full,rev=true)
    U_full = G_hat_decomp.vectors[:,sortperm(S_full)]
    
    # truncate the SVD if needed
    if r > 0
        U = U_full[:,1:r][:,S_full[1:r] .> svd_tol]
        S = S_full[1:r][S_full[1:r] .> svd_tol]
    else
        U = U_full[:,S_full .> svd_tol]
        S = S_full[S_full .> svd_tol]
    end
    r = length(S)

    # koopman eigenfunctions evaluated at the data points
    K_hat_decomp = eigen((inv(Diagonal(S))*U') * A_hat * (U * inv(Diagonal(S))))
    V_hat = hcat([K_hat_decomp.vectors[:,i] ./ norm(K_hat_decomp.vectors[:,i])
        for i in 1:r]...)
    V = U * Diagonal(S) * V_hat
    c = transpose(V)

    # koopman modes
    W_hat = pinv(V_hat)
    Z = hcat([transpose(transpose(W_hat[i,:]) * inv(Diagonal(S)) * U' * X') 
        for i in 1:r]...)

    # koopman eigenvalues
    σ = Complex.(K_hat_decomp.values)

    return c, Z, σ, r

end

function dmd(X::AbstractMatrix{Float64},Y::AbstractMatrix{Float64},
    algorithm::KernelResDMD, r::Int=0, svd_tol=1.0e-10)

    (; k, ϵ) = algorithm

    (X_1, Y_1) = (X[:,2:2:end], Y[:,2:2:end])
    (X_2, Y_2) = (X[:,1:2:end], Y[:,1:2:end])

    G_hat, A_hat, _ = make_dmd_matrices(X_1,Y_1,KernelDMD(k,0.0))

    # diagonalize the Gram matrix (method of snapshots)
    G_hat_decomp = eigen(G_hat)
    S_full = sqrt.(abs.(G_hat_decomp.values))
    U = G_hat_decomp.vectors[:,S_full .> svd_tol]
    S = S_full[S_full .> svd_tol]
    r = length(S)

    # koopman eigenfunctions evaluated at the data points
    K_hat_decomp = eigen((inv(Diagonal(S))*U') * A_hat * (U * inv(Diagonal(S))))
    V_hat = hcat([K_hat_decomp.vectors[:,i] ./ norm(K_hat_decomp.vectors[:,i])
        for i in 1:r]...)
    Q, _ = qr(V_hat')
    N_s1 = size(X_1,2)
    N_s2 = size(X_2,2)

    G_X2 = [k(X_2[:,i], X_1[:,j]) for i in 1:N_s2, j in 1:N_s1]
    G_Y2 = [k(Y_2[:,i], X_1[:,j]) for i in 1:N_s2, j in 1:N_s1]

    println((size(inv(Diagonal(S))),size(V_hat)))
    ϕ_X = hcat([G_X2 * (U * inv(Diagonal(S))) * Q[:,j] for j in 1:r]...)
    ϕ_Y = hcat([G_Y2 * (U * inv(Diagonal(S))) * Q[:,j] for j in 1:r]...)
    G = ϕ_X'*ϕ_X
    A = ϕ_X'*ϕ_Y

    K_decomp = eigen(pinv(G)*A)
    Z = X_2*pinv(K_decomp.vectors)
    c = transpose(K_decomp.vectors)
    σ = Complex.(K_decomp.values)
   
    return c, Z, σ, r
end

function generate_samples(sampling_strategy::GaussianSampling,  integrator::ODEIntegrator, t_s=nothing)
    (; σ, n) = sampling_strategy
    int_copy = deepcopy(integrator)
    u_centre = int_copy.sol.u[1]
    N = length(u_centre)
    X = Matrix{Float64}(undef, N, n)
    Y = Matrix{Float64}(undef, N, n)
    for i in 1:n
        if i == 1
            reinit!(int_copy,u_centre)
        else
            reinit!(int_copy,u_centre + σ*randn(size(u_centre)))
        end
        if !isnothing(t_s)
            step!(int_copy, t_s, true)
        else
            step!(int_copy)
        end
        X[:,i] = vec(int_copy.sol.u[1])
        Y[:,i] = vec(int_copy.sol.u[end])
    end
    return X, Y
end

function plot_analysis(analysis::AbstractDynamicalAnalysis,
    results::DynamicalAnalysisResults; e=1, i=1, modes = 0,
    scale=true, title="spectrum.pdf", xlims=nothing, ylims=nothing, 
        centre_on_circle=true)
    l = @layout [a{0.5w} b; c]
    if scale
        coeffs=results.c[:,i]
    else
        coeffs=ones(length(results.c[:,i]))
    end

    if modes == 0
        modes = 1:length(results.λ)
        conjugate_pairs = results.conjugate_pairs
    elseif modes isa Int
        modes = [modes]
        conjugate_pairs=nothing
    else
        conjugate_pairs = find_conjugate_pairs(results.σ[modes])
    end

    if isnothing(xlims)
        xlims=(minimum(real.(results.λ[modes]))*1.05,
            maximum(real.(results.λ[modes]))*1.05)
    end

    if isnothing(ylims)
        ylims=(minimum(imag.(results.λ[modes]))*1.05,
            maximum(imag.(results.λ[modes]))*1.1)
    end

    if centre_on_circle
        xlims_discrete = (-1.5,1.5)
        ylims_discrete = (-1.5,1.5)
    else
        xlims_discrete = nothing
        ylims_discrete = nothing
    end

    p = plot(plot_spectrum(analysis,results.λ[modes], 
            label="\\tilde{\\lambda}", unit_circle=false, 
            xlims=xlims,
            ylims=ylims,
            title="continuous_time.pdf", xscale=-0.03, yscale=0.03), 
        plot_spectrum(analysis,results.σ[modes], 
            label="\\exp(\\tilde{\\lambda} t_s)",
            unit_circle=true, xlims=xlims_discrete,ylims=ylims_discrete,
            title="discrete_time.pdf"),
        plot_modes(analysis,results.Z[:,modes]::Matrix{ComplexF64}; e=e, 
            coeffs=coeffs[modes], conjugate_pairs=conjugate_pairs),
            layout=l, framestyle=:box)
    
    savefig(p, string(analysis.results_path, title))
    return p
end

function plot_spectrum(eigs::Vector{Vector{ComplexF64}}, plots_path::String; 
    ylabel="\\lambda", xlims=nothing, ylims=nothing, title="spectra.pdf", 
    labels=["Upwind", "Central"])
    p = plot(legendfontsize=10, xlabelfontsize=13, ylabelfontsize=13, xtickfontsize=10, ytickfontsize=10)
    max_real = @sprintf "%.2e" maximum(real.(eigs[1]))
    for i in 1:length(eigs)
        max_real = @sprintf "%.2e" maximum(real.(eigs[i]))
        sr = @sprintf "%.2f" maximum(abs.(eigs[i]))
        plot!(p, eigs[i], 
        xlabel= latexstring(string("\\mathrm{Re}\\,(", ylabel, ")")), 
        ylabel= latexstring(string("\\mathrm{Im}\\,(", ylabel, ")")), 
        legend=:topleft,
        label=string(labels[i]," (max Re(λ): ", max_real,")"),  
        markershape=:star, seriestype=:scatter,
        markersize=5,
        markerstrokewidth=0, 
        size=(400,400)
        )
    end
    savefig(p, string(plots_path, title))
    return p
end

function plot_spectrum(analysis::AbstractDynamicalAnalysis, 
    eigs::Vector{ComplexF64}; label="\\exp(\\tilde{\\lambda} t_s)",unit_circle=true, xlims=nothing, ylims=nothing,
    xscale=0.02, yscale=0.07, title="spectrum.pdf", numbering=true)

    if unit_circle
        t=collect(LinRange(0.0, 2.0*π,100))
        p = plot(cos.(t), sin.(t), aspect_ratio=:equal, 
            linecolor="black",xticks=-1.0:1.0:1.0, yticks=-1.0:1.0:1.0)
    else
        p = plot()
    end

    plot!(p, eigs, xlabel=latexstring(string("\\mathrm{Re}\\,(", label, ")")), 
        ylabel=latexstring(string("\\mathrm{Im}\\,(", label, ")")), 
        xlims=xlims, ylims=ylims,legend=false,
        seriestype=:scatter)

    if !unit_circle && numbering
        annotate!(real(eigs) .+ xscale*(xlims[2]-xlims[1]), 
            imag(eigs)+sign.(imag(eigs) .+ 1.0e-15)*yscale*(ylims[2]-ylims[1]),
            text.(1:length(eigs), :right, 8))
    elseif numbering
        annotate!(real(eigs).-0.1, imag(eigs),
            text.(1:length(eigs), :right, 8))
    end
    savefig(p, string(analysis.results_path, title))

    return p
end

function plot_modes(analysis::AbstractDynamicalAnalysis, 
    Z::Matrix{ComplexF64}; e=1, 
    coeffs=nothing, projection=nothing,
    conjugate_pairs=nothing)
    #println("conj pairs: ", conjugate_pairs)
    (; N_p, N_c, N_e, plotter) = analysis
    (; x_plot, V_plot) = plotter

    n_modes = size(Z,2)
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

        sol = reshape(Z[:,j][1:N_p*N_c*N_e],(N_p, N_c, N_e))
        u = convert(Matrix, V_plot * real(coeffs[j]*sol[:,e,:]))

        if conjugate_pairs[j] == 0
            linelabel = string(j)
            scale_factor = 1.0
        else
            linelabel = string(j, ",", conjugate_pairs[j])
            scale_factor = 2.0
            skip[conjugate_pairs[j]] = true
        end

        plot!(p,vec(vcat(x_plot[1],fill(NaN,1,N_e))), 
            vec(vcat(scale_factor*u,fill(NaN,1,N_e))), 
            label=latexstring(linelabel),
            ylabel="Koopman Modes",
            legendfontsize=6)
    end

    if !isnothing(projection)
        sol = reshape(projection,(N_p, N_c, N_e))
        linelabel = string("\\mathrm{Projection}")
        u = convert(Matrix, V_plot * real(sol[:,e,:]))
        plot!(p,vec(vcat(x_plot[1],fill(NaN,1,N_e))), 
            vec(vcat(u,fill(NaN,1,N_e))), 
            label=latexstring(linelabel), xlabel=latexstring("x"), 
            linestyle=:dash, linecolor="black")
    end

    savefig(p, string(analysis.results_path, "modes.pdf")) 
    return p
end

@recipe function plot(eigs::Vector{Vector{ComplexF64}};
    symbol="\\lambda", labels=["Central", "Upwind"])

    legendfontsize --> 10
    xlabelfontsize --> 13
    ylabelfontsize --> 13
    xtickfontsize --> 10
    ytickfontsize --> 10
    markersize --> 5
    markerstrokewidth --> 0
    legend --> :topleft
    fontfamily --> "Computer Modern"

    for i in eachindex(eigs)
        @series begin
            #max_real = @sprintf "%.2e" maximum(real.(eigs[i]))
            #sr = @sprintf "%.2f" maximum(abs.(eigs[i]))
            seriestype --> :scatter
            markershape --> :star
            label --> labels[i]
            xlabel --> latexstring(string("\\mathrm{Re}\\,(", symbol, ")"))
            ylabel --> latexstring(string("\\mathrm{Im}\\,(", symbol, ")"))
            eigs[i]
        end
    end
end