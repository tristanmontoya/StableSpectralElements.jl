struct RefinementAnalysis{d} <: AbstractAnalysis
    exact_solution::AbstractParametrizedFunction{d}
    sequence_path::String
    analysis_path::String
end

struct RefinementAnalysisResults <: AbstractAnalysisResults
    error::Matrix{Float64} # columns are solution variables
    eoc::Matrix{Float64}
    dof::Matrix{Int} # columns are N_p N_eq N_el
end

function run_refinement(conservation_law::ConservationLaw{d,N_eq},        
    reference_approximation::ReferenceApproximation{d},
    initial_data::AbstractParametrizedFunction,
    form::AbstractResidualForm,
    strategy::AbstractStrategy,
    tspan::NTuple{2,Float64},
    sequence::Vector{Int},
    mesh_gen_func::Function,
    dt_func::Function,
    sequence_path::String;
    time_integrator::OrdinaryDiffEqAlgorithm=RK4(),
    n_s::Int=2) where {d,N_eq}

    number_of_grids = length(sequence)
    sol = Vector{ODESolution}(undef, number_of_grids)

    if isdir(sequence_path)
        dir_exists = true
        suffix = 1
        while dir_exists
            new_sequence_path = string(rstrip(sequence_path, '/'), 
                "_", suffix, "/")
            if !isdir(new_sequence_path)
                sequence_path=new_sequence_path
                dir_exists = false
            end
            suffix = suffix + 1
        end
    end
    mkpath(sequence_path)

    for i in 1:length(sequence)
        M = sequence[i]
        results_path = string(sequence_path, "grid_", i, "/")
        mesh = mesh_gen_func(M)

        spatial_discretization = SpatialDiscretization(mesh, reference_approximation)

        save_project(conservation_law,
            spatial_discretization, initial_data, form, 
            tspan, strategy, results_path, overwrite=true)

        ode_problem = semidiscretize(conservation_law, 
            spatial_discretization,
            initial_data, form,
            tspan, strategy)

        save_solution(ode_problem.u0, tspan[1], results_path, 0)
        sol[i] = solve(ode_problem, time_integrator, adaptive=false, 
            dt=dt_func(M), save_everystep=false,
            callback=save_callback(results_path, 
                floor(Int, (tspan[2]-tspan[1])/(dt_func(M)*(n_s-1)))))

        save_solution(last(sol[i].u), last(sol[i].t), results_path, "final")
    end

    return sequence_path
end

function analyze(analysis::RefinementAnalysis{d}) where {d}

    @unpack sequence_path, exact_solution = analysis

    results_path = string(sequence_path, "grid_1/")
    conservation_law, spatial_discretization = load_project(results_path) 
    (N_p, N_eq, N_el) = get_dof(spatial_discretization, conservation_law)
    dof = [N_p N_el]
    u, _ = load_solution(results_path, "final")
    error = transpose(analyze(ErrorAnalysis(results_path, conservation_law,  
    spatial_discretization), u, exact_solution))
    eoc = fill!(Array{Float64}(undef,1,N_eq), NaN)

    i = 2
    grid_exists = true
    while grid_exists
        results_path = string(sequence_path, "grid_", i, "/")
        conservation_law, spatial_discretization = load_project(results_path) 
        (N_p, N_eq, N_el) = get_dof(spatial_discretization, conservation_law)
        dof = [dof; [N_p N_el]]
        u, _ = load_solution(results_path, "final")
        error = [error; transpose(analyze(ErrorAnalysis(results_path,       
            conservation_law, spatial_discretization), u, exact_solution))]
        eoc = [eoc; transpose([ 
            (log(error[i,e]) - log(error[i-1,e])) /
                (log((dof[i,1]*dof[i,2])^(-1.0/d) ) - 
                log((dof[i-1,1]*dof[i-1,2])^(-1.0/d)))
            for e in 1:N_eq])]
    
        if !isdir(string(sequence_path, "grid_", i+1, "/"))
            grid_exists = false
        end
        i = i+1
    end
    return RefinementAnalysisResults(error, eoc, dof)
end

function plot_analysis(analysis::RefinementAnalysis{d},
    results::RefinementAnalysisResults; e=1) where {d}

    @unpack analysis_path = analysis
    @unpack error, dof = results

    if d == 1
        xlabel = latexstring("\\mathrm{DOF}")
    elseif d == 2
        xlabel = latexstring("\\sqrt{\\mathrm{DOF}}")
    else
        xlabel = latexstring(string("\\sqrt"),"[", d, "]{\\mathrm{DOF}}")
    end

    p = plot((dof[:,1].*dof[:,2]).^(1.0/d), error[:,e], 
        xlabel=xlabel, ylabel=latexstring("L^2 \\ \\mathrm{Error}"), 
        xaxis=:log, yaxis=:log, legend=false)
    savefig(p, string(analysis_path, "error.pdf"))
    return p
end
