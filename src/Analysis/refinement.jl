function run_refinement(conservation_law::ConservationLaw{d,N_eq},        
    reference_approximation::ReferenceApproximation{d},
    initial_data::AbstractInitialData,
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
end