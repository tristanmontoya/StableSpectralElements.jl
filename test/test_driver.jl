function test_driver(
    reference_approximation::ReferenceApproximation,
    conservation_law::AbstractConservationLaw,
    initial_data::AbstractGridFunction{d},
    form::AbstractResidualForm,
    operator_algorithm::AbstractOperatorAlgorithm,
    strategy::AbstractStrategy,
    L::Float64,
    M::Int,
    T::Float64,
    dt::Float64,
    mesh_perturb::Float64,
    test_name::String) where {d}

    if d == 1 spatial_discretization = SpatialDiscretization(           
        uniform_periodic_mesh(reference_approximation, 
            (0.0,L),M), reference_approximation)
    else spatial_discretization = SpatialDiscretization(
        warp_mesh(uniform_periodic_mesh(reference_approximation,
            Tuple((0.0,L) for m in 1:d), Tuple(M for m in 1:d)),
            reference_approximation, mesh_perturb), reference_approximation)
    end

    exact_solution = ExactSolution(conservation_law,initial_data)

    results_path = save_project(conservation_law,
        spatial_discretization, initial_data, form, (0.0, T),
        string("results/", test_name,"/"), overwrite=true, clear=true)

    ode_problem = semidiscretize(conservation_law,
        spatial_discretization,
        initial_data, 
        form,
        (0.0, T),
        strategy,
        operator_algorithm)

    sol = solve(ode_problem, CarpenterKennedy2N54(),
        adaptive=false, dt=dt, callback=save_callback(results_path, (0.0,T),  floor(Int, 1.0/(dt*50))))

    error = analyze(ErrorAnalysis(results_path, conservation_law, 
        spatial_discretization), last(sol.u), exact_solution, 1.0)
    conservation = analyze(PrimaryConservationAnalysis(results_path, 
        conservation_law, spatial_discretization), 
        load_time_steps(results_path))
    energy = analyze(EnergyConservationAnalysis(results_path,
        conservation_law, spatial_discretization),
        load_time_steps(results_path))

    return (error..., maximum(abs.(conservation.dEdt[:,1])), 
        maximum(energy.dEdt[:,1]))
end