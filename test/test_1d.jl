function test_driver(
    reference_approximation::ReferenceApproximation,
    conservation_law::AbstractConservationLaw,
    initial_data::AbstractGridFunction{1},
    form::AbstractResidualForm,
    operator_algorithm::AbstractOperatorAlgorithm,
    strategy::AbstractStrategy,
    L::Float64,
    M::Int,
    T::Float64,
    dt::Float64,
    test_name::String)

    d = dim(reference_approximation.reference_element.elementType)

    if d == 1
        mesh = uniform_periodic_mesh(reference_approximation.reference_element,
            (0.0,L),M)
    else
        mesh = uniform_periodic_mesh(reference_approximation.reference_element,
            Tuple((0.0,L) for m in 1:d), Tuple(M for m in 1:d))
    end

    exact_solution = ExactSolution(conservation_law,initial_data)

    reference_approximation = ReferenceApproximation(
        approx_type, element_type, mapping_degree=approx_type.p)

    mesh = uniform_periodic_mesh(reference_approximation.reference_element, 
        (0.0,T), M)

    spatial_discretization = SpatialDiscretization(mesh, 
        reference_approximation)

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