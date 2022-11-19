function test_2d(
    approx_type::AbstractApproximationType,
    element_type::AbstractElemShape,
    conservation_law::AbstractConservationLaw,
    initial_data::AbstractGridFunction{2},
    form::AbstractResidualForm,
    operator_algorithm::AbstractOperatorAlgorithm,
    strategy::AbstractStrategy,
    M::Int, 
    test_name::String)

    exact_solution = ExactSolution(conservation_law,initial_data)

    reference_approximation = ReferenceApproximation(
        approx_type, element_type, 
        mapping_degree=approx_type.p)

    mesh = warp_mesh(uniform_periodic_mesh(
        reference_approximation.reference_element, 
        ((0.0,1.0),(0.0,1.0)), (M,M)), 
        reference_approximation.reference_element, 0.1)

    spatial_discretization = SpatialDiscretization(mesh, 
        reference_approximation)

    results_path = save_project(conservation_law,
        spatial_discretization, initial_data, form, (0.0, 1.0),
        string("results/", test_name,"/"), overwrite=true, clear=true)

    ode_problem = semidiscretize(conservation_law,
        spatial_discretization,
        initial_data, 
        form,
        (0.0, 1.0),
        strategy,
        operator_algorithm)
    
    save_solution(ode_problem.u0, 0.0, results_path, 0)
    sol = solve(ode_problem, CarpenterKennedy2N54(),
        adaptive=false, dt=0.0005/M)
    save_solution(last(sol.u), last(sol.t), results_path, "final")

    error_analysis = ErrorAnalysis(results_path, conservation_law, 
        spatial_discretization)
    conservation_analysis = PrimaryConservationAnalysis(results_path, 
        conservation_law, spatial_discretization)
    energy_analysis = EnergyConservationAnalysis(results_path, 
        conservation_law, spatial_discretization)

    return (analyze(error_analysis, last(sol.u), exact_solution, 1.0)..., 
        analyze(conservation_analysis)[3][1], analyze(energy_analysis)[3][1])
end