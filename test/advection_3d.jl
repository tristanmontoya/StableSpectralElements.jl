function advection_3d()

    a = (1.0, 1.0, 1.0)  # advection velocity
    L = 1.0  # domain length
    T = 1.0  # end time

    conservation_law = LinearAdvectionEquation(a)
    exact_solution = InitialDataCosine(1.0, (2π / L, 2π / L, 2π / L))

    M = 2
    p = 4

    reference_approximation = ReferenceApproximation(
        ModalTensor(p),
        Tet(),
        mapping_degree = 4,
        sum_factorize_vandermonde = false,
    )

    form = StandardForm(
        mapping_form = SkewSymmetricMapping(),
        inviscid_numerical_flux = CentralNumericalFlux(),
    )

    uniform_mesh = uniform_periodic_mesh(
        reference_approximation,
        ((0.0, L), (0.0, L), (0.0, L)),
        (M, M, M),
    )

    mesh = warp_mesh(uniform_mesh, reference_approximation, 0.1, L)

    spatial_discretization =
        SpatialDiscretization(mesh, reference_approximation, ChanWilcoxMetrics())

    results_path = save_project(
        conservation_law,
        spatial_discretization,
        exact_solution,
        form,
        (0.0, T),
        "results/advection_3d/",
        overwrite = true,
        clear = true,
    )
    CFL = 0.1
    h = L / (reference_approximation.N_p * spatial_discretization.N_e)^(1 / 3)
    dt = CFL * h / sqrt(a[1]^2 + a[2]^2 + a[3]^2)

    ode_problem = semidiscretize(
        conservation_law,
        spatial_discretization,
        exact_solution,
        form,
        (0.0, T),
        ReferenceOperator(),
        BLASAlgorithm(),
    )

    sol = solve(
        ode_problem,
        CarpenterKennedy2N54(),
        adaptive = false,
        dt = dt,
        save_everystep = false,
        callback = save_callback(results_path, (0.0, T), floor(Int, T / (dt * 50))),
    )

    error_analysis = ErrorAnalysis(
        results_path,
        conservation_law,
        spatial_discretization,
        JaskowiecSukumarQuadrature(2p + 3),
    )

    error_results = analyze(error_analysis, last(sol.u), exact_solution, T)

    time_steps = load_time_steps(results_path)
    conservation_results = analyze(
        PrimaryConservationAnalysis(results_path, conservation_law, spatial_discretization),
        time_steps,
    )
    energy_results = analyze(
        EnergyConservationAnalysis(results_path, conservation_law, spatial_discretization),
        time_steps,
    )

    return error_results[1],
    conservation_results.E[end, 1] - conservation_results.E[1, 1],
    maximum(abs.(energy_results.dEdt[:, 1]))

end
