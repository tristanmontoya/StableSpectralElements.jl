function burgers_fluxdiff_1d()
    L = 2.0  # domain length
    T = 0.3  # end time for one period

    conservation_law = InviscidBurgersEquation()
    initial_data = InitialDataGassner(π, 0.01)

    M = 20
    p = 7

    form =
        FluxDifferencingForm(inviscid_numerical_flux = EntropyConservativeNumericalFlux())

    reference_approximation = ReferenceApproximation(NodalTensor(p), Line())

    mesh = uniform_periodic_mesh(reference_approximation, (0.0, L), M)

    spatial_discretization = SpatialDiscretization(mesh, reference_approximation)

    results_path = save_project(
        conservation_law,
        spatial_discretization,
        initial_data,
        form,
        (0.0, T),
        "results/burgers_1d/",
        clear = true,
        overwrite = true,
    )

    ode_problem = semidiscretize(
        conservation_law,
        spatial_discretization,
        initial_data,
        form,
        (0.0, T),
    )

    CFL = 0.1
    h = L / (reference_approximation.N_p * spatial_discretization.N_e)
    dt = CFL * h / 1.0

    solve(
        ode_problem,
        CarpenterKennedy2N54(),
        adaptive = false,
        dt = dt,
        save_everystep = false,
        callback = save_callback(results_path, (0.0, T), floor(Int, T / (dt * 50))),
    )

    conservation_analysis =
        PrimaryConservationAnalysis(results_path, conservation_law, spatial_discretization)
    conservation_results = analyze(conservation_analysis, load_time_steps(results_path))

    energy_analysis =
        EnergyConservationAnalysis(results_path, conservation_law, spatial_discretization)
    energy_results = analyze(energy_analysis, load_time_steps(results_path))

    return maximum(abs.(conservation_results.dEdt[:, 1])),
    maximum(abs.(energy_results.dEdt[:, 1]))
end
