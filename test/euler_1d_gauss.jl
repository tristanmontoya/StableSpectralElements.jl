function euler_1d_gauss()
    T = 2.0
    L = 2.0

    conservation_law = EulerEquations{1}(1.4)

    function exact_sol(x, t)
        γ = 1.4
        ρ = 1.0 + 0.2sin(π * x)
        v = 1.0
        E = 1.0 / (γ - 1) + 0.5 * ρ
        return SVector{3}(ρ, ρ * v, E)
    end

    p = 5
    M = 4

    reference_approximation = ReferenceApproximation(
        NodalTensor(p),
        Line(),
        volume_quadrature_rule = LGQuadrature(p),
    )

    mesh = uniform_periodic_mesh(reference_approximation, (0.0, L), M)

    spatial_discretization = SpatialDiscretization(mesh, reference_approximation)

    form =
        FluxDifferencingForm(inviscid_numerical_flux = EntropyConservativeNumericalFlux())

    ode = semidiscretize(
        conservation_law,
        spatial_discretization,
        exact_sol,
        form,
        (0.0, T),
        ReferenceOperator(),
        BLASAlgorithm(),
    )

    results_path = save_project(
        conservation_law,
        spatial_discretization,
        exact_sol,
        form,
        (0.0, T),
        "results/euler_1d/",
        overwrite = true,
        clear = true,
    )

    dt = T / 1000

    sol = solve(
        ode,
        CarpenterKennedy2N54(),
        dt = dt,
        adaptive = false,
        save_everystep = false,
        callback = save_callback(results_path, (0.0, T), floor(Int, T / (dt * 50))),
    )

    error_analysis = ErrorAnalysis(results_path, conservation_law, spatial_discretization)

    error_results = analyze(error_analysis, last(sol.u), exact_sol, T)

    time_steps = load_time_steps(results_path)
    conservation_results = analyze(
        PrimaryConservationAnalysis(results_path, conservation_law, spatial_discretization),
        time_steps,
    )
    entropy_results = analyze(
        EntropyConservationAnalysis(results_path, conservation_law, spatial_discretization),
        time_steps,
    )

    return error_results,
    conservation_results.E[end, :] - conservation_results.E[1, :],
    maximum(abs.(entropy_results.dEdt[:, 1]))
end
