
function euler_3d(M::Int = 2)
    L = 2.0
    T = L
    Ω = ((0.0, L), (0.0, L), (0.0, L))

    conservation_law = EulerEquations{3}(1.4)
    exact_solution = EulerPeriodicTest(conservation_law)

    p = 4

    form =
        FluxDifferencingForm(inviscid_numerical_flux = EntropyConservativeNumericalFlux())

    reference_approximation =
        ReferenceApproximation(NodalTensor(p), Hex(), mapping_degree = p)

    uniform_mesh = uniform_periodic_mesh(reference_approximation, Ω, (M, M, M))

    mesh = warp_mesh(uniform_mesh, reference_approximation, ChanWarping(1 / 16, (L, L, L)))

    spatial_discretization =
        SpatialDiscretization(mesh, reference_approximation, ConservativeCurlMetrics())

    results_path = save_project(
        conservation_law,
        spatial_discretization,
        exact_solution,
        form,
        (0.0, T),
        "results/euler_periodic_3d/",
        overwrite = true,
        clear = true,
    )

    ode = semidiscretize(
        conservation_law,
        spatial_discretization,
        exact_solution,
        form,
        (0.0, T),
        parallelism = Serial(),
    )

    N_t = 25 * M * (p + 1)
    sol = solve(
        ode,
        DP8(),
        adaptive = false,
        dt = T / N_t,
        save_everystep = false,
        callback = save_callback(results_path, (0.0, T), floor(Int, N_t / 50)),
    )

    error_analysis = ErrorAnalysis(results_path, conservation_law, spatial_discretization)

    error = analyze(error_analysis, last(sol.u), exact_solution, T)

    conservation = analyze(
        PrimaryConservationAnalysis(results_path, conservation_law, spatial_discretization),
        load_time_steps(results_path),
        normalize = false,
    )
    entropy_analysis =
        EntropyConservationAnalysis(results_path, conservation_law, spatial_discretization)
    entropy_results = analyze(entropy_analysis, load_time_steps(results_path))

    mass = plot(conservation, ylabel = "Mass", 1)
    xmom = plot(conservation, ylabel = "Momentum (\$x_1\$)", 2)
    ymom = plot(conservation, ylabel = "Momentum (\$x_2\$)", 3)
    zmom = plot(conservation, ylabel = "Momentum (\$x_3\$)", 4)
    energy = plot(conservation, ylabel = "Energy", 5)
    entropy = plot(entropy_results, ylabel = "Entropy")

    plt = plot(mass, energy, xmom, ymom, zmom, entropy, size = (800, 400))
    savefig(plt, string(results_path, "conservation_metrics.pdf"))

    for i in eachindex(sol.u)
        postprocess_vtk(
            spatial_discretization,
            string(results_path, "solution_", i, ".vtu"),
            sol.u[i],
            variable_name = "Density",
        )
    end

    return error,
    conservation.E[end, :] - conservation.E[1, :],
    maximum(abs.(entropy_results.dEdt[:, 1]))
end
