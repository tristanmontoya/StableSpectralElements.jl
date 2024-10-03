function euler_vortex_2d_modal(M::Int = 4)

    mach_number = 0.4
    angle = 0.0
    L = 1.0
    γ = 1.4
    T = L / mach_number # end time
    strength = sqrt(2 / (γ - 1) * (1 - 0.75^(γ - 1))) # for central value of ρ=0.75

    conservation_law = EulerEquations{2}(γ)
    exact_solution = IsentropicVortex(
        conservation_law,
        θ = angle,
        Ma = mach_number,
        β = strength,
        R = 1.0 / 10.0,
        x_0 = (L / 2, L / 2),
    )

    p = 3

    form = FluxDifferencingForm(inviscid_numerical_flux = LaxFriedrichsNumericalFlux())

    reference_approximation =
        ReferenceApproximation(ModalTensor(p), Tri(), mapping_degree = p)

    uniform_mesh =
        uniform_periodic_mesh(reference_approximation, ((0.0, L), (0.0, L)), (M, M))

    mesh = warp_mesh(uniform_mesh, reference_approximation, ChanWarping(1.0 / 16.0, (L, L)))

    spatial_discretization =
        SpatialDiscretization(mesh, reference_approximation, ChanWilcoxMetrics())

    results_path = save_project(
        conservation_law,
        spatial_discretization,
        exact_solution,
        form,
        (0.0, T),
        "results/euler_vortex_2d_modal/",
        overwrite = true,
        clear = true,
    )

    dt = T / 1000

    ode = semidiscretize(
        conservation_law,
        spatial_discretization,
        exact_solution,
        form,
        (0.0, T),
        ReferenceOperator(),
    )

    sol = solve(
        ode,
        CarpenterKennedy2N54(),
        dt = dt,
        adaptive = false,
        save_everystep = false,
        callback = save_callback(results_path, (0.0, T), floor(Int, T / (dt * 50))),
    )

    error_analysis = ErrorAnalysis(results_path, conservation_law, spatial_discretization)
    error_results = analyze(error_analysis, last(sol.u), exact_solution, T)
    conservation_analysis =
        PrimaryConservationAnalysis(results_path, conservation_law, spatial_discretization)
    conservation_results = analyze(conservation_analysis, load_time_steps(results_path))

    return error_results, conservation_results.E[end, :] - conservation_results.E[1, :]
end
