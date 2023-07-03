function euler_vortex_2d()
    L = 20.0  # domain length
    T = 20.0  # end time

    conservation_law = EulerEquations{2}(1.4)
    exact_solution = IsentropicVortex(conservation_law);

    p = 3
    M = 4

    form = StandardForm()

    reference_approximation = ReferenceApproximation(
        NodalTensor(p), Quad(), volume_quadrature_rule=LGLQuadrature(p),
        facet_quadrature_rule=LGLQuadrature(p))

    mesh = uniform_periodic_mesh(reference_approximation, ((-L/2, L/2),(-L/2, L/2)), (M,M))

    spatial_discretization = SpatialDiscretization(mesh, reference_approximation)


    results_path = save_project(conservation_law,
        spatial_discretization, exact_solution, form, (0.0, T),
        "results/euler_vortex_2d/", overwrite=true, clear=true);

    ode = semidiscretize(conservation_law, spatial_discretization, 
        exact_solution, form, (0.0, T));

    dt = T/1000
    sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
                dt=dt, adaptive=false, save_everystep=false, callback=save_callback(results_path, (0.0,T),  
            floor(Int, T/(dt*50))))

    error_analysis = ErrorAnalysis(results_path, conservation_law, 
        spatial_discretization, LGQuadrature(3*p))
    error_results = analyze(error_analysis, last(sol.u), exact_solution, T, normalize=true)

    conservation_analysis = PrimaryConservationAnalysis(results_path, 
        conservation_law, spatial_discretization)
    conservation_results = analyze(conservation_analysis, 
        load_time_steps(results_path))

    return error_results, conservation_results.E[end,:] - conservation_results.E[1,:]
end