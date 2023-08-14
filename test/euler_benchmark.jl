function init_benchmark_euler()
    mach_number = 0.4
    angle = 0.0
    L = 1.0
    γ=1.4
    T = L/mach_number # end time
    strength = sqrt(2/(γ-1)*(1-0.75^(γ-1))) # for central value of ρ=0.75

    conservation_law = EulerEquations{2}(γ)
    exact_solution = IsentropicVortex(conservation_law, θ=angle,
        Ma=mach_number, β=strength, R=1.0/10.0, x_0=(L/2,L/2));

    p = 3
    M = 4

    form = FluxDifferencingForm(inviscid_numerical_flux=LaxFriedrichsNumericalFlux(), 
        entropy_projection=true, facet_correction=true)

    reference_approximation = ReferenceApproximation(ModalTensor(p), 
        Tri(), mapping_degree=p, N_plot=25)

    uniform_mesh = uniform_periodic_mesh(reference_approximation, ((0.0,L),(0.0,L)), (M,M))

    mesh = warp_mesh(uniform_mesh, reference_approximation, ChanWarping(1/16, (L,L)))

    spatial_discretization = SpatialDiscretization(mesh, reference_approximation, project_jacobian=true)

    results_path = save_project(conservation_law,
        spatial_discretization, exact_solution, form, (0.0, T),
        "results/euler_vortex_2d/", overwrite=true, clear=true);

    ode = semidiscretize(conservation_law, spatial_discretization, 
        exact_solution, form, (0.0, T))

    return ode
end