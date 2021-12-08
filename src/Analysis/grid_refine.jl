function grid_refine(conservation_law::ConservationLaw{d,N_eq},        
    reference_approximation::ReferenceApproximation{d},
    initial_data::AbstractInitialData,
    form::AbstractResidualForm,
    strategy::AbstractStrategy,
    tspan::NTuple{2,Float64},
    sequence::Vector{Int},
    time_integrator::OrdinaryDiffEqAlgorithm,
    h_func::Function,
    mesh_gen_func::Function,
    dt_func::Function) where {d,N_eq}

    number_of_grids = length(sequence)
    err = Matrix{Float64}(undef, number_of_grids, N_eq)
    h = Vector{Float64}(undef, number_of_grids)

    for i in 1:length(sequence)
        M = sequence[i]
        mesh = mesh_gen_func(M)
        spatial_discretization = SpatialDiscretization(mesh, reference_approximation)

        ode_problem = semidiscretize(conservation_law, 
            spatial_discretization,
            initial_data, form,
            tspan, strategy)

        sol = solve(ode_problem, time_integrator, adaptive=false, 
            dt_func(M), save_everystep=false)

        h[i] = h_func(M)
        err[i,:] = [calculate_error(error_analysis, last(sol.u), u_exact, e=eq) 
            for eq in 1:N_eq]
    end
end