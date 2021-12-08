function run_refinement(conservation_law::ConservationLaw{d,N_eq},        
    reference_approximation::ReferenceApproximation{d},
    initial_data::AbstractInitialData,
    form::AbstractResidualForm,
    strategy::AbstractStrategy,
    tspan::NTuple{2,Float64},
    sequence::Vector{Int},
    time_integrator::OrdinaryDiffEqAlgorithm,
    mesh_gen_func::Function,
    dt_func::Function) where {d,N_eq}

    number_of_grids = length(sequence)
    sol = Vector{ODESolution}(undef, number_of_grids)

    for i in 1:length(sequence)
        M = sequence[i]
        mesh = mesh_gen_func(M)
        spatial_discretization = SpatialDiscretization(mesh, reference_approximation)

        ode_problem = semidiscretize(conservation_law, 
            spatial_discretization,
            initial_data, form,
            tspan, strategy)

        sol[i] = solve(ode_problem, time_integrator, adaptive=false, 
            dt_func(M), save_everystep=false)
    end

    return sol
end