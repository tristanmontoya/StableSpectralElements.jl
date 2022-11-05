function initialize(initial_data::AbstractGridFunction,
    conservation_law::AbstractConservationLaw,
    spatial_discretization::SpatialDiscretization{d}) where {d}

    @unpack M, geometric_factors = spatial_discretization
    @unpack N_q, V, W = spatial_discretization.reference_approximation
    @unpack xyzq = spatial_discretization.mesh
    N_p, N_c, N_e = get_dof(spatial_discretization, conservation_law)
    
    u0 = Array{Float64}(undef, N_p, N_c, N_e)
    Threads.@threads for k in 1:N_e
        rhs = similar(u0[:,:,k])
        mul!(rhs, V' * W * Diagonal(geometric_factors.J_q[:,k]), 
            evaluate(initial_data, Tuple(xyzq[m][:,k] for m in 1:d)))
        u0[:,:,k] = M[k] \ rhs
    end
    return u0
end

function semidiscretize(
    conservation_law::AbstractConservationLaw,spatial_discretization::SpatialDiscretization,
    initial_data::AbstractGridFunction, 
    form::AbstractResidualForm,
    tspan::NTuple{2,Float64}, 
    strategy::AbstractStrategy)

    u0 = initialize(
        initial_data,
        conservation_law,
        spatial_discretization)

    return semidiscretize(Solver(conservation_law,spatial_discretization,
        form,strategy),u0, tspan)
end

function semidiscretize(solver::Solver, u0::Array{Float64,3},
    tspan::NTuple{2,Float64})
    println("tspan = ", tspan)
    return ODEProblem(rhs!, u0, tspan, solver)
end