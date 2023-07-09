function initialize(initial_data::AbstractGridFunction{d},
    conservation_law::AbstractConservationLaw,
    spatial_discretization::SpatialDiscretization{d}) where {d}

    (; geometric_factors) = spatial_discretization
    (; N_q, V, W) = spatial_discretization.reference_approximation
    (; xyzq) = spatial_discretization.mesh
    N_p, N_c, N_e = get_dof(spatial_discretization, conservation_law)

    if V isa UniformScalingMap
        u0 = Array{Float64}(undef, N_p, N_c, N_e)
        Threads.@threads for k in 1:N_e
            u0[:,:,k] .= evaluate(initial_data, Tuple(xyzq[m][:,k] for m in 1:d),0.0)
        end
    else
        u0 = Array{Float64}(undef, N_p, N_c, N_e)
        Threads.@threads for k in 1:N_e
            M =  Matrix(V' * W * Diagonal(geometric_factors.J_q[:,k]) * V)
            rhs = similar(u0[:,:,k])
            mul!(rhs, V' * W * Diagonal(geometric_factors.J_q[:,k]), 
                evaluate(initial_data, Tuple(xyzq[m][:,k] for m in 1:d),0.0))
            u0[:,:,k] .= M \ rhs
        end
    end
    return u0
end

function semidiscretize(
    conservation_law::AbstractConservationLaw{d,PDEType},spatial_discretization::SpatialDiscretization{d},
    initial_data,
    form::AbstractResidualForm,
    tspan::NTuple{2,Float64}, 
    strategy::AbstractStrategy=ReferenceOperator(),
    operator_algorithm::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm();
    mass_matrix_solver::AbstractMassMatrixSolver=CholeskySolver(spatial_discretization),
    periodic_connectivity_check::Bool=true,
    tol::Float64=1e-12) where {d, PDEType}

    if periodic_connectivity_check
        normal_error = check_normals(spatial_discretization)
        for m in eachindex(normal_error)
            max = maximum(abs.(normal_error[m]))
            if max > tol
                error(string("Connectivity Error: Facet normals not equal and opposite. If this is not a periodic problem, or if you're alright with a loss of conservation, run again with periodic_connectivity_check=false. Max error = ", max))
            end
        end
    end

    u0 = initialize(initial_data, conservation_law, spatial_discretization)

    if PDEType == SecondOrder && strategy isa ReferenceOperator
        @warn "Reference-operator approach only implemented for first-order equations. Using physical-operator formulation."
        return semidiscretize(Solver(conservation_law,spatial_discretization,
            form, PhysicalOperator(), operator_algorithm, mass_matrix_solver),
            u0, tspan)
    else
        return semidiscretize(Solver(conservation_law,spatial_discretization,
        form,strategy,operator_algorithm,mass_matrix_solver), u0, tspan)
    end
end

function semidiscretize(solver::Solver, u0::Array{Float64,3},
    tspan::NTuple{2,Float64})
    return ODEProblem(rhs!, u0, tspan, solver)
end