function initialize(initial_data::AbstractGridFunction,
    conservation_law::AbstractConservationLaw,
    spatial_discretization::SpatialDiscretization{d}) where {d}

    @unpack geometric_factors = spatial_discretization
    @unpack N_q, V, W = spatial_discretization.reference_approximation
    @unpack xyzq = spatial_discretization.mesh
    N_p, N_c, N_e = get_dof(spatial_discretization, conservation_law)
    
    u0 = Array{Float64}(undef, N_p, N_c, N_e)
    Threads.@threads for k in 1:N_e
        M =  Matrix(V' * W * Diagonal(geometric_factors.J_q[:,k]) * V)
        rhs = similar(u0[:,:,k])
        mul!(rhs, V' * W * Diagonal(geometric_factors.J_q[:,k]), 
            evaluate(initial_data, Tuple(xyzq[m][:,k] for m in 1:d)))
        u0[:,:,k] .= M \ rhs
    end
    return u0
end

function Solver(conservation_law::AbstractConservationLaw,     
    spatial_discretization::SpatialDiscretization,
    form::AbstractResidualForm,
    ::PhysicalOperator,
    operator_algorithm::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm(),
    mass_solver::AbstractMassMatrixSolver=WeightAdjustedSolver(spatial_discretization))

    operators = make_operators(spatial_discretization, form,
        operator_algorithm, mass_solver)

    return Solver(conservation_law, operators, mass_solver,
        spatial_discretization.mesh.mapP, form)
end

function Solver(conservation_law::AbstractConservationLaw,     
    spatial_discretization::SpatialDiscretization{d},
    form::AbstractResidualForm,
    ::ReferenceOperator,
    alg::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm(),
    mass_solver::AbstractMassMatrixSolver=WeightAdjustedSolver(spatial_discretization)) where {d}

    @unpack D, V, W, R, B = spatial_discretization.reference_approximation
    @unpack J_q, Λ_q, nJf, J_f = spatial_discretization.geometric_factors
    @unpack N_e = spatial_discretization

    halfWΛ = Array{Diagonal,3}(undef, d, d, N_e)
    halfN = Matrix{Diagonal}(undef, d, N_e)
    BJf = Vector{Diagonal}(undef, N_e)
    n_f = Vector{NTuple{d, Vector{Float64}}}(undef,N_e)

    Threads.@threads for k in 1:N_e
            halfWΛ[:,:,k] = [Diagonal(0.5 * W * Λ_q[:,m,n,k]) 
                for m in 1:d, n in 1:d]
            halfN[:,k] = [Diagonal(0.5 * nJf[m][:,k] ./ J_f[:,k]) for m in 1:d]
            BJf[k] = Diagonal(B .* J_f[:,k])
            n_f[k] = Tuple(nJf[m][:,k] ./ J_f[:,k] for m in 1:d)
    end

    operators = ReferenceOperators{d}(
        Tuple(make_operator(D[m], alg) for m in 1:d), 
        make_operator(V, alg), make_operator(R, alg), 
        W, B, halfWΛ, halfN, BJf, n_f)

    return Solver(conservation_law, operators, mass_solver,
        spatial_discretization.mesh.mapP, form)
end

function semidiscretize(
    conservation_law::AbstractConservationLaw,spatial_discretization::SpatialDiscretization,
    initial_data::AbstractGridFunction, 
    form::AbstractResidualForm,
    tspan::NTuple{2,Float64}, 
    strategy::AbstractStrategy=ReferenceOperator(),
    operator_algorithm::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm();
    mass_matrix_solver::AbstractMassMatrixSolver=CholeskySolver(spatial_discretization),
    periodic_connectivity_check::Bool=true,
    tol::Float64=1e-12)

    if periodic_connectivity_check
        normal_error = check_normals(spatial_discretization)
        for m in eachindex(normal_error)
            max = maximum(abs.(normal_error[m]))
            if max > tol
                error(string("Connectivity Error: Facet normals not equal and opposite. If this is not a periodic problem, or if you're alright with a loss of conservation, run again with periodic_connectivity_check=false. Max error = ", max))
            end
        end
    end

    u0 = initialize(initial_data,conservation_law, spatial_discretization)

    return semidiscretize(Solver(conservation_law,spatial_discretization,
        form,strategy,operator_algorithm,mass_matrix_solver),u0, tspan)
end

function semidiscretize(solver::Solver, u0::Array{Float64,3},
    tspan::NTuple{2,Float64})
    return ODEProblem(rhs!, u0, tspan, solver)
end