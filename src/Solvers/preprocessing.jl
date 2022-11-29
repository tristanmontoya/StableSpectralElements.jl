function initialize(initial_data::AbstractGridFunction,
    conservation_law::AbstractConservationLaw,
    spatial_discretization::SpatialDiscretization{d}) where {d}

    @unpack M, geometric_factors = spatial_discretization
    @unpack N_q, V, W = spatial_discretization.reference_approximation
    @unpack xyzq = spatial_discretization.mesh
    N_p, N_c, N_e = get_dof(spatial_discretization, conservation_law)
    
    u0 = Array{Float64}(undef, N_p, N_c, N_e)
    for k in 1:N_e
        rhs = similar(u0[:,:,k])
        mul!(rhs, V' * W * Diagonal(geometric_factors.J_q[:,k]), 
            evaluate(initial_data, Tuple(xyzq[m][:,k] for m in 1:d)))
        u0[:,:,k] = M[k] \ rhs
    end
    return u0
end

function Solver(conservation_law::AbstractConservationLaw,     
    spatial_discretization::SpatialDiscretization,
    form::AbstractResidualForm,
    ::ReferenceOperator,
    operator_algorithm::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm())

    operators = make_operators(spatial_discretization, form, operator_algorithm)
    return Solver(conservation_law, 
        operators,
        spatial_discretization.mesh.xyzq,
        spatial_discretization.mesh.mapP, form)
end

function Solver(conservation_law::AbstractConservationLaw,     
    spatial_discretization::SpatialDiscretization,
    form::AbstractResidualForm,
    ::PhysicalOperator,
    operator_algorithm::AbstractOperatorAlgorithm=BLASAlgorithm())

    operators = make_operators(spatial_discretization, form)

    # make sure physical operator matrices are being formed
    if !(operator_algorithm isa GenericMatrixAlgorithm || 
        operator_algorithm isa BLASAlgorithm)
        operator_algorithm = BLASAlgorithm()
    end

    return Solver(conservation_law, 
            [precompute(operators[k], operator_algorithm) 
                for k in 1:spatial_discretization.N_e],
            spatial_discretization.mesh.xyzq,
            spatial_discretization.mesh.mapP, form)
end

function Solver(conservation_law::AbstractConservationLaw,     
    spatial_discretization::SpatialDiscretization,
    form::AbstractResidualForm)
    return Solver(conservation_law,spatial_discretization,
        form, ReferenceOperator())
end

function precompute(operators::DiscretizationOperators{d}, 
    operator_algorithm::AbstractOperatorAlgorithm=BLASAlgorithm()) where {d}
    @unpack VOL, FAC, SRC, M, V, Vf, n_f, N_p, N_q, N_f = operators

    return DiscretizationOperators{d}(
        Tuple(make_operator(VOL[n], operator_algorithm) for n in 1:d),
        make_operator(FAC, operator_algorithm), 
        make_operator(SRC, operator_algorithm),
        M, V, Vf, n_f, N_p, N_q, N_f)
end

function semidiscretize(
    conservation_law::AbstractConservationLaw,spatial_discretization::SpatialDiscretization,
    initial_data::AbstractGridFunction, 
    form::AbstractResidualForm,
    tspan::NTuple{2,Float64}, 
    strategy::AbstractStrategy=ReferenceOperator(),
    operator_algorithm::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm())

    u0 = initialize(
        initial_data,
        conservation_law,
        spatial_discretization)

    return semidiscretize(Solver(conservation_law,spatial_discretization,
        form,strategy,operator_algorithm),u0, tspan)
end

function semidiscretize(solver::Solver, u0::Array{Float64,3},
    tspan::NTuple{2,Float64})
    return ODEProblem(rhs!, u0, tspan, solver)
end