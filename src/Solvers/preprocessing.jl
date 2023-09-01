@inline function project_function!(u0::AbstractMatrix{Float64},
    initial_data::AbstractGridFunction{d}, ::UniformScalingMap, ::Diagonal,
    ::Vector{Float64}, x::NTuple{d,AbstractVector{Float64}}) where {d}
    u0 .= evaluate(initial_data,x,0.0)
end

@inline function project_function!(u0::AbstractMatrix{Float64},
    initial_data::AbstractGridFunction{d}, V::LinearMap, W::Diagonal,
    J::Diagonal, x::NTuple{d,AbstractVector{Float64}}) where {d}
    M = Matrix(V' * W * J * V)
    factorization = cholesky(Symmetric(M))
    mul!(u0, V' * W * J, evaluate(initial_data,x,0.0))
    ldiv!(factorization, u0)
end

    """Returns an array of initial data as nodal or modal DOF"""
@views function initialize(initial_data::AbstractGridFunction{d},
    spatial_discretization::SpatialDiscretization{d}, ::Serial) where {d}

    (; geometric_factors, N_e) = spatial_discretization
    (; V, W, N_p) = spatial_discretization.reference_approximation
    (; xyzq) = spatial_discretization.mesh
    (; N_c) = initial_data

    u0 = Array{Float64}(undef, N_p, N_c, N_e)

    @inbounds for k in 1:N_e
        project_function!(u0[:,:,k], initial_data, V, W, 
            Diagonal(geometric_factors.J_q[:,k]), 
            Tuple(xyzq[m][:,k] for m in 1:d))
    end
    return u0
end

@views function initialize(initial_data::AbstractGridFunction{d},
    spatial_discretization::SpatialDiscretization{d}, ::Threaded) where {d}

    (; geometric_factors, N_e) = spatial_discretization
    (; V, W, N_p) = spatial_discretization.reference_approximation
    (; xyzq) = spatial_discretization.mesh
    (; N_c) = initial_data

    u0 = Array{Float64}(undef, N_p, N_c, N_e)

    @inbounds Threads.@threads for k in 1:N_e
        project_function!(u0[:,:,k], initial_data, V, W, 
            Diagonal(geometric_factors.J_q[:,k]),
            Tuple(xyzq[m][:,k] for m in 1:d))
    end
    return u0
end

"""Returns an ODEProblem struct"""
function semidiscretize(
    conservation_law::AbstractConservationLaw{d,PDEType},spatial_discretization::SpatialDiscretization{d},
    initial_data,
    form::AbstractResidualForm,
    tspan::NTuple{2,Float64}, 
    strategy::AbstractStrategy=ReferenceOperator(),
    operator_algorithm::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm();
    mass_matrix_solver::AbstractMassMatrixSolver=WeightAdjustedSolver(spatial_discretization,operator_algorithm),
    periodic_connectivity_check::Bool=false,
    parallelism::AbstractParallelism=Threaded(),
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

    if Threads.nthreads() == 1
        parallelism = Serial()
    end

    u0 = initialize(initial_data, spatial_discretization, parallelism)

    if PDEType == SecondOrder && strategy isa ReferenceOperator
        @warn "Reference-operator approach not implemented for second-order equations. Using physical-operator formulation."
        return semidiscretize(Solver(conservation_law,spatial_discretization,
            form, PhysicalOperator(), operator_algorithm, mass_matrix_solver),
            u0, tspan)
    elseif strategy isa PhysicalOperator && form isa FluxDifferencingForm
        @warn "Physical-operator approach not implemented for flux-differencing form. Using reference-operator algorithm"
        return semidiscretize(Solver(conservation_law,spatial_discretization,
            form, ReferenceOperator(), operator_algorithm, mass_matrix_solver),
            u0, tspan)
    else
        return semidiscretize(Solver(conservation_law,spatial_discretization,
        form,strategy,operator_algorithm,mass_matrix_solver,parallelism), 
        u0, tspan)
    end
end

function semidiscretize(solver::Solver, u0::Array{Float64,3},
    tspan::NTuple{2,Float64})
    return ODEProblem(semi_discrete_residual!, u0, tspan, solver)
end

"""
Make operators for weak conservation form
"""
function make_operators(spatial_discretization::SpatialDiscretization{1}, 
    ::StandardForm{StandardMapping},
    alg::AbstractOperatorAlgorithm,
    mass_solver::AbstractMassMatrixSolver)

    (; N_e, reference_approximation) = spatial_discretization
    (; D, V, R, W, B) = reference_approximation
    (; nJf) = spatial_discretization.geometric_factors

    VOL = Vector{NTuple{1,LinearMap}}(undef,N_e)
    FAC = Vector{LinearMap}(undef,N_e)
    V_ar = Vector{LinearMap}(undef,N_e)
    R_ar = Vector{LinearMap}(undef,N_e)

    @inbounds Threads.@threads for k in 1:N_e
        M⁻¹ = mass_matrix_inverse(mass_solver, k)
        VOL[k] = (make_operator(M⁻¹ * Matrix(V' * D[1]' * W), alg),)
        FAC[k] = make_operator(-M⁻¹ * Matrix(V' * R' * B), alg)
        V_ar[k] = make_operator(reference_approximation.V, alg)
        R_ar[k] = make_operator(reference_approximation.R, alg)
    end
    return PhysicalOperators(VOL, FAC, V_ar, R_ar, nJf)
end

function make_operators(spatial_discretization::SpatialDiscretization{d}, 
    ::StandardForm{StandardMapping},
    alg::AbstractOperatorAlgorithm,
    mass_solver::AbstractMassMatrixSolver) where {d}

    (; N_e, reference_approximation, geometric_factors) = spatial_discretization
    (; V, R, W, B, D) = reference_approximation

    (; Λ_q, nJf, J_f) = apply_reference_mapping(geometric_factors,
        reference_approximation.reference_mapping)
    
    VOL = Vector{NTuple{d,LinearMap}}(undef,N_e)
    FAC = Vector{LinearMap}(undef,N_e)
    V_ar = Vector{LinearMap}(undef,N_e)
    R_ar = Vector{LinearMap}(undef,N_e)
    n_f = Array{Float64,3}(undef, d, N_f, N_e)

    @inbounds Threads.@threads for k in 1:N_e 
        M⁻¹ = mass_matrix_inverse(mass_solver, k)
        VOL[k] = Tuple(make_operator(M⁻¹ * Matrix(V' * 
            sum(D[m]' * Diagonal(W * Λ_q[:,m,n,k]) for m in 1:d)), alg) 
            for n in 1:d)
        FAC[k] = make_operator(-M⁻¹ * 
            Matrix(V' * R' * Diagonal(B * J_f[:,k])), alg)
        V_ar[k] = make_operator(reference_approximation.V, alg)
        R_ar[k] = make_operator(reference_approximation.R, alg)
        @inbounds for m in 1:d
            n_f[m,:,k] = nJf[m,:,k] ./ J_f[:,k]
        end
    end

    return PhysicalOperators(VOL, FAC, V_ar, R_ar, n_f)
end

function make_operators(spatial_discretization::SpatialDiscretization{d}, 
    ::StandardForm{SkewSymmetricMapping},
    alg::AbstractOperatorAlgorithm,
    mass_solver::AbstractMassMatrixSolver) where {d}

    (; N_e, reference_approximation, geometric_factors) = spatial_discretization
    (; V, R, W, B, D) = reference_approximation
    (; Λ_q, nJf, J_f) = apply_reference_mapping(geometric_factors,
        reference_approximation.reference_mapping)
 
    VOL = Vector{NTuple{d,LinearMap}}(undef,N_e)
    FAC = Vector{LinearMap}(undef,N_e)
    V_ar = Vector{LinearMap}(undef,N_e)
    R_ar = Vector{LinearMap}(undef,N_e)
    n_f = Array{Float64,3}(undef, d, N_f, N_e)

    @inbounds Threads.@threads for k in 1:N_e
        M⁻¹ = mass_matrix_inverse(mass_solver, k)
        VOL[k] = Tuple(make_operator(M⁻¹ * Matrix(V' * 
            (sum(D[m]' * Diagonal(0.5 * W * Λ_q[:,m,n,k]) -
            Diagonal(0.5 * W * Λ_q[:,m,n,k]) * D[m] for m in 1:d) +
            R' * Diagonal(0.5 * B * nJf[n,:,k]) * R)), alg) for n in 1:d)
        FAC[k] = make_operator(-M⁻¹ * 
            Matrix(V' * R' * Diagonal(B * J_f[:,k])), alg)
        V_ar[k] = make_operator(reference_approximation.V, alg)
        R_ar[k] = make_operator(reference_approximation.R, alg)
        @inbounds for m in 1:d
            n_f[m,:,k] = nJf[m,:,k] ./ J_f[:,k]
        end
    end

    return PhysicalOperators(VOL, FAC, V_ar, R_ar, n_f)
end