struct CholeskySolver{V_type <: LinearMap} <: AbstractMassMatrixSolver
    M::Vector{Cholesky{Float64, Matrix{Float64}}}
    WJ::Vector{Diagonal{Float64, Vector{Float64}}}
    V::V_type
end

struct DiagonalSolver <: AbstractMassMatrixSolver
    WJ⁻¹::Vector{Diagonal{Float64, Vector{Float64}}}
end

struct WeightAdjustedSolver{Minv_type, V_type <: LinearMap, Vt_type <: LinearMap} <:
       AbstractMassMatrixSolver
    M⁻¹::Minv_type
    J⁻¹W::Vector{Diagonal{Float64, Vector{Float64}}}
    V::V_type
    Vᵀ::Vt_type
end

function default_mass_matrix_solver(spatial_discretization::SpatialDiscretization,
        alg::AbstractOperatorAlgorithm = DefaultOperatorAlgorithm())
    (; V, W) = spatial_discretization.reference_approximation
    (; J_q) = spatial_discretization.geometric_factors
    return WeightAdjustedSolver(J_q, V, W, alg, Val(true), Float64(0.0))
end

function CholeskySolver(J_q::Matrix{Float64}, V::UniformScalingMap, W::Diagonal)
    return DiagonalSolver(J_q, V, W)
end

function CholeskySolver(J_q::Matrix{Float64}, V::LinearMap, W::Diagonal)
    N_e = size(J_q, 2)
    WJ = Vector{Diagonal{Float64, Vector{Float64}}}(undef, N_e)
    M = Vector{Cholesky{Float64, Matrix{Float64}}}(undef, N_e)
    @inbounds for k in 1:N_e
        WJ[k] = Diagonal(W .* J_q[:, k])
        M[k] = cholesky(Symmetric(Matrix(V' * WJ[k] * V)))
    end
    return CholeskySolver(M, WJ, V)
end

function WeightAdjustedSolver(J_q::Matrix{Float64},
        V::UniformScalingMap,
        W::Diagonal,
        ::AbstractOperatorAlgorithm,
        ::Val{true},
        ::Float64)
    return DiagonalSolver(J_q, V, W)
end

function WeightAdjustedSolver(J_q::Matrix{Float64},
        V::UniformScalingMap,
        W::Diagonal,
        ::AbstractOperatorAlgorithm,
        ::Val{false},
        ::Float64)
    return DiagonalSolver(J_q, V, W)
end

function WeightAdjustedSolver(J_q::Matrix{Float64},
        V::LinearMap,
        W::Diagonal,
        operator_algorithm::AbstractOperatorAlgorithm,
        ::Val{true},
        ::Float64)
    N_e = size(J_q, 2)
    J⁻¹W = Vector{Diagonal{Float64, Vector{Float64}}}(undef, N_e)
    @inbounds for k in 1:N_e
        J⁻¹W[k] = Diagonal(W ./ J_q[:, k])
    end

    return WeightAdjustedSolver(I,
        J⁻¹W,
        make_operator(V, operator_algorithm),
        make_operator(V', operator_algorithm))
end

function WeightAdjustedSolver(J_q::Matrix{Float64},
        V::LinearMap,
        W::Diagonal,
        operator_algorithm::AbstractOperatorAlgorithm,
        ::Val{false},
        tol::Float64)
    N_e = size(J_q, 2)
    VDM = Matrix(V)
    M = VDM' * W * VDM
    M_diag = diag(M)
    if maximum(abs.(M - diagm(M_diag))) < tol
        if maximum(abs.(M_diag .- 1.0)) < tol
            M⁻¹ = I
        else
            M⁻¹ = inv(Diagonal(M_diag))
        end
    else
        M⁻¹ = inv(M)
    end

    J⁻¹W = Vector{Diagonal{Float64, Vector{Float64}}}(undef, N_e)
    for k in 1:N_e
        J⁻¹W[k] = Diagonal(W ./ J_q[:, k])
    end

    return WeightAdjustedSolver(M⁻¹,
        J⁻¹W,
        make_operator(V, operator_algorithm),
        make_operator(V', operator_algorithm))
end

function DiagonalSolver(J_q::Matrix{Float64}, ::UniformScalingMap, W::Diagonal)
    N_e = size(J_q, 2)
    WJ⁻¹ = Vector{Diagonal{Float64, Vector{Float64}}}(undef, N_e)
    @inbounds for k in 1:N_e
        WJ⁻¹[k] = inv(Diagonal(W .* J_q[:, k]))
    end
    return DiagonalSolver(WJ⁻¹)
end

function CholeskySolver(spatial_discretization::SpatialDiscretization)
    (; V, W) = spatial_discretization.reference_approximation
    (; J_q) = spatial_discretization.geometric_factors

    return CholeskySolver(J_q, V, W)
end

function DiagonalSolver(spatial_discretization::SpatialDiscretization)
    (; V, W) = spatial_discretization.reference_approximation
    (; J_q) = spatial_discretization.geometric_factors
    return DiagonalSolver(J_q, V, W)
end

function WeightAdjustedSolver(spatial_discretization::SpatialDiscretization,
        operator_algorithm = DefaultOperatorAlgorithm();
        assume_orthonormal::Bool = true,
        tol = 1.0e-13)
    (; V, W) = spatial_discretization.reference_approximation
    (; J_q) = spatial_discretization.geometric_factors

    return WeightAdjustedSolver(J_q, V, W, operator_algorithm, Val(assume_orthonormal), tol)
end

@inline function mass_matrix(mass_solver::CholeskySolver, k::Int)
    (; WJ, V) = mass_solver
    return Matrix(V' * WJ[k] * V)
end

@inline function mass_matrix(mass_solver::DiagonalSolver, k::Int)
    (; WJ⁻¹) = mass_solver
    return inv(WJ⁻¹[k])
end

@inline function mass_matrix(mass_solver::WeightAdjustedSolver, k::Int)
    (; M⁻¹, J⁻¹W, V, Vᵀ) = mass_solver
    return inv(Matrix(M⁻¹ * Vᵀ * J⁻¹W[k] * V * M⁻¹))
end

@inline function mass_matrix_inverse(mass_solver::CholeskySolver, k::Int)
    (; WJ, V) = mass_solver
    return inv(Matrix(V' * WJ[k] * V))
end

@inline function mass_matrix_inverse(mass_solver::DiagonalSolver, k::Int)
    return mass_solver.WJ⁻¹[k]
end

@inline function mass_matrix_inverse(mass_solver::WeightAdjustedSolver, k::Int)
    (; M⁻¹, Vᵀ, J⁻¹W, V) = mass_solver
    return Matrix(M⁻¹ * Vᵀ * J⁻¹W[k] * V * M⁻¹)
end

@inline function mass_matrix_solve!(mass_solver::CholeskySolver,
        k::Int,
        rhs::AbstractMatrix,
        ::AbstractMatrix)
    ldiv!(mass_solver.M[k], rhs)
    return rhs
end

@inline function mass_matrix_solve!(mass_solver::DiagonalSolver,
        k::Int,
        rhs::AbstractMatrix,
        ::AbstractMatrix)
    lmul!(mass_solver.WJ⁻¹[k], rhs)
    return rhs
end

@inline function mass_matrix_solve!(mass_solver::WeightAdjustedSolver,
        k::Int,
        rhs::AbstractMatrix,
        temp::AbstractMatrix)
    (; M⁻¹, Vᵀ, J⁻¹W, V) = mass_solver
    lmul!(M⁻¹, rhs)
    mul!(temp, V, rhs)
    lmul!(J⁻¹W[k], temp)
    mul!(rhs, Vᵀ, temp)
    lmul!(M⁻¹, rhs)
    return rhs
end
