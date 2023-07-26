struct CholeskySolver <: AbstractMassMatrixSolver 
    M::Vector{Cholesky}
    WJ::Vector{Diagonal}
    V::LinearMap
end

struct DiagonalSolver <: AbstractMassMatrixSolver 
    WJ⁻¹::Vector{Diagonal}
end

struct WeightAdjustedSolver <: AbstractMassMatrixSolver
    M⁻¹::Union{UniformScaling,AbstractMatrix}
    J⁻¹W::Vector{Diagonal}
    V::LinearMap
    Vᵀ::LinearMap
end

function CholeskySolver(spatial_discretization::SpatialDiscretization)
    (; V, W) = spatial_discretization.reference_approximation
    (; J_q) = spatial_discretization.geometric_factors
    (; N_e) = spatial_discretization

    if V isa UniformScalingMap
        return DiagonalSolver(spatial_discretization)
    end

    WJ = Vector{Diagonal}(undef, N_e)
    M = Vector{Cholesky}(undef, N_e)
    Threads.@threads for k in 1:N_e
        WJ[k] = Diagonal(W .* J_q[:,k])
        M[k] = cholesky(Symmetric(Matrix(V' * WJ[k] * V)))
    end
    return CholeskySolver(M, WJ, V)
end

function DiagonalSolver(spatial_discretization::SpatialDiscretization)
    (; V, W) = spatial_discretization.reference_approximation
    (; J_q) = spatial_discretization.geometric_factors
    (; N_e) = spatial_discretization

    if !(V isa UniformScalingMap)
        return CholeskySolver(spatial_discretization)
    end

    WJ⁻¹ = Vector{Diagonal}(undef, N_e)
    Threads.@threads for k in 1:N_e
        WJ⁻¹[k] = inv(Diagonal(W .* J_q[:,k]))
    end
    return DiagonalSolver(WJ⁻¹)
end

function WeightAdjustedSolver(spatial_discretization::SpatialDiscretization; 
    operator_algorithm=DefaultOperatorAlgorithm(), assume_orthonormal=false,
    tol=1.0e-13)
    
    (; V, W) = spatial_discretization.reference_approximation
    (; J_q) = spatial_discretization.geometric_factors
    (; N_e) = spatial_discretization

    if V isa UniformScalingMap
        return DiagonalSolver(spatial_discretization)
    end

    M = Matrix(V'*W*V)
    M_diag = diag(M)

    if assume_orthonormal 
        M⁻¹ = I
    elseif maximum(abs.(M - diagm(M_diag))) < tol
        if maximum(abs.(M_diag .- 1.0)) < tol
            M⁻¹ = I
        else
            M⁻¹ = inv(Diagonal(M_diag))
        end
    else
        M⁻¹ = inv(M)
    end

    J⁻¹W = Vector{Diagonal}(undef, N_e)
    Threads.@threads for k in 1:N_e
        J⁻¹W[k] = Diagonal(W ./ J_q[:,k])
    end

    return WeightAdjustedSolver(M⁻¹,J⁻¹W, make_operator(V, operator_algorithm),
        make_operator(V', operator_algorithm))
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
    (; M⁻¹,J⁻¹W, V, Vᵀ) = mass_solver
    return inv(Matrix(M⁻¹*Vᵀ*J⁻¹W[k]*V*M⁻¹))
end

@inline function mass_matrix_inverse(mass_solver::CholeskySolver, k::Int)
    (; WJ, V) = mass_solver
    return inv(Matrix(V'*WJ[k]*V))
end

@inline function mass_matrix_inverse(mass_solver::DiagonalSolver, k::Int)
    return mass_solver.WJ⁻¹[k]
end

@inline function mass_matrix_inverse(mass_solver::WeightAdjustedSolver, k::Int)
    (; M⁻¹, Vᵀ, J⁻¹W, V) = mass_solver
    return Matrix(M⁻¹*Vᵀ*J⁻¹W[k]*V*M⁻¹)
end

@inline function mass_matrix_solve!(mass_solver::CholeskySolver, k::Int,
    rhs::AbstractMatrix, ::AbstractMatrix)
    ldiv!(mass_solver.M[k], rhs)
    return rhs
end

@inline function mass_matrix_solve!(mass_solver::DiagonalSolver, k::Int,
    rhs::AbstractMatrix, ::AbstractMatrix)
    lmul!(mass_solver.WJ⁻¹[k], rhs)
    return rhs
end

@inline function mass_matrix_solve!(mass_solver::WeightAdjustedSolver, k::Int,
    rhs::AbstractMatrix, temp::AbstractMatrix)
    (; M⁻¹, Vᵀ, J⁻¹W, V) = mass_solver
    lmul!(M⁻¹, rhs)
    mul!(temp, V, rhs)
    lmul!(J⁻¹W[k], temp)
    mul!(rhs, Vᵀ, temp)
    lmul!(M⁻¹, rhs)
    return rhs
end