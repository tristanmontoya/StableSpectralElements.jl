
abstract type AbstractMassMatrixSolver end
struct CholeskySolver <: AbstractMassMatrixSolver end
struct WeightAdjustedSolver <: AbstractMassMatrixSolver end

function mass_matrix(V::LinearMap, W::Diagonal, J::Diagonal, 
    ::CholeskySolver)
    M = Matrix(V'*W*J*V)
    return cholesky(Symmetric(M))
end

function mass_matrix(V::LinearMap, W::Diagonal, J::Diagonal,
    ::WeightAdjustedSolver)
    return WeightAdjustedMap(V, W, inv(J))
end

function mass_matrix(::UniformScalingMap,  W::Diagonal, 
    J::Diagonal, ::CholeskySolver)
    return factorize(W*J)
end

function mass_matrix(::UniformScalingMap,  W::Diagonal, 
    J::Diagonal, ::WeightAdjustedSolver)
    return factorize(W*J)
end