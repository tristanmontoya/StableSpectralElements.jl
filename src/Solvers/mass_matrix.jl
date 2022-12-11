
abstract type AbstractMassMatrixSolver end
struct CholeskySolver <: AbstractMassMatrixSolver end
struct WeightAdjustedSolver <: AbstractMassMatrixSolver end

function mass_matrix(V::LinearMap, W::Diagonal, J::Diagonal, ::CholeskySolver)
    return cholesky(Symmetric(Matrix(V'*W*J*V)))
end

function mass_matrix(V::LinearMap, W::Diagonal, J::Diagonal,
    ::WeightAdjustedSolver)
    return WeightAdjustedMap(V, W, inv(J))
end