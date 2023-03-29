
abstract type AbstractMassMatrixSolver end
struct CholeskySolver <: AbstractMassMatrixSolver end
struct WeightAdjustedSolver <: AbstractMassMatrixSolver end
struct PreInvert <: AbstractMassMatrixSolver end

function mass_matrix(V::LinearMap, W::Diagonal, J::Diagonal, 
    ::CholeskySolver)
    M = Matrix(V'*W*J*V)
    return cholesky(Symmetric(M))
end

function mass_matrix(V::LinearMap, W::Diagonal, J::Diagonal,
    ::WeightAdjustedSolver)
    return WeightAdjustedMap(V, W, inv(J))
end

function mass_matrix(V::UniformScalingMap,  W::Diagonal, 
    J::Diagonal, ::CholeskySolver)
    @assert V.λ == true
    return factorize(W*J)
end

function mass_matrix(V::UniformScalingMap,  W::Diagonal, 
    J::Diagonal, ::WeightAdjustedSolver)
    @assert V.λ == true
    return factorize(W*J)
end

function mass_matrix(V::LinearMap, ::Any, ::Any, ::PreInvert)
    return LinearMap(I,size(V,2))
end