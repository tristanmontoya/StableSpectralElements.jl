# determines the sparsity of the D, R matrices 
import LinearAlgebra
using StableSpectralElements
using SparseArrays
function compute_sparsity()
    D_vec = zeros(10,1)
    R_vec = zeros(10,1)

    for p in 2:6 
        reference_approximation = ReferenceApproximation(NodalTPSSOptimal(p), Tri(), mapping_degree=p)

        D = reference_approximation.D[1]
        R = reference_approximation.R

        D_sparse = sparse(D)
        R_sparse = sparse(Matrix(R))

        nnzD = nnz(D_sparse)
        nnzR = nnz(R_sparse)

        D_vec[p] = 1 - nnzD / length(D)
        R_vec[p] = 1 - nnzR / length(R)
    end

    return D_vec, R_vec
end

(D_vec,R_vec)=compute_sparsity()
println("Sparsity of D: ", D_vec)
println("Sparsity of R: ", R_vec)

