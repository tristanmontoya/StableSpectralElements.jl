function mass_matrix_solve(M::AbstractMatrix{Float64}, 
    rhs::AbstractMatrix{Float64})
    return M \ rhs
end