# Express the semi-discrete residual operator as a LinearMap
struct LinearResidual{SolverType} <: LinearMap{Float64}
    solver::SolverType
end

function Base.size(L::LinearResidual)
    (N_p, N_c, N_e) = size(L.solver)
    return (N_p * N_c * N_e, N_p * N_c * N_e)
end

function LinearAlgebra.mul!(y::AbstractVector, L::LinearResidual, x::AbstractVector)
    (N_p, N_c, N_e) = size(L.solver)
    u = reshape(x, (N_p, N_c, N_e))
    dudt = similar(u)
    semi_discrete_residual!(dudt, u, L.solver, 0.0)
    y[:] = vec(dudt)
    return y
end
