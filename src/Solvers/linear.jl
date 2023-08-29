""" Semi-discrete residual operator as a LinearMap"""
struct LinearResidual <: LinearMap{Float64}
    solver::Solver
end

Base.size(L::LinearResidual) = (L.solver.N_p*L.solver.N_c*L.solver.N_e, 
    L.solver.N_p*L.solver.N_c*L.solver.N_e)


function LinearAlgebra.mul!(y::AbstractVector, 
        L::LinearResidual,
        x::AbstractVector)
    u = reshape(x,(L.solver.N_p,L.solver.N_c,L.solver.N_e))
    dudt = Array{Float64}(undef,L.solver.N_p,L.solver.N_c,L.solver.N_e)
    semi_discrete_residual!(dudt,u,L.solver,0.0)
    y[:] = vec(dudt)
    return y
end