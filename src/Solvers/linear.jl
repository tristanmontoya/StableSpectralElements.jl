""" Semi-discrete residual operator as a LinearMap"""
struct LinearResidual <: LinearMap{Float64}
    solver::Solver
    N_p::Int
    N_c::Int
    N_e::Int
end

Base.size(L::LinearResidual) = (L.N_p*L.N_c*L.N_e, L.N_p*L.N_c*L.N_e)

function LinearResidual(
    solver::Solver{ResidualForm,DiscretizationOperators,d}) where {ResidualForm,DiscretizationOperators,d}

    return LinearResidual(solver,solver.operators[1].N_p,
        solver.conservation_law.N_c,length(solver.operators))
end

function LinearAlgebra.mul!(y::AbstractVector, 
        L::LinearResidual,
        x::AbstractVector)
    u = reshape(x,(L.N_p,L.N_c,L.N_e))
    dudt = Array{Float64}(undef,L.N_p,L.N_c,L.N_e)
    rhs!(dudt,u,L.solver,0.0)
    y[:] = vec(dudt)
    return y
end