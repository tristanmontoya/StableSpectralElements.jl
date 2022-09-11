""" Semi-discrete residual operator as a LinearMap"""
struct LinearResidual <: LinearMap{Float64}
    solver::Solver
    N_p::Int
    N_eq::Int
    N_el::Int
end

Base.size(L::LinearResidual) = (L.N_p*L.N_eq*L.N_el, L.N_p*L.N_eq*L.N_el)

function LinearResidual(
    solver::Solver{ResidualForm,PhysicalOperators,d}) where {ResidualForm,PhysicalOperators,d}

    return LinearResidual(solver,size(solver.operators[1].VOL[1],1),num_equations(solver.conservation_law),length(solver.operators))
end

function LinearAlgebra.mul!(y::AbstractVector, 
        L::LinearResidual,
        x::AbstractVector)
    u = reshape(x,(L.N_p,L.N_eq,L.N_el))
    dudt = Array{Float64}(undef,L.N_p,L.N_eq,L.N_el)
    rhs!(dudt,u,L.solver,0.0,print=false)
    y[:] = vec(dudt)
    return y
end