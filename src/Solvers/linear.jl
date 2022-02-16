struct LinearResidual{T} <: LinearMap{T}
    solver::Solver
    N_p::Int
    N_eq::Int
    N_el::Int
end

Base.size(L::LinearResidual) = (L.N_p*L.N_eq*L.N_el, L.N_p*L.N_eq*L.N_el)

function LinearResidual(
    solver::Solver{ResidualForm,PhysicalOperators,d,N_eq}) where {ResidualForm,PhysicalOperators,d,N_eq}

    return LinearResidual{Float64}(solver,size(solver.operators[1].VOL[1],1),N_eq,length(solver.operators))
end

function LinearAlgebra.mul!(y::AbstractVector, 
        L::LinearResidual,
        x::AbstractVector)
    #println(x)
    u = reshape(x,(L.N_p,L.N_eq,L.N_el))
    dudt = Array{Float64}(undef,L.N_p,L.N_eq,L.N_el)
    rhs!(dudt,u,L.solver,0.0,print=false)
    y[:] = vec(dudt)
    return y
end