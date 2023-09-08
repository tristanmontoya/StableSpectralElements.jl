@inline @views function auxiliary_variable!(dudt::AbstractArray{Float64,3},
    solver::Solver{<:AbstractConservationLaw{d,SecondOrder},<:PhysicalOperators,
    <:AbstractMassMatrixSolver,<:StandardForm}, k::Int) where {d}
    
    (; conservation_law, operators, connectivity, form) = solver
    (; V, R, VOL, FAC, n_f) = operators
    (; viscous_numerical_flux) = form
    (; u_q, u_f, u_n, temp, CI, q_q, q_f) = solver.preallocated_arrays

    numerical_flux!(u_n[:,:,:,k], conservation_law, viscous_numerical_flux, 
        u_f[:,k,:], u_f[CI[connectivity[:,k]],:], n_f[:,:,k])
        
    @inbounds for m in 1:d
        fill!(dudt[:,:,k],0.0)
        mul!(temp[:,:,k], VOL[k][m], u_q[:,:,k])
        dudt[:,:,k] .-= temp[:,:,k]

        mul!(temp[:,:,k], FAC[k], u_n[:,:,m,k])
        dudt[:,:,k] .-= temp[:,:,k]

        mul!(q_q[:,:,m,k], V, dudt[:,:,k])
        mul!(q_f[:,k,:,m], R, q_q[:,:,m,k])
    end
end

@inline @views function time_derivative!(dudt::AbstractArray{Float64,3},
    solver::Solver{<:AbstractConservationLaw{d,SecondOrder},<:PhysicalOperators,
    <:AbstractMassMatrixSolver,<:StandardForm}, k::Int) where {d}
    
    (; conservation_law, operators, connectivity, form) = solver
    (; VOL, FAC, n_f) = operators
    (; inviscid_numerical_flux, viscous_numerical_flux) = form
    (; f_q, f_f, u_q, u_f, temp, CI, q_q, q_f) = solver.preallocated_arrays

    physical_flux!(f_q[:,:,:,k], conservation_law, u_q[:,:,k], q_q[:,:,:,k])

    numerical_flux!(f_f[:,:,k],conservation_law, inviscid_numerical_flux,
            u_f[:,k,:], u_f[CI[connectivity[:,k]],:], n_f[:,:,k])
    
    numerical_flux!(f_f[:,:,k], conservation_law, viscous_numerical_flux, 
            u_f[:,k,:], u_f[CI[connectivity[:,k]],:], q_f[:,k,:,:], 
            q_f[CI[connectivity[:,k]],:,:], n_f[:,:,k])

    fill!(dudt[:,:,k],0.0)
    @inbounds for m in 1:d
        mul!(temp[:,:,k], VOL[k][m], f_q[:,:,m,k])
        dudt[:,:,k] .+= temp[:,:,k]
    end

    mul!(temp[:,:,k], FAC[k], f_f[:,:,k])
    dudt[:,:,k] .+= temp[:,:,k]
end