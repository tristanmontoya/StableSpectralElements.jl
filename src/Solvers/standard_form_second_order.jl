"""
Evaluate semi-discrete residual for a second-order problem
"""
@timeit "rhs" function rhs!(
    dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d,<:StandardForm,SecondOrder,PhysicalOperators{d},N_p,N_q,N_f,N_c,N_e},
    t::Float64) where {d,N_p,N_q,N_f,N_c,N_e}

    (; conservation_law, operators, connectivity, form) = solver
    (; inviscid_numerical_flux, viscous_numerical_flux) = form
    (; f_q, f_f, u_q, u_f, temp, CI, u_n, q_q, q_f) = solver.preallocated_arrays

    @views Threads.@threads for k in 1:N_e
        mul!(u_q[:,:,k], operators.V[k], u[:,:,k])
        mul!(u_f[:,k,:], operators.R[k], u_q[:,:,k])
    end

    #auxiliary variable
    @views Threads.@threads for k in 1:N_e
        numerical_flux!(u_n[:,:,:,k], conservation_law,
            viscous_numerical_flux, u_f[:,k,:], u_f[CI[connectivity[:,k]],:], 
             operators.n_f[k])
        
        fill!(dudt[:,:,k],0.0)
        @inbounds for m in 1:d
            mul!(temp[:,:,k],operators.VOL[k][m],u_q[:,:,k])
            dudt[:,:,k] .-= temp[:,:,k]

            mul!(temp[:,:,k],operators.FAC[k],u_n[:,:,m,k])
            dudt[:,:,k] .-= temp[:,:,k]

            mul!(q_q[:,:,m,k], operators.V[k], dudt[:,:,k])
            mul!(q_f[:,k,:,m], operators.R[k], q_q[:,:,m,k])
        end
    end

    # time derivative
   @views Threads.@threads for k in 1:N_e
        physical_flux!(f_q[:,:,:,k], conservation_law, u_q[:,:,k], q_q[:,:,:,k])

        f_f[:,:,k] .= numerical_flux(conservation_law, inviscid_numerical_flux,
                u_f[:,k,:], u_f[CI[connectivity[:,k]],:], operators.n_f[k]) .+ 
            numerical_flux(conservation_law, viscous_numerical_flux, 
                u_f[:,k,:], u_f[CI[connectivity[:,k]],:], q_f[:,k,:,:], 
                q_f[CI[connectivity[:,k]],:,:], operators.n_f[k])

        fill!(dudt[:,:,k],0.0)
        @inbounds for m in 1:d
            mul!(temp[:,:,k],operators.VOL[k][m],f_q[:,:,m,k])
            dudt[:,:,k] .+= temp[:,:,k]
        end

        mul!(temp[:,:,k], operators.FAC[k], f_f[:,:,k])
        dudt[:,:,k] .+= temp[:,:,k]
    end

    return dudt
end