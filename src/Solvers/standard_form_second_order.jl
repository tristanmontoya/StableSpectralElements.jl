"""
Evaluate semi-discrete residual for a second-order problem
"""
@timeit "rhs" function rhs!(
    dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:StandardForm, SecondOrder}, t::Float64) where {d}

    @unpack conservation_law, operators, connectivity, form, N_e = solver
    @unpack inviscid_numerical_flux, viscous_numerical_flux = form
    @unpack source_term, N_c = conservation_law
    @unpack f_q, f_f, f_n, u_q, r_q, u_f, CI, u_n, q_q, q_f = solver.preallocated_arrays

    @threads_optional for k in 1:N_e
        mul!(view(u_q, :,:,k), operators[k].V, u[:,:,k])
        mul!(view(u_f,:,k,:), operators[k].R, u_q[:,:,k])
    end

    #auxiliary variable
    @threads_optional for k in 1:N_e

        numerical_flux!(view(u_n,:,:,:,k), conservation_law,
            viscous_numerical_flux, u_f[:,k,:], u_f[CI[connectivity[:,k]],:], 
            operators[k].n_f)
        
        @inbounds for m in 1:d
            fill!(view(r_q,:,:,k),0.0)

            mul!(view(f_q,:,:,1,k),operators[k].VOL[m],u_q[:,:,k])
            r_q[:,:,k] .-= f_q[:,:,1,k] # use first comp. of f_q as temp storage

            mul!(view(f_q,:,:,1,k),operators[k].FAC,u_n[:,:,m,k])
            r_q[:,:,k] .-= f_q[:,:,1,k]

            # store modal coeffs of q_m in dudt
            mul!(view(dudt,:,:,k), operators[k].V', r_q[:,:,k])
            ldiv!(operators[k].M, view(dudt,:,:,k))

            mul!(view(q_q,:,:,m,k), operators[k].V, dudt[:,:,k])
            mul!(view(q_f,:,k,:,d), operators[k].R, q_q[:,:,m,k])
        end
    end

    # time derivative
    @threads_optional for k in 1:N_e

        physical_flux!(view(f_q,:,:,:,k),conservation_law, 
            u_q[:,:,k], q_q[:,:,:,k])

        f_f[:,:,k] .= numerical_flux(conservation_law, inviscid_numerical_flux,
            u_f[:,k,:], u_f[CI[connectivity[:,k]],:], operators[k].n_f)

        f_f[:,:,k] .+= numerical_flux(conservation_law, viscous_numerical_flux, 
            u_f[:,k,:], u_f[CI[connectivity[:,k]],:], q_f[:,k,:,:], 
            q_f[CI[connectivity[:,k]],:,:], operators[k].n_f)

        fill!(view(r_q,:,:,k),0.0)
        @inbounds for m in 1:d
            mul!(view(u_q,:,:,k),operators[k].VOL[m],f_q[:,:,m,k])
            r_q[:,:,k] .+= u_q[:,:,k] # reuse u_q as temp storage
        end

        mul!(view(u_q,:,:,k), operators[k].FAC, f_f[:,:,k])
        r_q[:,:,k] .+= u_q[:,:,k] # reuse u_q as temp storage 

        mul!(view(dudt,:,:,k), operators[k].V', r_q[:,:,k])
        ldiv!(operators[k].M, view(dudt,:,:,k))
    end

    return dudt
end