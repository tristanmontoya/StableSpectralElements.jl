"""
Evaluate semi-discrete residual for a second-order problem
"""
@timeit "rhs" function rhs!(
    dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:StandardForm, SecondOrder,PhysicalOperators{d}},
    t::Float64) where {d}

    @unpack conservation_law, operators, connectivity, form, N_e = solver
    @unpack inviscid_numerical_flux, viscous_numerical_flux = form
    @unpack source_term, N_c = conservation_law
    @unpack f_q, f_f, f_n, u_q, r_q, u_f, temp, CI, u_n, q_q, q_f = solver.preallocated_arrays

    Threads.@threads for k in 1:N_e
        mul!(view(u_q, :,:,k), operators.V[k], u[:,:,k])
        mul!(view(u_f,:,k,:), operators.R[k], u_q[:,:,k])
    end

    #auxiliary variable
    Threads.@threads for k in 1:N_e
        numerical_flux!(view(u_n,:,:,:,k), conservation_law,
            viscous_numerical_flux, u_f[:,k,:], u_f[CI[connectivity[:,k]],:], 
            operators.n_f[k])
        
        @inbounds for m in 1:d
            fill!(view(dudt,:,:,k),0.0)

            mul!(view(temp,:,:,k),operators.VOL[k][m],u_q[:,:,k])
            dudt[:,:,k] .-= temp[:,:,k] 

            mul!(view(temp,:,:,k),operators.FAC[k],u_n[:,:,m,k])
            dudt[:,:,k] .-= temp[:,:,k]

            mul!(view(q_q,:,:,m,k), operators.V[k], dudt[:,:,k])
            mul!(view(q_f,:,k,:,d), operators.R[k], q_q[:,:,m,k])
        end
    end

    # time derivative
    Threads.@threads for k in 1:N_e
        physical_flux!(view(f_q,:,:,:,k),conservation_law, 
            u_q[:,:,k], q_q[:,:,:,k])

        f_f[:,:,k] .= numerical_flux(conservation_law, inviscid_numerical_flux,
            u_f[:,k,:], u_f[CI[connectivity[:,k]],:], operators.n_f[k])

        f_f[:,:,k] .+= numerical_flux(conservation_law, viscous_numerical_flux, 
            u_f[:,k,:], u_f[CI[connectivity[:,k]],:], q_f[:,k,:,:], 
            q_f[CI[connectivity[:,k]],:,:], operators.n_f[k])

        fill!(view(dudt,:,:,k),0.0)
        @inbounds for m in 1:d
            mul!(view(temp,:,:,k),operators.VOL[k][m],f_q[:,:,m,k])
            dudt[:,:,k] .+= temp[:,:,k]
        end

        mul!(view(temp,:,:,k), operators.FAC[k], f_f[:,:,k])
        dudt[:,:,k] .+= temp[:,:,k]
    end

    return dudt
end