"""
Evaluate semi-discrete residual for a first-order problem
"""
@timeit "du/dt" function rhs!(
    dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:StandardForm, FirstOrder}, t::Float64) where {d}

    @unpack conservation_law, operators, connectivity, form, N_e = solver
    @unpack inviscid_numerical_flux = form
    @unpack source_term, N_c = conservation_law
    @unpack f_q, f_f, f_n, u_q, r_q, u_f, CI = solver.preallocated_arrays
    
    @timeit "reconstruct nodal solution" Threads.@threads for k in 1:N_e
        mul!(view(u_q, :,:,k), operators[k].V, u[:,:,k])
        mul!(view(u_f,:,k,:), operators[k].R, u_q[:,:,k])
    end

    @timeit "eval residual" Threads.@threads for k in 1:N_e
        println("element ", k, "on thread ", Threads.threadid())
        physical_flux!(view(f_q,:,:,:,k),conservation_law, u_q[:,:,k])

        f_f[:,:,k] .= numerical_flux(conservation_law, inviscid_numerical_flux,
            u_f[:,k,:], u_f[CI[connectivity[:,k]],:], operators[k].n_f)

        diff_with_extrap_flux!(view(f_f,:,:,k), view(f_n,:,:,k), 
            operators[k].NTR, f_q[:,:,:,k])

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