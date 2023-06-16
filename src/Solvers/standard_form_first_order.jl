"""
Evaluate semi-discrete residual for a first-order problem
"""
@timeit "du/dt" function rhs!(
    dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:StandardForm, FirstOrder, PhysicalOperators{d}},
    t::Float64) where {d}

    (; conservation_law, operators, connectivity, form, N_e) = solver
    (; inviscid_numerical_flux) = form
    (; f_q, f_f, u_q, u_f, temp, CI) = solver.preallocated_arrays
    
    @timeit "reconstruct nodal solution" Threads.@threads for k in 1:N_e
        mul!(view(u_q, :,:,k), operators.V[k], u[:,:,k])
        mul!(view(u_f,:,k,:), operators.R[k], u_q[:,:,k])
    end

    @timeit "eval residual" Threads.@threads for k in 1:N_e
        physical_flux!(view(f_q,:,:,:,k),conservation_law, u_q[:,:,k])

        f_f[:,:,k] .= numerical_flux(conservation_law, inviscid_numerical_flux,
            u_f[:,k,:], u_f[CI[connectivity[:,k]],:], operators.n_f[k])

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

"""
Evaluate semi-discrete residual for a first-order problem
"""
@timeit "du/dt" function rhs!(
    dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:StandardForm{SkewSymmetricMapping}, FirstOrder, ReferenceOperators{d}},
    t::Float64) where {d}

    (; conservation_law, connectivity, form, N_e) = solver
    (; inviscid_numerical_flux) = form
    (; f_q, f_f, f_n, u_q, r_q, u_f, temp, CI) = solver.preallocated_arrays
    (; D, V, R, halfWΛ, halfN, BJf, n_f) = solver.operators
    
    @timeit "reconstruct nodal solution" Threads.@threads for k in 1:N_e
        mul!(view(u_q, :,:,k), V, u[:,:,k])
        mul!(view(u_f,:,k,:), R, u_q[:,:,k])
    end

    @timeit "eval residual" Threads.@threads for k in 1:N_e
        physical_flux!(view(f_q,:,:,:,k),conservation_law, u_q[:,:,k])

        f_f[:,:,k] .= numerical_flux(conservation_law, inviscid_numerical_flux,
            u_f[:,k,:], u_f[CI[connectivity[:,k]],:], n_f[k])

        fill!(view(r_q,:,:,k),0.0)
        for n in 1:d
            # apply volume operators
            for m in 1:d
                mul!(view(temp,:,:,k),halfWΛ[m,n,k],f_q[:,:,n,k])
                mul!(view(u_q,:,:,k),D[m]',temp[:,:,k])
                r_q[:,:,k] .+= u_q[:,:,k] 
                mul!(view(u_q,:,:,k),D[m],f_q[:,:,n,k])
                lmul!(halfWΛ[m,n,k],view(u_q,:,:,k))
                r_q[:,:,k] .-= u_q[:,:,k] 
            end

            # difference facet flux
            mul!(view(f_n,:,:,k), R, f_q[:,:,n,k])
            lmul!(halfN[n,k], view(f_n,:,:,k))
            f_f[:,:,k] .-= f_n[:,:,k]
        end

        # apply facet operators
        lmul!(BJf[k], view(f_f,:,:,k))
        mul!(view(u_q,:,:,k), R', view(f_f,:,:,k))
        r_q[:,:,k] .-= u_q[:,:,k]

        # solve for time derivative
        mul!(view(dudt,:,:,k), V', r_q[:,:,k])
        mass_matrix_solve!(solver.mass_solver, 
            k, view(dudt,:,:,k), view(u_q,:,:,k))
    end
    return dudt
end