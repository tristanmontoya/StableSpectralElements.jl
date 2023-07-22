"""
Evaluate semi-discrete residual for a first-order problem
"""
@timeit "du/dt" function rhs!(
    dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:StandardForm, FirstOrder, PhysicalOperators{d}, N_p,N_q,N_f,N_c,N_e},
    t::Float64) where {d,N_p,N_q,N_f,N_c,N_e}

    (; conservation_law, operators, connectivity, form) = solver
    (; inviscid_numerical_flux) = form
    (; f_q, f_f, u_q, u_f, temp, CI) = solver.preallocated_arrays

    @views @timeit "reconstruct nodal solution" Threads.@threads for k in 1:N_e
        mul!(u_q[:,:,k], operators.V[k], u[:,:,k])
        mul!(u_f[:,k,:], operators.R[k], u_q[:,:,k])
    end

    @views @timeit "eval residual" Threads.@threads for k in 1:N_e

        physical_flux!(f_q[:,:,:,k], conservation_law, u_q[:,:,k])

        numerical_flux!(f_f[:,:,k],
            conservation_law, inviscid_numerical_flux, u_f[:,k,:], 
            u_f[CI[connectivity[:,k]],:], operators.n_f[k])
        
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

@timeit "du/dt" function rhs!(
    dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:StandardForm{SkewSymmetricMapping}, FirstOrder, ReferenceOperators{d},N_p,N_q,N_f,N_c,N_e},
    t::Float64) where {d,N_p,N_q,N_f,N_c,N_e}

    (; conservation_law, connectivity, form) = solver
    (; inviscid_numerical_flux) = form
    (; f_q, f_f, f_n, u_q, r_q, u_f, temp, CI) = solver.preallocated_arrays
    (; D, V, R, halfWΛ, halfN, BJf, n_f) = solver.operators

    @views @timeit "reconstruct nodal solution" Threads.@threads for k in 1:N_e
        mul!(u_q[:,:,k], V, u[:,:,k])
        mul!(u_f[:,k,:], R, u_q[:,:,k])
    end

    @views @timeit "eval residual" Threads.@threads for k in 1:N_e
        physical_flux!(f_q[:,:,:,k], conservation_law, u_q[:,:,k])

        numerical_flux!(f_f[:,:,k], conservation_law, inviscid_numerical_flux,
            u_f[:,k,:], u_f[CI[connectivity[:,k]],:], n_f[k])

        fill!(r_q[:,:,k],0.0)
        @inbounds for n in 1:d
            # apply volume operators
            @inbounds for m in 1:d
                mul!(temp[:,:,k],halfWΛ[m,n,k],f_q[:,:,n,k])
                mul!(u_q[:,:,k],D[m]',temp[:,:,k])
                r_q[:,:,k] .+= u_q[:,:,k] 
                mul!(u_q[:,:,k],D[m],f_q[:,:,n,k])
                lmul!(halfWΛ[m,n,k],u_q[:,:,k])
                r_q[:,:,k] .-= u_q[:,:,k] 
            end

            # difference facet flux
            mul!(f_n[:,:,k], R, f_q[:,:,n,k])
            lmul!(halfN[n,k], f_n[:,:,k])
            f_f[:,:,k] .-= f_n[:,:,k]
        end

        # apply facet operators
        lmul!(BJf[k], f_f[:,:,k])
        mul!(u_q[:,:,k], R', f_f[:,:,k])
        r_q[:,:,k] .-= u_q[:,:,k]

        # solve for time derivative
        mul!(dudt[:,:,k], V', r_q[:,:,k])
        mass_matrix_solve!(solver.mass_solver, k, dudt[:,:,k], u_q[:,:,k])
    end
    return dudt
end