@inline @views function flux_difference!(
    r_q::AbstractMatrix{Float64}, # N_q x N_c 
    S::NTuple{d,<:LinearMap{Float64}}, # N_q x N_q
    conservation_law::AbstractConservationLaw{d},
    two_point_flux::AbstractTwoPointFlux,
    Λ_q::AbstractArray{Float64,3}, # N_q x d x d
    u_q::AbstractMatrix{Float64}) where {d}

    fill!(r_q, 0.0)
    for i in axes(u_q,1)
        for j in (i+1):size(u_q,1)
            Λ_ij = SMatrix{d,d}(0.5*(Λ_q[i,:,:] .+ Λ_q[j,:,:]))
            F_ij = compute_two_point_flux(conservation_law,
                two_point_flux, u_q[i,:],u_q[j,:])
            @inbounds for m in 1:d
                diff_ij = S[m].lmap[i,j] * sum(Λ_ij[m,n] * F_ij[:,n] 
                    for n in 1:d)
                r_q[i,:] .-= diff_ij
                r_q[j,:] .+= diff_ij
            end
        end
    end

end

@timeit "du/dt" function rhs!(
    dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:FluxDifferencingForm, FirstOrder, FluxDifferencingOperators{d}, N_p,N_q,N_f,N_c,N_e},
    t::Float64) where {d,N_p,N_q,N_f,N_c,N_e}

    (; conservation_law, connectivity, form) = solver
    (; inviscid_numerical_flux, two_point_flux) = form
    (; f_f, u_q, r_q, u_f, CI) = solver.preallocated_arrays
    (; S, V, R, Λ_q, BJf, n_f) = solver.operators

    @views @timeit "reconstruct nodal solution" Threads.@threads for k in 1:N_e
        mul!(u_q[:,:,k], V, u[:,:,k])
        mul!(u_f[:,k,:], R, u_q[:,:,k])
    end

    @views @timeit "eval residual" Threads.@threads for k in 1:N_e
        numerical_flux!(f_f[:,:,k],
            conservation_law, inviscid_numerical_flux, u_f[:,k,:], 
            u_f[CI[connectivity[:,k]],:], n_f[k], two_point_flux)

        flux_difference!(r_q[:,:,k], S, conservation_law, 
            two_point_flux, Λ_q[:,:,:,k], u_q[:,:,k])

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