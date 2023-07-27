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

# specialized for no entropy projection 
@inline function get_nodal_values!(
    ::AbstractMassMatrixSolver,
    ::AbstractConservationLaw,
    u_q::AbstractMatrix, 
    u_f::AbstractMatrix, 
    ::AbstractMatrix,
    ::AbstractMatrix,
    V::LinearMap, R::LinearMap, ::Diagonal,
    u::AbstractMatrix, ::Int,
    ::Val{false})
    mul!(u_q, V, u)
    mul!(u_f, R, u_q)
end

# specialized for nodal schemes (not necessarily diagonal-E)
# this is really an "entropy extrapolation" and not "projection"
@inline function get_nodal_values!(
    ::AbstractMassMatrixSolver,
    conservation_law::AbstractConservationLaw,
    u_q::AbstractMatrix, 
    u_f::AbstractMatrix, 
    w_q::AbstractMatrix,
    w_f::AbstractMatrix,
    V::UniformScalingMap, R::LinearMap, WJ::Diagonal,
    u::AbstractMatrix, ::Int,
    ::Val{true})
    
    mul!(u_q, V, u)
    for i in axes(u, 1)
        w_q[i,:] .= conservative_to_entropy(conservation_law,u_q[i,:])
    end
    mul!(w_f, R, w_q)
    for i in axes(u_f, 1)
        u_f[i,:] .= entropy_to_conservative(conservation_law,w_f[i,:])
    end
end

# general (i.e. suitable for modal) approach
# uses a full entropy projection
@inline function get_nodal_values!(
    mass_solver::AbstractMassMatrixSolver,
    conservation_law::AbstractConservationLaw,
    u_q::AbstractMatrix, 
    u_f::AbstractMatrix, 
    w_q::AbstractMatrix,
    w_f::AbstractMatrix,
    V::LinearMap, R::LinearMap, WJ::Diagonal,
    u::AbstractMatrix, k::Int,
    ::Val{true})
    
    # evaluate entropy variables in terms of nodal conservative variables
    mul!(u_q, V, u)
    for i in axes(u_q, 1)
        w_q[i,:] .= conservative_to_entropy(conservation_law,u_q[i,:])
    end

    # project entropy variables and store modal coeffs in w
    w = similar(u)
    lmul!(WJ, w_q)
    mul!(w, V', w_q)
    mass_matrix_solve!(mass_solver, k, w, w_q)

    # get nodal values of projected entropy variables
    mul!(w_q, V, w)
    mul!(w_f, R, w_q)

    # convert back to conservative variables
    for i in axes(u_q, 1)
        u_q[i,:] .= entropy_to_conservative(conservation_law, w_q[i,:])
    end
    for i in axes(u_f, 1)
        u_f[i,:] .= entropy_to_conservative(conservation_law, w_f[i,:])
    end
end

@timeit "du/dt" function rhs!(
    dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:FluxDifferencingForm, FirstOrder, FluxDifferencingOperators{d}, N_p,N_q,N_f,N_c,N_e},
    t::Float64) where {d,N_p,N_q,N_f,N_c,N_e}

    (; conservation_law, connectivity, form, mass_solver) = solver
    (; inviscid_numerical_flux, two_point_flux, entropy_projection) = form
    (; f_f, u_q, r_q, u_f, CI) = solver.preallocated_arrays
    (; S, V, R, WJ, Λ_q, BJf, n_f) = solver.operators

    @views @timeit "reconstruct nodal solution" Threads.@threads for k in 1:N_e
        get_nodal_values!(mass_solver, conservation_law, u_q[:,:,k], 
            u_f[:,k,:], r_q[:,:,k], f_f[:,:,k], V, R, WJ[k], u[:,:,k], 
            k, Val(entropy_projection))
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
        mass_matrix_solve!(mass_solver, k, dudt[:,:,k], u_q[:,:,k])
    end
    return dudt
end