@inline @views function flux_difference!(
    r_q::AbstractMatrix{Float64}, # N_q x N_c 
    S::NTuple{d,<:LinearMap{Float64}}, # N_q x N_q
    conservation_law::AbstractConservationLaw{d,FirstOrder,N_c},
    two_point_flux::AbstractTwoPointFlux,
    Λ_q::AbstractArray{Float64,3}, # N_q x d x d
    u_q::AbstractMatrix{Float64}) where {d,N_c}

    fill!(r_q, 0.0)
    @inbounds for i in axes(u_q,1)
        @inbounds for j in (i+1):size(u_q,1)
            # evaluate two-point flux
            F_ij = compute_two_point_flux(conservation_law,
                two_point_flux, u_q[i,:],u_q[j,:])
            
            # apply flux-differencing operator to flux tensor
            @inbounds for e in 1:N_c
                diff_ij = 0.0
                @inbounds for m in 1:d
                    Fm_ij = 0.0
                    @inbounds for n in 1:d
                        Λ_ij = Λ_q[i,m,n] + Λ_q[j,m,n]
                        @muladd Fm_ij = Fm_ij + Λ_ij * F_ij[e,n]
                    end
                    @muladd diff_ij = diff_ij + S[m].lmap[i,j] * Fm_ij
                end
                r_q[i,e] -= diff_ij
                r_q[j,e] += diff_ij
            end
        end
    end
end

@inline function facet_correction!(
    r_q::AbstractMatrix{Float64}, # N_q x N_c 
    f_f::AbstractMatrix{Float64}, # N_f x N_c
    CORR::LinearMap, # N_f x N_q
    conservation_law::AbstractConservationLaw{d},
    two_point_flux::AbstractTwoPointFlux,
    halfnJf::AbstractMatrix{Float64},
    halfnJq::AbstractArray{Float64,3},
    u_q::AbstractMatrix{Float64},
    u_f::AbstractMatrix{Float64},
    nodes_per_face::Int,
    ::Val{false}) where {d}
end

@inline @views function facet_correction!(
    r_q::AbstractMatrix{Float64}, # N_q x N_c 
    f_f::AbstractMatrix{Float64}, # N_f x N_c
    CORR::LinearMap, # N_f x N_q
    conservation_law::AbstractConservationLaw{d,FirstOrder,N_c},
    two_point_flux::AbstractTwoPointFlux,
    halfnJf::AbstractMatrix{Float64},
    halfnJq::AbstractArray{Float64,3},
    u_q::AbstractMatrix{Float64},
    u_f::AbstractMatrix{Float64},
    nodes_per_face::Int,
    ::Val{true}) where {d, N_c}

    @inbounds for i in axes(u_q,1)
        @inbounds for j in axes(u_f,1)
            F_ij = compute_two_point_flux(conservation_law, 
                two_point_flux, u_q[i,:], u_f[j,:])
        
            f = (j-1)÷nodes_per_face + 1

            @inbounds for e in 1:N_c
                F_dot_n_ij = 0.0
                @inbounds for n in 1:d
                    nJ_ij = halfnJf[n,j] + halfnJq[n,f,i]
                    @muladd F_dot_n_ij = F_dot_n_ij + nJ_ij * F_ij[e,n]
                end
                diff_ij = CORR.lmap[i,j] * F_dot_n_ij
                r_q[i,e] -= diff_ij
                f_f[j,e] -= diff_ij
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
    w::AbstractMatrix,
    V::UniformScalingMap, R::LinearMap, WJ::Diagonal,
    u::AbstractMatrix, ::Int,
    ::Val{true})
    
    mul!(u_q, V, u)
    @inbounds for i in axes(u, 1)
        w_q[i,:] .= conservative_to_entropy(conservation_law,u_q[i,:])
    end
    mul!(w_f, R, w_q)
    @inbounds for i in axes(u_f, 1)
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
    w::AbstractMatrix,
    V::LinearMap, R::LinearMap, WJ::Diagonal,
    u::AbstractMatrix, k::Int,
    ::Val{true})
    
    # evaluate entropy variables in terms of nodal conservative variables
    mul!(u_q, V, u)
    @inbounds for i in axes(u_q, 1)
        w_q[i,:] .= conservative_to_entropy(conservation_law, u_q[i,:])
    end

    # project entropy variables and store modal coeffs in w
    lmul!(WJ, w_q)
    mul!(w, V', w_q)
    mass_matrix_solve!(mass_solver, k, w, w_q)

    # get nodal values of projected entropy variables
    mul!(w_q, V, w)
    mul!(w_f, R, w_q)

    # convert back to conservative variables
    @inbounds for i in axes(u_q, 1)
        u_q[i,:] .= entropy_to_conservative(conservation_law, w_q[i,:])
    end
    @inbounds for i in axes(u_f, 1)
        u_f[i,:] .= entropy_to_conservative(conservation_law, w_f[i,:])
    end
end

@timeit "du/dt" function rhs!(
    dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:FluxDifferencingForm, FirstOrder, FluxDifferencingOperators{d}, N_p,N_q,N_f,N_c,N_e},
    t::Float64) where {d,N_p,N_q,N_f,N_c,N_e}

    (; conservation_law, connectivity, form, mass_solver) = solver
    (; inviscid_numerical_flux, two_point_flux, 
        entropy_projection, facet_correction) = form
    (; f_f, u_q, r_q, u_f, temp, CI) = solver.preallocated_arrays
    (; S, V, R, WJ, Λ_q, BJf, CORR, halfnJq, halfnJf, n_f,
        nodes_per_face) = solver.operators
    
    # get the nodal solution using the entropy projection if specified
    @inbounds @views @timeit "get nodal vals" Threads.@threads for k in 1:N_e
        get_nodal_values!(mass_solver, conservation_law, u_q[:,:,k], 
            u_f[:,k,:], r_q[:,:,k], f_f[:,:,k], temp[:,:,k], 
            V, R, WJ[k], u[:,:,k], k, Val(entropy_projection))
    end

    # compute the local residual
    @inbounds @views @timeit "eval residual" Threads.@threads for k in 1:N_e

        # evaluate interface numerical flux
        numerical_flux!(f_f[:,:,k], conservation_law, inviscid_numerical_flux, 
            u_f[:,k,:], u_f[CI[connectivity[:,k]],:], n_f[k], two_point_flux)
        
        # scale numerical flux by quadrature weights
        lmul!(BJf[k], f_f[:,:,k])

        # volume flux differencing term
        flux_difference!(r_q[:,:,k], S, conservation_law, 
            two_point_flux, Λ_q[:,:,:,k], u_q[:,:,k])

        # apply facet correction term (for operators w/o boundary nodes)
        facet_correction!(r_q[:,:,k], f_f[:,:,k], CORR, conservation_law,
            two_point_flux, halfnJf[:,:,k], halfnJq[:,:,:,k], 
            u_q[:,:,k], u_f[:,k,:], nodes_per_face, Val(facet_correction))

        # apply facet operators
        mul!(u_q[:,:,k], R', f_f[:,:,k])
        r_q[:,:,k] .-= u_q[:,:,k]

        # solve for time derivative
        mul!(dudt[:,:,k], V', r_q[:,:,k])
        mass_matrix_solve!(mass_solver, k, dudt[:,:,k], u_q[:,:,k])
    end

    return dudt
end