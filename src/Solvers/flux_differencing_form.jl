@inline @views function flux_difference!(r_q::AbstractMatrix{Float64}, # N_q x N_c 
        S::NTuple{d, Matrix{Float64}}, # N_q x N_q
        conservation_law::AbstractConservationLaw{d,
            FirstOrder,
            N_c},
        two_point_flux::AbstractTwoPointFlux,
        Λ_q::AbstractArray{Float64, 3}, # N_q x d x d
        u_q::AbstractMatrix{Float64}) where {d, N_c}
    fill!(r_q, 0.0)
    @inbounds for i in axes(u_q, 1)
        for j in (i + 1):size(u_q, 1)

            # evaluate two-point flux (must be symmetric)
            F_ij = compute_two_point_flux(conservation_law,
                two_point_flux,
                u_q[i, :],
                u_q[j, :])

            # apply flux-differencing operator to flux tensor
            for e in 1:N_c
                diff_ij = 0.0
                for m in 1:d
                    Fm_ij = 0.0
                    for n in 1:d
                        Λ_ij = Λ_q[i, m, n] + Λ_q[j, m, n]
                        @muladd Fm_ij = Fm_ij + Λ_ij * F_ij[e, n]
                    end
                    @muladd diff_ij = diff_ij + S[m][i, j] * Fm_ij
                end
                r_q[i, e] -= diff_ij
                r_q[j, e] += diff_ij
            end
        end
    end
end

@inline @views function flux_difference!(r_q::AbstractMatrix{Float64}, # N_q x N_c 
        S::NTuple{d, SparseMatrixCSC{Float64}}, # N_q x N_q
        conservation_law::AbstractConservationLaw{d,
            FirstOrder,
            N_c},
        two_point_flux::AbstractTwoPointFlux,
        Λ_q::AbstractArray{Float64, 3}, # N_q x d x d
        u_q::AbstractMatrix{Float64}) where {d, N_c}
    fill!(r_q, 0.0)
    @inbounds for m in 1:d
        Sm_nz = nonzeros(S[m])
        row_index = rowvals(S[m])
        for j in axes(u_q, 1)
            for ii in nzrange(S[m], j)
                i = row_index[ii]
                if i < j
                    # evaluate two-point flux
                    F_ij = compute_two_point_flux(conservation_law,
                        two_point_flux,
                        u_q[i, :],
                        u_q[j, :])
                    Sm_ij = Sm_nz[ii]

                    # apply flux-differencing operator to flux tensor
                    for e in 1:N_c
                        Fm_ij = 0.0
                        for n in 1:d
                            Λ_ij = Λ_q[i, m, n] + Λ_q[j, m, n]
                            @muladd Fm_ij = Fm_ij + Λ_ij * F_ij[e, n]
                        end
                        diff_ij = Sm_ij * Fm_ij
                        r_q[i, e] -= diff_ij
                        r_q[j, e] += diff_ij
                    end
                end
            end
        end
    end
end

# no-op for LGL/diagonal-E operators
@inline function facet_correction!(::AbstractMatrix{Float64},
        ::AbstractMatrix{Float64},
        ::Nothing, # no facet correction operator
        ::AbstractConservationLaw{d},
        ::AbstractTwoPointFlux,
        ::AbstractMatrix{Float64},
        ::AbstractArray{Float64, 3},
        ::AbstractMatrix{Float64},
        ::AbstractMatrix{Float64},
        ::Int) where {d}
    return
end

@inline @views function facet_correction!(r_q::AbstractMatrix{Float64}, # N_q x N_c 
        f_f::AbstractMatrix{Float64}, # N_f x N_c
        C::Matrix{Float64}, # N_f x N_q
        conservation_law::AbstractConservationLaw{d,
            FirstOrder,
            N_c},
        two_point_flux::AbstractTwoPointFlux,
        halfnJf::AbstractMatrix{Float64},
        halfnJq::AbstractArray{Float64, 3},
        u_q::AbstractMatrix{Float64},
        u_f::AbstractMatrix{Float64},
        nodes_per_face::Int) where {d, N_c}
    @inbounds for i in axes(u_q, 1)
        for j in axes(u_f, 1)
            F_ij = compute_two_point_flux(conservation_law,
                two_point_flux,
                u_q[i, :],
                u_f[j, :])

            f = (j - 1) ÷ nodes_per_face + 1

            for e in 1:N_c
                F_dot_n_ij = 0.0
                for m in 1:d
                    nJ_ij = halfnJf[m, j] + halfnJq[m, f, i]
                    @muladd F_dot_n_ij = F_dot_n_ij + nJ_ij * F_ij[e, m]
                end
                diff_ij = C[i, j] * F_dot_n_ij
                r_q[i, e] -= diff_ij
                f_f[j, e] -= diff_ij
            end
        end
    end
end

@inline @views function facet_correction!(r_q::AbstractMatrix{Float64}, # N_q x N_c 
        f_f::AbstractMatrix{Float64}, # N_f x N_c
        C::SparseMatrixCSC{Float64}, # N_f x N_q
        conservation_law::AbstractConservationLaw{d,
            FirstOrder,
            N_c},
        two_point_flux::AbstractTwoPointFlux,
        halfnJf::AbstractMatrix{Float64},
        halfnJq::AbstractArray{Float64, 3},
        u_q::AbstractMatrix{Float64},
        u_f::AbstractMatrix{Float64},
        nodes_per_face::Int) where {d, N_c}
    C_nz = nonzeros(C)
    row_index = rowvals(C)

    @inbounds for j in axes(u_f, 1)
        for ii in nzrange(C, j)
            i = row_index[ii]

            # evaluate two-point flux
            F_ij = compute_two_point_flux(conservation_law,
                two_point_flux,
                u_q[i, :],
                u_f[j, :])
            C_ij = C_nz[ii]

            # get facet index 
            # (note this won't work if different number of nodes per facet)
            f = (j - 1) ÷ nodes_per_face + 1

            for e in 1:N_c
                F_dot_n_ij = 0.0
                for m in 1:d
                    nJ_ij = halfnJf[m, j] + halfnJq[m, f, i]
                    @muladd F_dot_n_ij = F_dot_n_ij + nJ_ij * F_ij[e, m]
                end
                diff_ij = C_ij * F_dot_n_ij
                r_q[i, e] -= diff_ij
                f_f[j, e] -= diff_ij
            end
        end
    end
end

# specialize for LGL/Diag-E nodal operators (no entropy projection)
@inline function entropy_projection!(::AbstractMassMatrixSolver,
        conservation_law::AbstractConservationLaw,
        u_q::AbstractMatrix,
        u_f::AbstractMatrix,
        ::AbstractMatrix,
        ::AbstractMatrix,
        ::AbstractMatrix,
        V::UniformScalingMap, # nodal
        ::LinearMap,
        R::SelectionMap, # diag-E
        ::Diagonal,
        u::AbstractMatrix,
        ::Int)
    mul!(u_q, V, u)
    mul!(u_f, R, u_q)
    return
end

# specialized for nodal schemes (not necessarily diagonal-E)
@inline @views function entropy_projection!(::AbstractMassMatrixSolver,
        conservation_law::AbstractConservationLaw,
        u_q::AbstractMatrix,
        u_f::AbstractMatrix,
        w_q::AbstractMatrix,
        w_f::AbstractMatrix,
        w::AbstractMatrix,
        V::UniformScalingMap, # nodal
        ::LinearMap,
        R::LinearMap, # not just SelectionMap
        ::Diagonal,
        u::AbstractMatrix,
        ::Int)
    mul!(u_q, V, u)
    @inbounds for i in axes(u, 1)
        conservative_to_entropy!(w_q[i, :], conservation_law, u_q[i, :])
    end
    mul!(w_f, R, w_q)
    @inbounds for i in axes(u_f, 1)
        entropy_to_conservative!(u_f[i, :], conservation_law, w_f[i, :])
    end
end

# most general (i.e. suitable for modal) approach
@inline @views function entropy_projection!(mass_solver::AbstractMassMatrixSolver,
        conservation_law::AbstractConservationLaw,
        u_q::AbstractMatrix,
        u_f::AbstractMatrix,
        w_q::AbstractMatrix,
        w_f::AbstractMatrix,
        w::AbstractMatrix,
        V::LinearMap, # not just UniformScalingMap
        Vᵀ::LinearMap,
        R::LinearMap, # not just SelectionMap
        WJ::Diagonal,
        u::AbstractMatrix,
        k::Int)

    # evaluate entropy variables in terms of nodal conservative variables
    mul!(u_q, V, u)
    @inbounds for i in axes(u_q, 1)
        conservative_to_entropy!(w_q[i, :], conservation_law, u_q[i, :])
    end

    # project entropy variables and store modal coeffs in w
    lmul!(WJ, w_q)
    mul!(w, Vᵀ, w_q)
    mass_matrix_solve!(mass_solver, k, w, w_q) # w = M[k] \ w

    # get nodal values of projected entropy variables
    mul!(w_q, V, w)
    mul!(w_f, R, w_q)

    # convert back to conservative variables
    @inbounds for i in axes(u_q, 1)
        entropy_to_conservative!(u_q[i, :], conservation_law, w_q[i, :])
    end
    @inbounds for i in axes(u_f, 1)
        entropy_to_conservative!(u_f[i, :], conservation_law, w_f[i, :])
    end
end

# for scalar equations, no entropy projection
@inline @views function nodal_values!(u::AbstractArray{Float64, 3},
        solver::Solver{<:AbstractConservationLaw{<:Any, 1},
            <:FluxDifferencingOperators,
            <:AbstractMassMatrixSolver,
            <:FluxDifferencingForm},
        k::Int)
    (; u_q, u_f) = solver.preallocated_arrays
    (; V, R) = solver.operators

    mul!(u_q[:, :, k], V, u[:, :, k])
    mul!(u_f[:, k, :], R, u_q[:, :, k])
    return
end

# for systems, dispatch entropy projection on V and R
@inline @views function nodal_values!(u::AbstractArray{Float64, 3},
        solver::Solver{<:AbstractConservationLaw,
            <:FluxDifferencingOperators,
            <:AbstractMassMatrixSolver,
            <:FluxDifferencingForm},
        k::Int)
    (; conservation_law, preallocated_arrays, mass_solver) = solver
    (; f_f, u_q, r_q, u_f, temp) = preallocated_arrays
    (; V, Vᵀ, R, WJ) = solver.operators

    id = Threads.threadid()
    entropy_projection!(mass_solver,
        conservation_law,
        u_q[:, :, k],
        u_f[:, k, :],
        r_q[:, :, id],
        f_f[:, :, id],
        temp[:, :, id],
        V,
        Vᵀ,
        R,
        WJ[k],
        u[:, :, k],
        k)
end

@inline @views function time_derivative!(dudt::AbstractArray{Float64, 3},
        solver::Solver{<:AbstractConservationLaw,
            <:FluxDifferencingOperators,
            <:AbstractMassMatrixSolver,
            <:FluxDifferencingForm},
        k::Int)
    (; conservation_law, connectivity, form, mass_solver) = solver
    (; inviscid_numerical_flux, two_point_flux) = form
    (; f_f, u_q, r_q, u_f, CI) = solver.preallocated_arrays
    (; S, C, Vᵀ, Rᵀ, Λ_q, BJf, halfnJq, halfnJf, n_f, nodes_per_face) = solver.operators

    # get thread id for temporary register
    id = Threads.threadid()

    # evaluate interface numerical flux
    numerical_flux!(f_f[:, :, id],
        conservation_law,
        inviscid_numerical_flux,
        u_f[:, k, :],
        u_f[CI[connectivity[:, k]], :],
        n_f[:, :, k],
        two_point_flux)

    # scale numerical flux by quadrature weights
    lmul!(BJf[k], f_f[:, :, id])

    # volume flux differencing term
    flux_difference!(r_q[:, :, id],
        S,
        conservation_law,
        two_point_flux,
        Λ_q[:, :, :, k],
        u_q[:, :, k])

    # apply facet correction term (if C is not nothing)
    facet_correction!(r_q[:, :, id],
        f_f[:, :, id],
        C,
        conservation_law,
        two_point_flux,
        halfnJf[:, :, k],
        halfnJq[:, :, :, k],
        u_q[:, :, k],
        u_f[:, k, :],
        nodes_per_face)

    # apply facet operators
    mul!(u_q[:, :, k], Rᵀ, f_f[:, :, id])
    r_q[:, :, id] .-= u_q[:, :, k]

    # solve for time derivative
    mul!(dudt[:, :, k], Vᵀ, r_q[:, :, id])
    mass_matrix_solve!(mass_solver, k, dudt[:, :, k], u_q[:, :, k])
end
