@inline @views function nodal_values!(u::AbstractArray{Float64, 3},
        solver::Solver{<:AbstractConservationLaw,
            <:AbstractDiscretizationOperators,
            <:AbstractMassMatrixSolver,
            <:StandardForm},
        k::Int)
    (; u_q, u_f) = solver.preallocated_arrays
    (; V, R) = solver.operators

    mul!(u_q[:, :, k], V, u[:, :, k])
    mul!(u_f[:, k, :], R, u_q[:, :, k])

    return
end

@inline @views function time_derivative!(dudt::AbstractArray{Float64, 3},
        solver::Solver{<:AbstractConservationLaw{d,
                FirstOrder},
            <:ReferenceOperators,
            <:AbstractMassMatrixSolver,
            <:StandardForm},
        k::Int) where {d}
    (; conservation_law, connectivity, form) = solver
    (; inviscid_numerical_flux) = form
    (; f_q, f_f, f_n, u_q, r_q, u_f, temp, CI) = solver.preallocated_arrays
    (; D, Dᵀ, Vᵀ, R, Rᵀ, halfWΛ, halfN, BJf, n_f) = solver.operators

    id = Threads.threadid()
    physical_flux!(f_q[:, :, :, id], conservation_law, u_q[:, :, k])
    numerical_flux!(f_f[:, :, id],
        conservation_law,
        inviscid_numerical_flux,
        u_f[:, k, :],
        u_f[CI[connectivity[:, k]], :],
        n_f[:, :, k])

    fill!(r_q[:, :, id], 0.0)
    @inbounds for n in 1:d
        # apply volume operators
        @inbounds for m in 1:d
            mul!(temp[:, :, id], halfWΛ[m, n, k], f_q[:, :, n, id])
            mul!(u_q[:, :, k], Dᵀ[m], temp[:, :, id])
            r_q[:, :, id] .+= u_q[:, :, k]
            mul!(u_q[:, :, k], D[m], f_q[:, :, n, id])
            lmul!(halfWΛ[m, n, k], u_q[:, :, k])
            r_q[:, :, id] .-= u_q[:, :, k]
        end

        # difference facet flux
        mul!(f_n[:, :, id], R, f_q[:, :, n, id])
        lmul!(halfN[n, k], f_n[:, :, id])
        f_f[:, :, id] .-= f_n[:, :, id]
    end

    # apply facet operators
    lmul!(BJf[k], f_f[:, :, id])
    mul!(temp[:, :, id], Rᵀ, f_f[:, :, id])
    r_q[:, :, id] .-= temp[:, :, id]

    # solve for time derivative
    mul!(dudt[:, :, k], Vᵀ, r_q[:, :, id])
    mass_matrix_solve!(solver.mass_solver, k, dudt[:, :, k], temp[:, :, id])
end

@inline @views function time_derivative!(dudt::AbstractArray{Float64, 3},
        solver::Solver{<:AbstractConservationLaw{d,
                FirstOrder},
            <:PhysicalOperators,
            <:AbstractMassMatrixSolver,
            <:StandardForm},
        k::Int) where {d}
    (; conservation_law, operators, connectivity, form) = solver
    (; VOL, FAC, n_f) = operators
    (; inviscid_numerical_flux) = form
    (; f_q, f_f, u_q, u_f, temp, CI) = solver.preallocated_arrays

    id = Threads.threadid()
    physical_flux!(f_q[:, :, :, id], conservation_law, u_q[:, :, k])
    numerical_flux!(f_f[:, :, id],
        conservation_law,
        inviscid_numerical_flux,
        u_f[:, k, :],
        u_f[CI[connectivity[:, k]], :],
        n_f[:, :, k])

    fill!(dudt[:, :, k], 0.0)
    @inbounds for m in 1:d
        mul!(temp[:, :, id], VOL[k][m], f_q[:, :, m, id])
        dudt[:, :, k] .+= temp[:, :, id]
    end

    mul!(temp[:, :, id], FAC[k], f_f[:, :, id])
    dudt[:, :, k] .+= temp[:, :, id]
end
