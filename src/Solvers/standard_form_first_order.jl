@inline @views function nodal_values!(u::AbstractArray{Float64,3},
    solver::Solver{d,ResidualForm,PDEType,ConservationLaw,
    Operators,MassSolver,Parallelism,N_p,N_q,N_f,N_c,N_e}, 
    k::Int) where {d, ResidualForm<:StandardForm,PDEType,ConservationLaw,
    Operators<:ReferenceOperators,MassSolver,
    Parallelism,N_p,N_q,N_f,N_c,N_e}

    (; u_q, u_f) = solver.preallocated_arrays
    (; V, R) = solver.operators

    mul!(u_q[:,:,k], V, u[:,:,k])
    mul!(u_f[:,k,:], R, u_q[:,:,k])
end

@inline @views function nodal_values!(u::AbstractArray{Float64,3},
    solver::Solver{d,ResidualForm,PDEType,ConservationLaw,
    Operators,MassSolver,Parallelism,N_p,N_q,N_f,N_c,N_e}, 
    k::Int) where {d,ResidualForm<:StandardForm,PDEType,ConservationLaw,
    Operators<:PhysicalOperators,MassSolver,Parallelism,N_p,N_q,N_f,N_c,N_e}

    (; u_q, u_f) = solver.preallocated_arrays
    (; V, R) = solver.operators

    mul!(u_q[:,:,k], V[k], u[:,:,k])
    mul!(u_f[:,k,:], R[k], u_q[:,:,k])
end

@inline @views function time_derivative!(dudt::AbstractArray{Float64,3},
    solver::Solver{d,ResidualForm,FirstOrder,ConservationLaw,
    Operators,MassSolver,Parallelism,N_p,N_q,N_f,N_c,N_e}, 
    k::Int) where {d,ResidualForm<:StandardForm,ConservationLaw,
    Operators<:ReferenceOperators,MassSolver,Parallelism,N_p,N_q,N_f,N_c,N_e}

    (; conservation_law, connectivity, form) = solver
    (; inviscid_numerical_flux) = form
    (; f_q, f_f, f_n, u_q, r_q, u_f, temp, CI) = solver.preallocated_arrays
    (; D, Dᵀ, Vᵀ, R, Rᵀ, halfWΛ, halfN, BJf, n_f) = solver.operators

    physical_flux!(f_q[:,:,:,k], conservation_law, u_q[:,:,k])

    numerical_flux!(f_f[:,:,k], conservation_law, inviscid_numerical_flux,
        u_f[:,k,:], u_f[CI[connectivity[:,k]],:], n_f[:,:,k])

    fill!(r_q[:,:,k], 0.0)
    @inbounds for n in 1:d
        # apply volume operators
        @inbounds for m in 1:d
            mul!(temp[:,:,k],halfWΛ[m,n,k],f_q[:,:,n,k])
            mul!(u_q[:,:,k],Dᵀ[m],temp[:,:,k])
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
    mul!(u_q[:,:,k], Rᵀ, f_f[:,:,k])
    r_q[:,:,k] .-= u_q[:,:,k]

    # solve for time derivative
    mul!(dudt[:,:,k], Vᵀ, r_q[:,:,k])
    mass_matrix_solve!(solver.mass_solver, k, dudt[:,:,k], u_q[:,:,k])
end

@inline @views function time_derivative!(dudt::AbstractArray{Float64,3},
    solver::Solver{d,ResidualForm,FirstOrder,ConservationLaw,
    Operators,MassSolver,Parallelism,N_p,N_q,N_f,N_c,N_e}, 
    k::Int) where {d, ResidualForm<:StandardForm, ConservationLaw,
    Operators<:PhysicalOperators,MassSolver,Parallelism,N_p,N_q,N_f,N_c,N_e}

    (; conservation_law, operators, connectivity, form) = solver
    (; VOL, FAC, n_f) = operators
    (; inviscid_numerical_flux) = form
    (; f_q, f_f, u_q, u_f, temp, CI) = solver.preallocated_arrays

    physical_flux!(f_q[:,:,:,k], conservation_law, u_q[:,:,k])

    numerical_flux!(f_f[:,:,k], conservation_law, inviscid_numerical_flux, 
        u_f[:,k,:], u_f[CI[connectivity[:,k]],:], n_f[:,:,k])
    
    fill!(dudt[:,:,k],0.0)
    @inbounds for m in 1:d
        mul!(temp[:,:,k], VOL[k][m], f_q[:,:,m,k])
        dudt[:,:,k] .+= temp[:,:,k]
    end

    mul!(temp[:,:,k], FAC[k], f_f[:,:,k])
    dudt[:,:,k] .+= temp[:,:,k]
end