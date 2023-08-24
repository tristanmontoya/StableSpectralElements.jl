using BenchmarkTools

"""We define a benchmark version of rhs! so that threading etc. doesn't get in the way"""
@views @timeit "du/dt" function rhs_benchmark!(
    dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:StandardForm, FirstOrder, PhysicalOperators{d},N_p,N_q,N_f,N_c,N_e},
    t::Float64) where {d,N_p,N_q,N_f,N_c,N_e}

    @timeit "unpack" begin
        (; conservation_law, operators, connectivity, form) = solver
        (; inviscid_numerical_flux) = form
        (; f_q, f_f, u_q, u_f, temp, CI) = solver.preallocated_arrays
    end
    
    k = 1  # just one element

    @timeit "vandermonde" mul!(u_q[:,:,k], operators.V[k], u[:,:,k])
    @timeit "extrap solution" mul!(u_f[:,k,:], operators.R[k], u_q[:,:,k])

    @timeit "phys flux" physical_flux!(f_q[:,:,:,k],
        conservation_law, u_q[:,:,k])

    @timeit "num flux" numerical_flux!(f_f[:,:,k],
        conservation_law,
        inviscid_numerical_flux, u_f[:,k,:], 
        u_f[CI[connectivity[:,k]],:], operators.n_f[k])
    
    @timeit "fill w zeros" fill!(view(dudt,:,:,k),0.0)
    
    @inbounds for m in 1:d
        @timeit "volume operators" begin
            mul!(view(temp,:,:,k),operators.VOL[k][m],f_q[:,:,m,k])
            dudt[:,:,k] .+= temp[:,:,k] 
        end
    end
    
    mul!(view(temp,:,:,k), operators.FAC[k], f_f[:,:,k])
    dudt[:,:,k] .+= temp[:,:,k]

    return dudt
end

@views @timeit "du/dt" function rhs_benchmark!(
    dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:StandardForm{SkewSymmetricMapping}, FirstOrder, ReferenceOperators{d},N_p,N_q,N_f,N_c,N_e},
    t::Float64) where {d,N_p,N_q,N_f,N_c,N_e}

    @timeit "unpack" begin
        (; conservation_law, connectivity, form) = solver
        (; inviscid_numerical_flux) = form
        (; f_q, f_f, f_n, u_q, r_q, u_f, temp, CI) = solver.preallocated_arrays
        (; D, V, R, halfWΛ, halfN, BJf, n_f) = solver.operators
    end
    
    k = 1  #just one element
    
    @timeit "vandermonde" mul!(u_q[:,:,k], V, u[:,:,k])
    
    @timeit "extrap solution" mul!(u_f[:,k,:], R, u_q[:,:,k])

    @timeit "phys flux" physical_flux!(f_q[:,:,:,k],
        conservation_law, u_q[:,:,k])

    @timeit "num flux" numerical_flux!(f_f[:,:,k],
        conservation_law,
        inviscid_numerical_flux, u_f[:,k,:], 
        u_f[CI[connectivity[:,k]],:], n_f[k])

    @timeit "fill w zeros" fill!(r_q[:,:,k],0.0)

    @inbounds for n in 1:d
        @inbounds @timeit "volume operators" for m in 1:d
            mul!(temp[:,:,k],halfWΛ[m,n,k],f_q[:,:,n,k])
            mul!(u_q[:,:,k],D[m]',temp[:,:,k])
            r_q[:,:,k] .+= u_q[:,:,k] 
            mul!(u_q[:,:,k],D[m],f_q[:,:,n,k])
            lmul!(halfWΛ[m,n,k],u_q[:,:,k])
            r_q[:,:,k] .-= u_q[:,:,k] 
        end
        
        # difference facet flux
        @timeit "diff flux" begin
            mul!(f_n[:,:,k], R, f_q[:,:,n,k])
            lmul!(halfN[n,k], f_n[:,:,k])
            f_f[:,:,k] .-= f_n[:,:,k]
        end
    end

    # apply facet operators
    @timeit "facet operators" begin
        lmul!(BJf[k], f_f[:,:,k])
        mul!(u_q[:,:,k], R', f_f[:,:,k])
        r_q[:,:,k] .-= u_q[:,:,k]
    end

    # solve for time derivative
    @timeit "trans. VDM" mul!(dudt[:,:,k], V', r_q[:,:,k])
    @timeit "mass solve" mass_matrix_solve!(
        solver.mass_solver, k, dudt[:,:,k], u_q[:,:,k])
    return dudt
end

@timeit "du/dt" function rhs_volume!(
    dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:StandardForm, FirstOrder, PhysicalOperators{d}, N_p,N_q,N_f,N_c,N_e},
    t::Float64) where {d,N_p,N_q,N_f,N_c,N_e}

    (; conservation_law, operators, connectivity, form) = solver
    (; inviscid_numerical_flux) = form
    (; f_q, f_f, u_q, u_f, temp, CI) = solver.preallocated_arrays
    
    @timeit "reconstruct nodal solution" for k in 1:N_e
        for e in 1:N_c
            u_q[:,e,k] = operators.V[k]*u[:,e,k]
            u_f[:,k,e] = operators.R[k]*u_q[:,e,k]
        end
    end

    @views @timeit "eval residual" for k in 1:N_e
        physical_flux!(f_q[:,:,:,k], conservation_law, u_q[:,:,k])

        numerical_flux!(f_f[:,:,k],
            conservation_law, inviscid_numerical_flux, u_f[:,k,:], 
            u_f[CI[connectivity[:,k]],:], operators.n_f[k])

        fill!(dudt[:,:,k],0.0)
        for e in 1:N_c
            @inbounds for m in 1:d
                dudt[:,e,k] = dudt[:,e,k] + operators.VOL[k][m]*u_q[:,e,k]
            end
            #dudt[:,e,k] = dudt[:,e,k] + operators.FAC[k]*f_f[:,e,k]
        end
    end
    return dudt
end

@timeit "du/dt" function rhs_facet!(
    dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:StandardForm, FirstOrder, PhysicalOperators{d}, N_p,N_q,N_f,N_c,N_e},
    t::Float64) where {d,N_p,N_q,N_f,N_c,N_e}

    (; conservation_law, operators, connectivity, form) = solver
    (; inviscid_numerical_flux) = form
    (; f_q, f_f, u_q, u_f, temp, CI) = solver.preallocated_arrays
    
    @timeit "reconstruct nodal solution" for k in 1:N_e
        for e in 1:N_c
            u_q[:,e,k] = operators.V[k]*u[:,e,k]
            u_f[:,k,e] = operators.R[k]*u_q[:,e,k]
        end
    end

    @views @timeit "eval residual" for k in 1:N_e
        physical_flux!(f_q[:,:,:,k], conservation_law, u_q[:,:,k])

        numerical_flux!(f_f[:,:,k],
            conservation_law, inviscid_numerical_flux, u_f[:,k,:], 
            u_f[CI[connectivity[:,k]],:], operators.n_f[k])

        fill!(dudt[:,:,k],0.0)
        for e in 1:N_c
            #@inbounds for m in 1:d
            #    dudt[:,e,k] = dudt[:,e,k] + operators.VOL[k][m]*f_q[:,e,m,k]
            #end
            dudt[:,e,k] = dudt[:,e,k] + operators.FAC[k]*f_f[:,e,k]
        end
    end
    return dudt
end

@timeit "du/dt" function rhs_benchmark!(
    dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:FluxDifferencingForm, FirstOrder, FluxDifferencingOperators{d}, N_p,N_q,N_f,N_c,N_e},
    t::Float64) where {d,N_p,N_q,N_f,N_c,N_e}

    (; conservation_law, connectivity, form, mass_solver) = solver
    (; inviscid_numerical_flux, two_point_flux, 
        entropy_projection, facet_correction) = form
    (; f_f, u_q, r_q, u_f, temp, CI) = solver.preallocated_arrays
    (; S, V, R, WJ, Λ_q, BJf, C, halfnJq, halfnJf, n_f,
        nodes_per_face) = solver.operators
    
    # get the nodal solution using the entropy projection if specified
    @inbounds @views @timeit "get nodal vals" for k in 1:N_e
        get_nodal_values!(mass_solver, conservation_law, u_q[:,:,k], 
            u_f[:,k,:], r_q[:,:,k], f_f[:,:,k], temp[:,:,k], 
            V, R, WJ[k], u[:,:,k], k, Val(entropy_projection))
    end

    # compute the local residual
    @inbounds @views @timeit "eval residual" for k in 1:N_e
        
        # evaluate interface numerical flux
        @timeit "num flux" numerical_flux!(f_f[:,:,k], conservation_law,
            inviscid_numerical_flux, u_f[:,k,:], u_f[CI[connectivity[:,k]],:], 
            n_f[k], two_point_flux)
        
        # scale numerical flux by quadrature weights
        @timeit "fac quadrature scale" lmul!(BJf[k], f_f[:,:,k])

        # volume flux differencing term
        @timeit "flux diff" flux_difference!(r_q[:,:,k], S, conservation_law, 
            two_point_flux, Λ_q[:,:,:,k], u_q[:,:,k])

        # apply facet correction term (for operators w/o boundary nodes)
        @timeit "facet corr" facet_correction!(r_q[:,:,k], f_f[:,:,k], C,
            conservation_law, two_point_flux, halfnJf[:,:,k], halfnJq[:,:,:,k], 
            u_q[:,:,k], u_f[:,k,:], nodes_per_face, Val(facet_correction))

        # apply facet operators
        @timeit "facet operator" begin
            mul!(u_q[:,:,k], R', f_f[:,:,k])
            r_q[:,:,k] .-= u_q[:,:,k]
        end

        # solve for time derivative
        @timeit "trans. VDM" mul!(dudt[:,:,k], V', r_q[:,:,k])
        @timeit "mass solve" mass_matrix_solve!(
            mass_solver, k, dudt[:,:,k], u_q[:,:,k])
    end

    return dudt
end