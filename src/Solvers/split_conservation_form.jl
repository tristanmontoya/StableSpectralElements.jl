"""
Evaluate semi-discrete residual for a first-order problem
"""
function rhs!(dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:SplitConservationForm, FirstOrder},
    t::Float64) where {d}

    @unpack conservation_law, operators, connectivity, form, N_e = solver
    @unpack inviscid_numerical_flux = form
    @unpack source_term, N_c = conservation_law
    @unpack N_p, N_q, N_f = operators[1] # assume all operators same size

    facet_states = Array{Float64}(undef, N_f, N_e, N_c)
    u_q = Matrix{Float64}(undef,N_q,N_c)
    u_in = Matrix{Float64}(undef,N_f,N_c)
    local_rhs =  Matrix{Float64}(undef,N_p,N_c)
    CI = CartesianIndices((N_f,N_e))

    # get facet states through extrapolation
    @views @inbounds for k in 1:N_e
        @timeit thread_timer() "get facet states" begin
            mul!(u_in,  operators[k].Vf, u[:,:,k])
            facet_states[:,k,:] = u_in
        end
    end

    # evaluate all local residuals
    @views @inbounds for k in 1:N_e
        @timeit thread_timer() "local residual" begin

            @timeit thread_timer() "eval nodal solution" mul!(
                u_q, operators[k].V, u[:,:,k])

            @timeit thread_timer() "eval flux" begin
                f = physical_flux(conservation_law, u_q)
            end

            @timeit thread_timer() "eval num flux" begin
                f_star = numerical_flux(
                    conservation_law, inviscid_numerical_flux,
                    facet_states[:,k,:], 
                    facet_states[CI[connectivity[:,k]],:], 
                    operators[k].n_f)
            end

            @timeit thread_timer() "add extrap flux" @inbounds for m in 1:d
                mul!(u_in, operators[k].NTR[m], f[m])
                f_star .+= u_in
            end

            if source_term isa NoSourceTerm s = nothing else
                @timeit thread_timer() "eval src term" begin
                    s = evaluate(source_term, Tuple(x_q[m][:,k] for m in 1:d),t)
                end
            end

            @timeit thread_timer() "apply operators" apply_operators!(
                local_rhs, u_q, operators[k], f, f_star, s)
            dudt[:,:,k] = local_rhs
        end
    end

    return dudt
end
