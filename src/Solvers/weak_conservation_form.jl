"""
Evaluate semi-discrete residual for a first-order problem
"""
function rhs!(dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:WeakConservationForm, FirstOrder}, t::Float64) where {d}

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

"""
Evaluate semi-discrete residual for a second-order problem
"""
function rhs!(dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:WeakConservationForm, SecondOrder},
    t::Float64) where {d}

    @unpack conservation_law, operators, x_q, connectivity, form, N_e = solver
    @unpack inviscid_numerical_flux, viscous_numerical_flux = form
    @unpack source_term, N_c = conservation_law
    @unpack N_p, N_q, N_f = operators[1] # assume all operators same size
    
    local_rhs =  Matrix{Float64}(undef,N_p,N_c)
    q = Tuple(Array{Float64}(undef, N_p, N_c, N_e) for m in 1:d) 
    u_in = Array{Float64}(undef, N_f, N_c, N_e)
    q_in = Tuple(Array{Float64}(undef, N_f, N_c, N_e) for m in 1:d)
        
    # get all internal facet state values
    for k in 1:N_e
        @unpack Vf = operators[k]
        @inbounds for e in 1:N_c
            @timeit thread_timer() "extrap solution" begin
                u_in[:,e,k] = Vf * u[:,e,k]
            end
        end
    end

    # evaluate auxiliary variable 
    for k in 1:N_e

        @timeit thread_timer() "auxiliary variable" begin

            @unpack V, n_f = operators[k]

            u_nodal = Matrix{Float64}(undef,N_q,N_c)
            u_out = Matrix{Float64}(undef,N_f,N_c)

            @inbounds for e in 1:N_c
                @timeit thread_timer() "gather ext state" begin
                    u_out[:,e] = u_in[:,e,:][connectivity[:,k]]
                end
                @timeit thread_timer() "eval nodal solution" begin
                    u_nodal[:,e] = V * u[:,e,k]
                end
            end

            # evaluate numerical trace (d-vector of approximations to u nJf)
            @timeit thread_timer() "eval num trace" begin
                u_star =  numerical_flux(
                    conservation_law, viscous_numerical_flux,
                    u_in[:,:,k], u_out, operators[k].n_f)
            end
            
            @timeit thread_timer() "apply operators" @inbounds for m in 1:d
                q[m][:,:,k] = auxiliary_variable(
                    m, operators[k], u_nodal, u_star[m])
            end
        end
        
        @inbounds for e in 1:N_c, m in 1:d
            @timeit thread_timer() "extrap aux variable" begin
                q_in[m][:,e,k] = operators[k].Vf * q[m][:,e,k]
            end
        end
    end

    # evaluate all local residuals
    for k in 1:N_e

        @timeit thread_timer() "local residual" begin

            @unpack V, n_f = operators[k]

            u_nodal = Matrix{Float64}(undef,N_q,N_c)
            u_out = Matrix{Float64}(undef,N_f,N_c)
            q_nodal = Tuple(Array{Float64}(undef, N_q, N_c) for m in 1:d)
            q_out = Tuple(Array{Float64}(undef, N_f, N_c) for m in 1:d)

            @inbounds for e in 1:N_c

                @timeit thread_timer() "gather ext state" begin
                    u_out[:,e] = u_in[:,e,:][connectivity[:,k]]
                    @inbounds for m in 1:d
                        q_out[m][:,e] = q_in[m][:,e,:][connectivity[:,k]]
                    end
                end
                
                @timeit thread_timer() "eval nodal solution" begin
                    u_nodal[:,e] = V * u[:,e,k]
                end
                
                @timeit thread_timer() "eval nodal aux var" begin
                    @inbounds for m in 1:d
                        q_nodal[m][:,e] = V * q[m][:,e,k]
                    end
                end
            end

            @timeit thread_timer() "eval flux" begin
                f = physical_flux(conservation_law, u_nodal, q_nodal)
            end
            
            @timeit thread_timer() "eval inv num flux" begin
                f_star = numerical_flux(
                    conservation_law, inviscid_numerical_flux,
                    u_in[:,:,k], u_out, operators[k].n_f)
            end
                
            @timeit thread_timer() "eval visc num flux" begin
                f_star = f_star + numerical_flux(conservation_law,
                    viscous_numerical_flux, u_in[:,:,k], u_out, 
                        Tuple(q_in[m][:,:,k] for m in 1:d), 
                        q_out, operators[k].n_f)
            end
            
            if source_term isa NoSourceTerm
                s = nothing
            else
                @timeit thread_timer() "eval src term" begin
                    s = evaluate(source_term, Tuple(x_q[m][:,k] 
                        for m in 1:d),t)
                end
            end

            @timeit thread_timer() "apply operators" begin
                apply_operators!(local_rhs, u_nodal, operators[k], f, f_star, s)
            end
            dudt[:,:,k] = local_rhs
        end
    end

    return dudt
end