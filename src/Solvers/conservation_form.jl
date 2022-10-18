Base.@kwdef struct StrongConservationForm{MappingForm,TwoPointFlux} <: AbstractResidualForm{MappingForm,TwoPointFlux}
    mapping_form::MappingForm = StandardMapping()
    inviscid_numerical_flux::AbstractInviscidNumericalFlux =
        LaxFriedrichsNumericalFlux()
    viscous_numerical_flux::AbstractViscousNumericalFlux = BR1()
    two_point_flux::TwoPointFlux = NoTwoPointFlux()
end

Base.@kwdef struct WeakConservationForm{MappingForm,TwoPointFlux} <: AbstractResidualForm{MappingForm,TwoPointFlux}
    mapping_form::MappingForm = StandardMapping()
    inviscid_numerical_flux::AbstractInviscidNumericalFlux =
        LaxFriedrichsNumericalFlux()
    viscous_numerical_flux::AbstractViscousNumericalFlux = BR1()
    two_point_flux::TwoPointFlux = NoTwoPointFlux()
end

"""
Make operators for strong conservation form
"""
function make_operators(spatial_discretization::SpatialDiscretization{d}, 
    ::StrongConservationForm{StandardMapping,<:AbstractTwoPointFlux}) where {d}

    @unpack N_e, M = spatial_discretization
    @unpack D, V, Vf, R, W, B = spatial_discretization.reference_approximation
    @unpack nrstJ = 
        spatial_discretization.reference_approximation.reference_element
    @unpack J_q, Λ_q, nJf = spatial_discretization.geometric_factors

    operators = Array{DiscretizationOperators}(undef, N_e)
    Threads.@threads for k in 1:N_e
        if d == 1
            VOL = (-V' * W * D[1] + Diagonal(nrstJ[1]) * R,)
        else
            VOL = Tuple(sum(-V' * W * D[m] * Diagonal(Λ_q[:,m,n,k]) + 
                Vf' * B * Diagonal(nrstJ[m]) * R * Diagonal(Λ_q[:,m,n,k]) 
                    for m in 1:d) for n in 1:d)
        end
        FAC = -Vf' * B
        SRC = V' * W * Diagonal(J_q[:,k])
        operators[k] = DiscretizationOperators{d}(VOL, FAC, SRC, M[k], V, Vf,
            Tuple(nJf[m][:,k] for m in 1:d))
    end
    return operators
end

"""
Make operators for weak conservation form
"""
function make_operators(spatial_discretization::SpatialDiscretization{1}, 
    ::WeakConservationForm{StandardMapping,<:AbstractTwoPointFlux})

    @unpack N_e, M, reference_approximation = spatial_discretization
    @unpack D, V, Vf, R, W, B = reference_approximation
    @unpack nrstJ = reference_approximation.reference_element
    @unpack J_q, Λ_q, nJf = spatial_discretization.geometric_factors

    operators = Array{DiscretizationOperators}(undef, N_e)
    Threads.@threads for k in 1:N_e

        VOL = (D[1]' * W,)
        FAC = -R' * B
        SRC = Diagonal(W * J_q[:,k])

        operators[k] = DiscretizationOperators{1}(VOL, FAC, SRC, M[k], V, Vf,
            (nJf[1][:,k],))
    end
    return operators
end

function make_operators(spatial_discretization::SpatialDiscretization{d}, 
    ::WeakConservationForm{StandardMapping,<:AbstractTwoPointFlux}) where {d}

    @unpack N_e, M, reference_approximation = spatial_discretization
    @unpack V, Vf, R, W, B, D = reference_approximation
    @unpack nrstJ = reference_approximation.reference_element
    @unpack J_q, Λ_q, nJf = spatial_discretization.geometric_factors

    operators = Array{DiscretizationOperators}(undef, N_e)
    Threads.@threads for k in 1:N_e
        VOL = Tuple(sum(D[m]' * Diagonal(W * Λ_q[:,m,n,k]) for m in 1:d)
                    for n in 1:d)
        FAC = -R' * B
        SRC = Diagonal(W * J_q[:,k])
        operators[k] = DiscretizationOperators{d}(VOL, FAC, SRC, M[k], V, Vf,
            Tuple(nJf[m][:,k] for m in 1:d))
    end
    return operators
end

function make_operators(spatial_discretization::SpatialDiscretization{d}, 
    ::WeakConservationForm{<:SkewSymmetricMapping,<:AbstractTwoPointFlux}) where {d}

    @unpack N_e, M, reference_approximation = spatial_discretization
    @unpack V, Vf, R, W, B, D = reference_approximation
    @unpack nrstJ = reference_approximation.reference_element
    @unpack J_q, Λ_q, nJf = spatial_discretization.geometric_factors

    operators = Array{DiscretizationOperators}(undef, N_e)
    Threads.@threads for k in 1:N_e
        VOL = Tuple(
                0.5*(sum(D[m]' * Diagonal(W * Λ_q[:,m,n,k]) -
                        Diagonal(W * Λ_q[:,m,n,k]) * D[m] for m in 1:d) +
                    R' * Diagonal(B * nJf[n][:,k]) * R)
                for n in 1:d)
        FAC = -R' * B
        SRC = Diagonal(W * J_q[:,k])
        operators[k] = DiscretizationOperators{d}(VOL, FAC, SRC, M[k], V, Vf,
            Tuple(nJf[m][:,k] for m in 1:d))
    end
    return operators
end

"""
Evaluate semi-discrete residual for a first-order problem
"""
function rhs!(dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:AbstractResidualForm, FirstOrder},
    t::Float64) where {d}

    @timeit thread_timer() "unpack/allocations" begin
        @unpack conservation_law, operators, x_q, connectivity, form, strategy = solver
        @unpack inviscid_numerical_flux = form
        @unpack source_term, N_c = conservation_law

        N_e = size(operators,1)
        N_q = size(operators[1].V,1)
        N_f = size(operators[1].Vf,1)

        u_in = Array{Float64}(undef, N_f, N_c, N_e)
    end

    # get all internal facet state values
    Threads.@threads for k in 1:N_e
        @unpack Vf = operators[k]
        for e in 1:N_c
            @timeit thread_timer() "extrap solution" begin
                u_in[:,e,k] = Vf * u[:,e,k]
            end
        end
    end

    # evaluate all local residuals
    Threads.@threads for k in 1:N_e
        @timeit thread_timer() "local residual" begin

            @unpack V, scaled_normal = operators[k]
            
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

            @timeit thread_timer() "eval flux" begin
                f = physical_flux(conservation_law, u_nodal)
            end

            @timeit thread_timer() "eval num flux" begin
                f_star = numerical_flux(
                    conservation_law, inviscid_numerical_flux,
                    u_in[:,:,k], u_out, scaled_normal)
            end

            if source_term isa NoSourceTerm
                s = nothing
            else
                @timeit thread_timer() "eval src term" begin
                    s = evaluate(source_term, Tuple(x_q[m][:,k] for m in 1:d),t)
                end
            end

            @timeit thread_timer() "apply operators" begin
                dudt[:,:,k] = apply_operators(operators[k], f, f_star,
                    strategy, s)
            end
        end
    end

    return dudt
end

"""
Evaluate semi-discrete residual for a second-order problem
"""
function rhs!(dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:AbstractResidualForm, SecondOrder},
    t::Float64) where {d}

    @timeit thread_timer() "unpack/allocations" begin

        @unpack conservation_law, operators, x_q, connectivity, form, strategy = solver
        @unpack inviscid_numerical_flux, viscous_numerical_flux = form
        @unpack source_term, N_c = conservation_law
        
        N_e = size(operators,1)
        N_f, N_p = size(operators[1].Vf)
        N_q = size(operators[1].V,1)
        
        q = Tuple(Array{Float64}(undef, N_p, N_c, N_e) for m in 1:d) 
        u_in = Array{Float64}(undef, N_f, N_c, N_e)
        q_in = Tuple(Array{Float64}(undef, N_f, N_c, N_e) for m in 1:d)
        
    end
    
    # get all internal facet state values
    Threads.@threads for k in 1:N_e
        @unpack Vf = operators[k]
        @inbounds for e in 1:N_c
            @timeit thread_timer() "extrap solution" begin
                u_in[:,e,k] = Vf * u[:,e,k]
            end
        end
    end

    # evaluate auxiliary variable 
    Threads.@threads for k in 1:N_e

        @timeit thread_timer() "auxiliary variable" begin

            @unpack V, scaled_normal = operators[k]

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
                    u_in[:,:,k], u_out, operators[k].scaled_normal)
            end
            
            @timeit thread_timer() "apply operators" @inbounds for m in 1:d
                q[m][:,:,k] = auxiliary_variable(
                    m, operators[k], u_nodal, u_star[m], strategy)
            end
        end
        
        @inbounds for e in 1:N_c, m in 1:d
            @timeit thread_timer() "extrap aux variable" begin
                q_in[m][:,e,k] = operators[k].Vf * q[m][:,e,k]
            end
        end
    end

    # evaluate all local residuals
    Threads.@threads for k in 1:N_e

        @timeit thread_timer() "local residual" begin

            @unpack V, scaled_normal = operators[k]

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
                    u_in[:,:,k], u_out, operators[k].scaled_normal)
            end
                
            @timeit thread_timer() "eval visc num flux" begin
                f_star = f_star + numerical_flux(conservation_law,
                    viscous_numerical_flux, u_in[:,:,k], u_out, 
                        Tuple(q_in[m][:,:,k] for m in 1:d), 
                        q_out, operators[k].scaled_normal)
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
                dudt[:,:,k] = apply_operators(
                    operators[k], f, f_star, strategy, s)
            end
        end
    end

    return dudt
end