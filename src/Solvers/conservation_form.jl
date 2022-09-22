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
    form::StrongConservationForm) where {d}

    @unpack N_el, M = spatial_discretization
    @unpack D, V, Vf, R, W, B = spatial_discretization.reference_approximation
    @unpack nrstJ = 
        spatial_discretization.reference_approximation.reference_element
    @unpack J_q, Λ_q, nJf = spatial_discretization.geometric_factors

    operators = Array{PhysicalOperators}(undef, N_el)
    for k in 1:N_el
        if d == 1
            VOL = (-V' * W * D[1] + Diagonal(nrstJ[1]) * R,)
        else
            if form.mapping_form isa SkewSymmetricMapping
                error("StrongConservationForm only implements standard conservative mapping")
            else
                VOL = Tuple(sum(-V' * W * D[m] * Diagonal(Λ_q[:,m,n,k]) + 
                    Vf' * B * Diagonal(nrstJ[m]) * R * Diagonal(Λ_q[:,m,n,k]) 
                        for m in 1:d) for n in 1:d)
            end
        end
        FAC = -Vf' * B
        SRC = V' * W * Diagonal(J_q[:,k])
        operators[k] = PhysicalOperators{d}(VOL, FAC, SRC, M[k], V, Vf,
            Tuple(nJf[m][:,k] for m in 1:d))
    end
    return operators
end

"""
Make operators for weak conservation form
"""
function make_operators(spatial_discretization::SpatialDiscretization{d}, 
    form::WeakConservationForm) where {d}

    @unpack N_el, M, reference_approximation = spatial_discretization
    @unpack ADVw, V, Vf, R, W, B, D = reference_approximation
    @unpack nrstJ = reference_approximation.reference_element
    @unpack J_q, Λ_q, nJf = spatial_discretization.geometric_factors

    operators = Array{PhysicalOperators}(undef, N_el)
    for k in 1:N_el
        if d == 1
            VOL = (ADVw[1],)
        else
            if form.mapping_form isa SkewSymmetricMapping
                VOL = Tuple(
                        0.5*(sum(ADVw[m] * Diagonal(Λ_q[:,m,n,k]) -
                            V' * Diagonal(Λ_q[:,m,n,k]) * W * D[m] 
                            for m in 1:d) +
                            Vf' * B * Diagonal(nJf[n][:,k]) * R)
                        for n in 1:d)
            else
                VOL = Tuple(sum(ADVw[m] * Diagonal(Λ_q[:,m,n,k]) for m in 1:d)  
                    for n in 1:d)
            end
        end
        FAC = -Vf' * B
        SRC = V' * W * Diagonal(J_q[:,k])
        operators[k] = PhysicalOperators{d}(VOL, FAC, SRC, M[k], V, Vf,
            Tuple(nJf[m][:,k] for m in 1:d))
    end
    return operators
end

"""
Evaluate semi-discrete residual for a hyperbolic problem
"""
function rhs!(dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:AbstractResidualForm, Hyperbolic},
    t::Float64) where {d}

    @timeit "rhs!" begin
        @unpack conservation_law, operators, x_q, connectivity, form, strategy = solver
        @unpack inviscid_numerical_flux = form
        @unpack source_term, N_eq = conservation_law

        N_el = size(operators,1)
        N_f = size(operators[1].Vf,1)
        u_facet = Array{Float64}(undef, N_f, N_eq, N_el)

        # get all facet state values
        Threads.@threads for k in 1:N_el
            u_facet[:,:,k] = @timeit get_timer(string("thread_timer_", Threads.threadid())) "extrap solution" convert(
                Matrix, operators[k].Vf * u[:,:,k])
        end

        # evaluate all local residuals
        Threads.@threads for k in 1:N_el
            to = get_timer(string("thread_timer_", Threads.threadid()))

            # gather external state to element
            @timeit to "gather ext state" begin
                u_out = Matrix{Float64}(undef, N_f, N_eq)
                @inbounds for e in 1:N_eq
                    u_out[:,e] = u_facet[:,e,:][connectivity[:,k]]
                end
            end
            
            # evaluate physical and numerical flux
            f = @timeit to "eval flux" physical_flux(
                conservation_law, Matrix(operators[k].V * u[:,:,k]))
            f_star = @timeit to "eval num flux" numerical_flux(
                conservation_law, inviscid_numerical_flux,
                u_facet[:,:,k], u_out, operators[k].scaled_normal)
            
            # evaluate source term, if there is one
            if conservation_law.source_term isa NoSourceTerm
                s = nothing
            else
                s = @timeit to "eval src term" evaluate(
                    source_term, Tuple(x_q[m][:,k] for m in 1:d),t)
            end

            # apply operators to obtain residual as
            # du/dt = M \ (VOL⋅f + FAC⋅f_star + SRC⋅s)
            dudt[:,:,k] = @timeit to "eval residual" apply_operators!(
                dudt[:,:,k], operators[k], f, f_star, strategy, s)
        end
    end
    return dudt
end

"""
Evaluate semi-discrete residual for a mixed/parabolic problem
"""
function rhs!(dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{d, <:AbstractResidualForm, <:Union{Mixed,Parabolic}},
    t::Float64) where {d}

    @timeit "rhs!" begin
        @unpack conservation_law, operators, x_q, connectivity, form, strategy = solver
        @unpack inviscid_numerical_flux, viscous_numerical_flux = form
        @unpack source_term, N_eq = conservation_law
        
        N_el = size(operators,1)
        N_f, N_p = size(operators[1].Vf)
        u_facet = Array{Float64}(undef, N_f, N_eq, N_el)

        # auxiliary variable q = ∇u
        q = Tuple(Array{Float64}(undef, N_p, N_eq, N_el) for m in 1:d) 
        q_facet = Tuple(Array{Float64}(undef, N_f, N_eq, N_el) for m in 1:d)

        # get all facet state values
        Threads.@threads for k in 1:N_el
            u_facet[:,:,k] = @timeit get_timer(string("thread_timer_", Threads.threadid())) "extrap solution" convert(
                Matrix, operators[k].Vf * u[:,:,k])
        end

        # evaluate auxiliary variable 
        Threads.@threads for k in 1:N_el
            to = get_timer(string("thread_timer_", Threads.threadid()))
            @timeit to "auxiliary variable" begin

                # gather external state to element
                @timeit to "gather extern state" begin
                    u_out = Matrix{Float64}(undef, N_f, N_eq)
                    @inbounds for e in 1:N_eq
                        u_out[:,:,e] = u_facet[:,e,:][connectivity[:,k]]
                    end
                end
                
                # evaluate nodal solution
                u_nodal = @timeit to "eval solution" Matrix(
                    operators[k].V * u[:,:,k])

                # evaluate numerical trace (d-vector of approximations to u nJf)
                u_star = @timeit to "eval num trace" numerical_flux(
                    conservation_law, viscous_numerical_flux,
                    u_facet[:,:,k], u_out, operators[k].scaled_normal)
                
                # apply operators
                @timeit to "apply operators" begin
                    @inbounds for m in 1:d
                        q[m][:,:,k] = auxiliary_variable!(m,
                            q[m][:,:,k], operators[k], u_nodal, 
                            u_star[m], strategy)
                    end
                end
            end
            
            @timeit to "extrap aux variable" begin
                @inbounds for m in 1:d
                    q_facet[m][:,:,k] = Matrix(operators[k].Vf * q[m][:,:,k])
                end
                
            end
        end

        # evaluate all local residuals
        Threads.@threads for k in 1:N_el
            to = get_timer(string("thread_timer_", Threads.threadid()))

            @timeit to "primary variable" begin

                # gather external state to element
                @timeit to "gather extern state" begin
                    u_out = Matrix{Float64}(undef, N_f, N_eq)
                    @inbounds for e in 1:N_eq
                        u_out[:,e] = u_facet[:,e,:][connectivity[:,k]]
                    end
                    q_out = Tuple(Matrix{Float64}(undef, N_f, N_eq) 
                        for m in 1:d)
                    @inbounds for e in 1:N_eq, m in 1:d
                        q_out[m][:,e] = q_facet[m][:,e,:][connectivity[:,k]]
                    end
                end
                
                # evaluate physical flux
                f = @timeit to "eval flux" physical_flux(
                    conservation_law, Matrix(operators[k].V * u[:,:,k]), 
                    Tuple(Matrix(operators[k].V * q[m][:,:,k]) for m in 1:d))
                
                # evaluate inviscid numerical flux 
                f_star_inv = @timeit to "eval inv num flux" numerical_flux(
                    conservation_law, inviscid_numerical_flux,
                    u_facet[:,:,k], u_out, operators[k].scaled_normal)
                    
                # evaluate viscous numerical flux
                f_star_vis = 
                    @timeit to "eval visc num flux" numerical_flux(
                        conservation_law, viscous_numerical_flux,
                        u_facet[:,:,k], u_out, 
                        Tuple(q_facet[m][:,:,k] for m in 1:d), 
                        Tuple(q_out[m] for m in 1:d),
                        operators[k].scaled_normal)
                
                # evaluate source term, if there is one
                if source_term isa NoSourceTerm
                    s = nothing
                else
                    s = @timeit to "eval src term" evaluate(
                        source_term, Tuple(x_q[m][:,k] for m in 1:d),t)
                end

                # apply operators
                dudt[:,:,k] = @timeit to "apply operators" apply_operators!(
                    dudt[:,:,k], operators[k], f, f_star_inv + f_star_vis, 
                    strategy, s)
            end
        end
    end
    return dudt
end