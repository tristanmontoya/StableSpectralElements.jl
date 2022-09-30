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

    operators = Array{DiscretizationOperators}(undef, N_el)
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
        operators[k] = DiscretizationOperators{d}(VOL, FAC, SRC, M[k], V, Vf,
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

    operators = Array{DiscretizationOperators}(undef, N_el)
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

    @CLOUD_timeit "rhs!" begin

        @unpack conservation_law, operators, x_q, connectivity, form, strategy = solver
        @unpack inviscid_numerical_flux = form
        @unpack source_term, N_eq = conservation_law

        N_el = size(operators,1)
        N_q = size(operators[1].V,1)
        N_f = size(operators[1].Vf,1)

        u_in = Array{Float64}(undef, N_f, N_eq, N_el)
        
        # get all internal facet state values
        Threads.@threads for k in 1:N_el
            u_in[:,:,k] = @CLOUD_timeit "extrap solution" mul!(similar(u_in[:,:,k]), operators[k].Vf, u[:,:,k])
        end

        # evaluate all local residuals
        Threads.@threads for k in 1:N_el

            @unpack V, scaled_normal = operators[k]
            
            # gather external state at facet nodes
            u_out = @CLOUD_timeit "gather ext state" hcat(
                [u_in[:,e,:][connectivity[:,k]] for e in 1:N_eq]... )

            # evaluate numerical flux at facet nodes
            f_star = @CLOUD_timeit "eval num flux" numerical_flux(
                conservation_law, inviscid_numerical_flux,
                u_in[:,:,k], u_out, scaled_normal)

            # evaluate solution at volume nodes
            u_nodal = Matrix{Float64}(undef,N_q,N_eq)
            @CLOUD_timeit "eval solution" mul!(u_nodal, V, u[:,:,k])

            # evaluate physical flux at volume nodes
            f = @CLOUD_timeit "eval flux" physical_flux(conservation_law,
                u_nodal)

            # evaluate source term, if there is one
            if source_term isa NoSourceTerm
                s = nothing
            else
                s = @CLOUD_timeit "eval src term" evaluate(
                    source_term, Tuple(x_q[m][:,k] for m in 1:d),t)
            end

            dudt[:,:,k] = @CLOUD_timeit "apply operators" apply_operators!(
                dudt[:,:,k], operators[k], f, f_star, strategy, s)
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

    @CLOUD_timeit "rhs!" begin

        @unpack conservation_law, operators, x_q, connectivity, form, strategy = solver
        @unpack inviscid_numerical_flux, viscous_numerical_flux = form
        @unpack source_term, N_eq = conservation_law
        
        N_el = size(operators,1)
        N_f, N_p = size(operators[1].Vf)
        N_q = size(operators[1].V,1)
        
        q = Tuple(Array{Float64}(undef, N_p, N_eq, N_el) for m in 1:d) 
        q_in = Tuple(Array{Float64}(undef, N_f, N_eq, N_el) for m in 1:d)
        u_in = Array{Float64}(undef, N_f, N_eq, N_el)
        
        # get all internal facet state values
        Threads.@threads for k in 1:N_el
            u_in[:,:,k] = @CLOUD_timeit "extrap solution" mul!(
                similar(u_in[:,:,k]), operators[k].Vf, u[:,:,k])
        end

        # evaluate auxiliary variable 
        Threads.@threads for k in 1:N_el

            @CLOUD_timeit "auxiliary variable" begin

                @unpack V, scaled_normal = operators[k]

                # gather external state at facet nodes
                u_out = @CLOUD_timeit "gather ext state" hcat(
                    [u_in[:,e,:][connectivity[:,k]] for e in 1:N_eq]...)
                
                # evaluate solution at volume nodes
                u_nodal = Matrix{Float64}(undef,N_q,N_eq)
                @CLOUD_timeit "eval solution" mul!(u_nodal, V, u[:,:,k])

                # evaluate numerical trace (d-vector of approximations to u nJf)
                u_star = @CLOUD_timeit "eval num trace" numerical_flux(
                    conservation_law, viscous_numerical_flux,
                    u_in[:,:,k], u_out, operators[k].scaled_normal)
                
                # apply operators
                @CLOUD_timeit "apply operators" @inbounds for m in 1:d
                    q[m][:,:,k] = auxiliary_variable!(m, q[m][:,:,k], 
                        operators[k], u_nodal, u_star[m], strategy)
                end
            end
            
            @CLOUD_timeit "extrap aux variable" @inbounds for m in 1:d
                q_in[m][:,:,k] = mul!(similar(u_in[:,:,k]), 
                    operators[k].Vf, q[m][:,:,k])
            end
        end

        # evaluate all local residuals
        Threads.@threads for k in 1:N_el

            @CLOUD_timeit "primary variable" begin

                @unpack V, scaled_normal = operators[k]

                # gather external state to element
                u_out = Matrix{Float64}(undef, N_f, N_eq)
                q_out = Tuple(Array{Float64}(undef, N_f, N_eq) for m in 1:d)
                @CLOUD_timeit "gather ext state" @inbounds for e in 1:N_eq
                    u_out[:,e] = u_in[:,e,:][connectivity[:,k]]
                    @inbounds for m in 1:d
                        q_out[m][:,e] = q_in[m][:,e,:][connectivity[:,k]]
                    end
                end

                # evaluate nodal values of auxiliary variable
                q_nodal = Tuple(Array{Float64}(undef, N_q, N_eq) for m in 1:d) 
                @CLOUD_timeit "eval aux variable" @inbounds for m in 1:d
                    mul!(q_nodal[m], V, q[m][:,:,k])
                end
    
                # evaluate nodal solution
                u_nodal = Matrix{Float64}(undef,N_q,N_eq)
                @CLOUD_timeit "eval solution" mul!(u_nodal, V, u[:,:,k])

                # evaluate physical flux
                f = @CLOUD_timeit "eval flux" physical_flux(
                    conservation_law, u_nodal, q_nodal)
                
                # evaluate inviscid numerical flux 
                f_star = @CLOUD_timeit "eval inv num flux" numerical_flux(
                    conservation_law, inviscid_numerical_flux,
                    u_in[:,:,k], u_out, operators[k].scaled_normal)
                    
                # evaluate viscous numerical flux
                f_star += @CLOUD_timeit "eval visc num flux" numerical_flux(
                        conservation_law, viscous_numerical_flux,
                        u_in[:,:,k], u_out, Tuple(q_in[m][:,:,k] for m in 1:d), 
                        q_out, operators[k].scaled_normal)
                
                # evaluate source term, if there is one
                if source_term isa NoSourceTerm
                    s = nothing
                else
                    s = @CLOUD_timeit "eval src term" evaluate(
                        source_term, Tuple(x_q[m][:,k] for m in 1:d),t)
                end

                # apply operators
                dudt[:,:,k] = @CLOUD_timeit "apply operators" apply_operators!(
                    dudt[:,:,k], operators[k], f, f_star, strategy, s)
            end
        end
    end
    
    return dudt
end