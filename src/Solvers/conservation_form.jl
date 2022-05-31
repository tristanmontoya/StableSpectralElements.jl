struct StrongConservationForm <: AbstractResidualForm 
    mapping_form::AbstractMappingForm
end

struct WeakConservationForm <: AbstractResidualForm 
    mapping_form::AbstractMappingForm
end

struct SplitConservationForm <: AbstractResidualForm 
    mapping_form::AbstractMappingForm
end

function StrongConservationForm()
    return StrongConservationForm(StandardMapping())
end

function WeakConservationForm()
    return WeakConservationForm(StandardMapping())
end

function SplitConservationForm()
    return SplitConservationForm(CreanMapping())
end

"""
    Make operators for strong conservation form
"""
function make_operators(spatial_discretization::SpatialDiscretization{d}, 
    ::StrongConservationForm) where {d}

    @unpack N_el, M = spatial_discretization
    @unpack ADVs, V, Vf, R, P, W, B = spatial_discretization.reference_approximation
    @unpack nrstJ = 
        spatial_discretization.reference_approximation.reference_element
    @unpack J_q, Λ_q, nJf = spatial_discretization.geometric_factors

    operators = Array{PhysicalOperators}(undef, N_el)
    for k in 1:N_el
        if d == 1
            VOL = (ADVs[1],)
            NTR = (Diagonal(nrstJ[1]) * R,)
        else
            VOL = Tuple(sum(ADVs[m] * Diagonal(Λ_q[:,m,n,k])
                for m in 1:d) for n in 1:d) 
            NTR = Tuple(sum(Diagonal(nrstJ[m]) * R * 
                Diagonal(Λ_q[:,m,n,k]) for m in 1:d) for n in 1:d)
        end
        FAC = -Vf' * B
        SRC = V' * W * Diagonal(J_q[:,k])
        operators[k] = PhysicalOperators(VOL, FAC, SRC, M[k], V, Vf, NTR,
            Tuple(nJf[m][:,k] for m in 1:d))
    end
    return operators
end

"""
    Make operators for weak conservation form
"""
function make_operators(spatial_discretization::SpatialDiscretization{d}, 
    form::WeakConservationForm) where {d}

    @unpack N_el, M = spatial_discretization
    @unpack ADVw, V, Vf, R, P, W, B, D = spatial_discretization.reference_approximation
    @unpack nrstJ = 
        spatial_discretization.reference_approximation.reference_element
    @unpack J_q, Λ_q, nJf = spatial_discretization.geometric_factors

    operators = Array{PhysicalOperators}(undef, N_el)
    for k in 1:N_el
        if d == 1
            VOL = (ADVw[1],)
            NTR = (Diagonal(nrstJ[1]) * R,)
        else
            if form.mapping_form isa SkewSymmetricMapping
                VOL = Tuple(
                    0.5*sum(ADVw[m] * Diagonal(Λ_q[:,m,n,k]) +
                        (P * Diagonal(Λ_q[:,m,n,k]) * V)' * ADVw[m] 
                        for m in 1:d)
                    for n in 1:d)

            elseif form.mapping_form isa CreanMapping
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
            
            # not used
            NTR = Tuple(sum(Diagonal(nrstJ[m]) * R * 
                Diagonal(Λ_q[:,m,n,k]) for m in 1:d) for n in 1:d)
        end
        FAC = -Vf' * B
        SRC = V' * W * Diagonal(J_q[:,k])
        operators[k] = PhysicalOperators(VOL, FAC, SRC, M[k], V, Vf, NTR,
            Tuple(nJf[m][:,k] for m in 1:d))
    end
    return operators
end

"""
    Make operators for split conservation form
"""
function make_operators(spatial_discretization::SpatialDiscretization{d}, 
    ::SplitConservationForm) where {d}
    @unpack N_el, M = spatial_discretization
    @unpack ADVw, V, Vf, R, P, W, B, D = spatial_discretization.reference_approximation
    @unpack nrstJ = 
        spatial_discretization.reference_approximation.reference_element
    @unpack J_q, Λ_q, nJf = spatial_discretization.geometric_factors

    operators = Array{PhysicalOperators}(undef, N_el)
    for k in 1:N_el
        if d == 1
            VOL = (ADVw[1],)
            NTR = (Diagonal(nrstJ[1]) * Vf,)
        else
            VOL = Tuple(
                0.5*sum(ADVw[m] * Diagonal(Λ_q[:,m,n,k]) - V' * Diagonal(Λ_q[:,m,n,k]) * W * D[m]
                    for m in 1:d)
                for n in 1:d)
            NTR = Tuple(0.5 * Diagonal(nJf[n][:,k]) * R
                for n in 1:d)
        end
        FAC = -Vf' * B
        SRC = V' * W * Diagonal(J_q[:,k])
        operators[k] = PhysicalOperators(VOL, FAC, SRC, M[k], V, Vf, NTR,
            Tuple(nJf[m][:,k] for m in 1:d))
    end
    return operators
end

"""
    Evaluate semi-discrete residual for strong/split conservation form
"""
function rhs!(dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{<:Union{StrongConservationForm,SplitConservationForm}, <:AbstractPhysicalOperators, d, N_eq}, t::Float64; print::Bool=false) where {d, N_eq}

    @timeit "rhs!" begin

        @unpack conservation_law, operators, x_q, connectivity, form, strategy = solver

        N_el = size(operators)[1]
        N_f = size(operators[1].Vf)[1]
        u_facet = Array{Float64}(undef, N_f, N_eq, N_el)

        # get all facet state values
        for k in 1:N_el
            u_facet[:,:,k] = 
                @timeit "extrapolate solution" Matrix(
                    operators[k].Vf * u[:,:,k])
        end

        # evaluate all local residuals
        for k in 1:N_el
            # gather external state to element
            @timeit "gather external state" begin
                u_out = Matrix{Float64}(undef, N_f, N_eq)
                @inbounds for e in 1:N_eq
                    u_out[:,e] = u_facet[:,e,:][connectivity[:,k]]
                end
            end
            
            # evaluate physical and numerical flux
            f = @timeit "eval flux" physical_flux(
                conservation_law.first_order_flux, 
                Matrix(operators[k].V * u[:,:,k]))
            f_star = @timeit "eval numerical flux" numerical_flux(
                conservation_law.first_order_numerical_flux,
                u_facet[:,:,k], u_out, operators[k].scaled_normal)
            f_fac = @timeit "eval flux diff" f_star - 
                sum(convert(Matrix,operators[k].NTR[m] * f[m]) 
                    for m in 1:d)

            # evaluate source term, if there is one
            s = @timeit "eval source term" evaluate(
                conservation_law.source_term, 
                Tuple(x_q[m][:,k] for m in 1:d), t)
                
            # apply operators
            dudt[:,:,k] = @timeit "eval residual" apply_operators!(
                dudt[:,:,k], operators[k], f, f_fac, strategy, s)
        end
    end
    return nothing
end

"""
    Evaluate semi-discrete residual for weak conservation form
"""
function rhs!(dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{WeakConservationForm, <:AbstractPhysicalOperators, d, N_eq},
    t::Float64; print::Bool=false) where {d, N_eq}

    @timeit "rhs!" begin

        @unpack conservation_law, operators, x_q, connectivity, form, strategy = solver

        N_el = size(operators)[1]
        N_f = size(operators[1].Vf)[1]
        u_facet = Array{Float64}(undef, N_f, N_eq, N_el)

        # get all facet state values
        for k in 1:N_el
            u_facet[:,:,k] = @timeit "extrapolate solution" convert(
                Matrix, operators[k].Vf * u[:,:,k])
        end

        # evaluate all local residuals
        for k in 1:N_el
            @timeit "gather external state" begin
                # gather external state to element
                u_out = Matrix{Float64}(undef, N_f, N_eq)
                @inbounds for e in 1:N_eq
                    u_out[:,e] = u_facet[:,e,:][connectivity[:,k]]
                end
            end
            
            # evaluate physical and numerical flux
            f = @timeit "eval flux" physical_flux(
                conservation_law.first_order_flux, 
                convert(Matrix,operators[k].V * u[:,:,k]))
            f_star = @timeit "eval numerical flux" numerical_flux(
                conservation_law.first_order_numerical_flux,
                u_facet[:,:,k], u_out, operators[k].scaled_normal)
            
            # evaluate source term, if there is one
            if isnothing(conservation_law.source_term)
                s = nothing
            else
                s = @timeit "eval source term" evaluate(
                    conservation_law.source_term, 
                    Tuple(x_q[m][:,k] for m in 1:d),t)
            end

            # apply operators
            dudt[:,:,k] = @timeit "eval residual" apply_operators!(
                dudt[:,:,k], operators[k], f, f_star, strategy, s)
        end
    end
    return dudt
end