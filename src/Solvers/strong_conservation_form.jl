struct StrongConservationForm <: AbstractResidualForm end

function make_operators(spatial_discretization::SpatialDiscretization{d}, 
    ::StrongConservationForm) where {d}

    @unpack N_el, M = spatial_discretization
    @unpack ADVs, V, R, P, B = spatial_discretization.reference_approximation
    @unpack nrstJ = 
        spatial_discretization.reference_approximation.reference_element
    @unpack JinvG, nJf = spatial_discretization.geometric_factors

    operators = Array{PhysicalOperators}(undef, N_el)
    for k in 1:N_el
        if d == 1
            VOL = (ADVs[1],)
            NTR = (Diagonal(nrstJ[1]) * R * P,)
        else
            VOL = Tuple(sum(ADVs[m] * Diagonal(JinvG[:,m,n,k])
                for m in 1:d) for n in 1:d) 
            NTR = Tuple(sum(Diagonal(nrstJ[m]) * R * P * 
                Diagonal(JinvG[:,m,n,k]) for m in 1:d) for n in 1:d)
        end
        FAC = -transpose(R) * B
        operators[k] = PhysicalOperators(VOL, FAC, M[k], V, R, NTR,
            Tuple(nJf[m][:,k] for m in 1:d))
    end
    return operators
end

function rhs!(dudt::Array{Float64,3}, u::Array{Float64,3}, 
    solver::Solver{StrongConservationForm, <:AbstractPhysicalOperators, d, N_eq}, t::Float64; print::Bool=false) where {d, N_eq}

    @timeit "rhs!" begin

        @unpack conservation_law, operators, connectivity, form, strategy = solver

        N_el = size(operators)[1]
        N_f = size(operators[1].R)[1]
        u_facet = Array{Float64}(undef, N_f, N_eq, N_el)

        # get all facet state values
        for k in 1:N_el
            u_facet[:,:,k] = 
                @timeit "extrapolate solution" convert(
                    Matrix, operators[k].R * u[:,:,k])
        end

        # evaluate all local residuals
        for k in 1:N_el
            # gather external state to element
            u_out = Matrix{Float64}(undef, N_f, N_eq)

            for e in 1:N_eq
                u_out[:,e] = @timeit "gather external state" u_facet[
                    :,e,:][connectivity[:,k]]
            end
            
            # evaluate physical and numerical flux
            f = @timeit "eval flux" physical_flux(
                conservation_law.first_order_flux, 
                convert(Matrix,operators[k].V * u[:,:,k]))

            f_star = @timeit "eval numerical flux" numerical_flux(
                conservation_law.first_order_numerical_flux,
                u_facet[:,:,k], u_out, operators[k].scaled_normal)

            f_fac = @timeit "eval flux diff" f_star - 
                sum(convert(Matrix,operators[k].NTR[m] * f[m]) 
                    for m in 1:d)

            if print
                println("numerical flux: \n", f_star)
                println("normal trace difference: \n", f_fac)
            end
            
            # apply operators
            dudt[:,:,k] = @timeit "eval residual" apply_operators(
                operators[k], f, f_fac, strategy)
        end
    end
    return nothing
end