struct StrongConservationForm<:AbstractResidualForm end

function semidiscretize(
    conservation_law::ConservationLaw{d,N_eq},spatial_discretization::SpatialDiscretization{d},
    initial_data::AbstractInitialData, 
    form::StrongConservationForm,
    tspan::NTuple{2,Float64}) where {d, N_eq}

    @unpack N_el, jacobian_inverse = spatial_discretization
    @unpack D, R, P, invM, B = spatial_discretization.reference_operators
    @unpack nrstJ = spatial_discretization.reference_element
    @unpack JinvG, nJf = spatial_discretization.geometric_factors

    u0 = initialize(
        initial_data,
        conservation_law,
        spatial_discretization)

    operators = Array{PhysicalOperatorsLinear}(undef, N_el)
    for k in 1:N_el
        if d == 1
            VOL = (-jacobian_inverse[k] * D[1] * P,)
            NORMAL_TRACE = (Diagonal(nJf[1][:,k]) * R * P,)
        else
            VOL = Tuple(-jacobian_inverse[k] * sum(D[m] * P *
                    Diagonal(JinvG[:,m,n,k]) for m in 1:d) for n in 1:d) 
            NORMAL_TRACE = Tuple(sum(
                Diagonal(nrstJ[m][:,k]) * R * P * Diagonal(JinvG[:,m,n,k]) for m in 1:d) for n in 1:d)
        end
        operators[k] = PhysicalOperatorsLinear(VOL,
            -jacobian_inverse[k] * invM * transpose(R) * B, R, 
            NORMAL_TRACE, Tuple(nJf[m][:,k] for m in 1:d))
    end

    solver = Solver(conservation_law, operators,
        spatial_discretization.mesh.mapP, form)

    return ODEProblem(rhs!, u0, tspan, solver)
end

function rhs!(dudt::Array{Float64,3}, u::Array{Float64,3}, 
    solver::Solver{StrongConservationForm, d, N_eq}, t::Float64) where {d, N_eq}

    @unpack conservation_law, operators, connectivity, form = solver

    N_el = size(operators)[1]
    N_f = size(operators[1].EXTRAPOLATE_SOLUTION)[1]
    u_facet = Array{Float64}(undef, N_f, N_eq, N_el)

    # get all facet state values
    for k in 1:N_el
        u_facet[:,:,k] = convert(Matrix,operators[k].EXTRAPOLATE_SOLUTION * u[:,:,k])
    end
    
    # evaluate all local residuals 
    for k in 1:N_el
        # gather external state to element
        u_out = Matrix{Float64}(undef, N_f, N_eq)
        for e in 1:N_eq
            u_out[:,e] = u_facet[:,e,:][connectivity[:,k]]
        end

        # evaluate physical and numerical flux
        f = physical_flux(conservation_law.first_order_flux, u[:,:,k])
        f_star = numerical_flux(conservation_law.first_order_numerical_flux,
            u_facet[:,:,k], u_out, operators[k].scaled_normal)
        
        # apply operators
        dudt[:,:,k] = convert(Matrix, sum(operators[k].VOL[m] * f[m] 
            for m in 1:d) + operators[k].FAC * 
                (f_star - sum(operators[k].NORMAL_TRACE[m] * f[m] 
                for m in 1:d)))
    end
    return nothing
end