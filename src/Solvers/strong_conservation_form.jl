struct StrongConservationForm <: AbstractResidualForm end

function semidiscretize(
    conservation_law::ConservationLaw{d,N_eq},spatial_discretization::SpatialDiscretization{d},
    initial_data::AbstractInitialData, 
    form::StrongConservationForm,
    tspan::NTuple{2,Float64}, ::Eager) where {d, N_eq}

    @unpack N_el, M = spatial_discretization
    @unpack ADVs, R, P, B = spatial_discretization.reference_approximation
    @unpack nrstJ = 
        spatial_discretization.reference_approximation.reference_element
    @unpack JinvG, nJf = spatial_discretization.geometric_factors

    u0 = initialize(
        initial_data,
        conservation_law,
        spatial_discretization)

    operators = Array{PhysicalOperatorsEager}(undef, N_el)
    for k in 1:N_el
        invM = inv(M[k])
        if d == 1
            VOL = (combine(-invM * ADVs[1]),)
            NTR = (combine(Diagonal(nJf[1][:,k]) * R * P),)
        else
            VOL = Tuple(combine(-invM * sum(ADVs[1] * Diagonal(JinvG[:,m,n,k])
                for m in 1:d) for n in 1:d)) 
            NTR = Tuple(combine(sum(Diagonal(nrstJ[m][:,k]) * R * P * 
                Diagonal(JinvG[:,m,n,k]) for m in 1:d)) for n in 1:d)
        end

        FAC = combine(-invM * transpose(R) * B)

        operators[k] = PhysicalOperatorsEager(VOL, FAC, R, NTR,
            Tuple(nJf[m][:,k] for m in 1:d))
    end

    solver = Solver(conservation_law, operators,
        spatial_discretization.mesh.mapP, form)

    return ODEProblem(rhs!, u0, tspan, solver)
end

function semidiscretize(
    conservation_law::ConservationLaw{d,N_eq},spatial_discretization::SpatialDiscretization{d},
    initial_data::AbstractInitialData, 
    form::StrongConservationForm,
    tspan::NTuple{2,Float64}, ::Lazy) where {d, N_eq}

    @unpack N_el, M = spatial_discretization
    @unpack ADVs, R, P, B = spatial_discretization.reference_approximation
    @unpack nrstJ = 
        spatial_discretization.reference_approximation.reference_element
    @unpack JinvG, nJf = spatial_discretization.geometric_factors

    u0 = initialize(
        initial_data,
        conservation_law,
        spatial_discretization)

    operators = Array{PhysicalOperatorsLazy}(undef, N_el)
    for k in 1:N_el

        if d == 1
            vol = (-ADVs[1],)
            NTR = (Diagonal(nJf[1][:,k]) * R * P,)
        else
            vol = Tuple(-sum(ADVs[1] *
                Diagonal(JinvG[:,m,n,k]) 
                for m in 1:d) for n in 1:d) 
            NTR = Tuple(sum(Diagonal(nrstJ[m][:,k]) * R * P *
                Diagonal(JinvG[:,m,n,k]) for m in 1:d) for n in 1:d)
        end

        fac = -transpose(R) * B

        operators[k] = PhysicalOperatorsLazy(vol, fac,
            M[k], R, NTR, Tuple(nJf[m][:,k] for m in 1:d))
    end

    solver = Solver(conservation_law, operators,
        spatial_discretization.mesh.mapP, form)

    return ODEProblem(rhs!, u0, tspan, solver)
end

function rhs!(dudt::Array{Float64,3}, u::Array{Float64,3}, 
    solver::Solver{StrongConservationForm, <:AbstractPhysicalOperators, d, N_eq}, t::Float64) where {d, N_eq}
    @timeit "rhs!" begin   

    @unpack conservation_law, operators, connectivity, form = solver

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
            u_out[:,e] = @timeit  "gather external state" u_facet[
                :,e,:][connectivity[:,k]]
        end
        
        # evaluate physical and numerical flux
        f = @timeit "eval flux" physical_flux(
            conservation_law.first_order_flux, u[:,:,k])

        f_star = @timeit "eval numerical flux" numerical_flux(
            conservation_law.first_order_numerical_flux,
            u_facet[:,:,k], u_out, operators[k].scaled_normal)

        f_fac = @timeit "eval flux diff" f_star - 
            sum(convert(Matrix,operators[k].NTR[m] * f[m]) 
                for m in 1:d)
        
        # apply operators
        dudt[:,:,k] = @timeit "eval residual" apply_operators(
            operators[k], f, f_fac)
    end
    end
    return nothing
end