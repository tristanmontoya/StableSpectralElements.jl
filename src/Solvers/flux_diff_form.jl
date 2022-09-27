struct StrongFluxDiffForm <: AbstractResidualForm 
    mapping_form::AbstractMappingForm
end

function StrongFluxDiffForm()
    return StrongFluxDiffForm(StandardMapping())
end

"""
Make operators for strong (diagonal-E) flux differencing form
"""
function make_operators(spatial_discretization::SpatialDiscretization{d}, 
    form::StrongFluxDiffForm) where {d}
    return make_operators(spatial_discretization,
        StrongConservationForm(form.mapping_form))
end

"""
Evaluate semi-discrete residual for strong flux-differencing form
"""
function rhs!(dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{StrongFluxDiffForm, <:AbstractDiscretizationOperators, d, N_eq}, t::Float64; print::Bool=false) where {d, N_eq}

    @timeit "rhs!" begin

        @unpack conservation_law, operators, x_q, connectivity, form, strategy = solver

        N_el = size(operators)[1]
        N_f = size(operators[1].Vf)[1]
        u_facet = Array{Float64}(undef, N_f, N_eq, N_el)

        # get all facet state values
        for k in 1:N_el
            u_facet[:,:,k] = 
                @timeit "extrapolate solution" convert(
                    Matrix, operators[k].Vf * u[:,:,k])
        end

        # evaluate all local residuals
        for k in 1:N_el
            # gather external state to element
            u_out = Matrix{Float64}(undef, N_f, N_eq)

            for e in 1:N_eq
                u_out[:,e] = @timeit "gather external state" u_facet[
                    :,e,:][connectivity[:,k]]
            end
            
            f = @timeit "eval flux" physical_flux(
                conservation_law.inviscid_flux, 
                convert(Matrix, operators[k].V * u[:,:,k]))

            F = @timeit "eval volume two-point flux" two_point_flux(
                conservation_law.two_point_flux, 
                convert(Matrix, operators[k].V * u[:,:,k]))

            f_star = @timeit "eval numerical flux" numerical_flux(
                conservation_law.inviscid_numerical_flux,
                u_facet[:,:,k], u_out, operators[k].scaled_normal)

            f_fac = @timeit "eval facet flux diff" f_star - 
                sum(convert(Matrix,operators[k].NTR[m] * f[m]) 
                    for m in 1:d)

            if isnothing(conservation_law.source_term)
                s = nothing
            else
                s = @timeit "eval source term" evaluate(
                    conservation_law.source_term, 
                    Tuple(x_q[m][:,k] for m in 1:d), t)
            end

            if print
                println("numerical flux: \n", f_star)
                println("normal trace difference: \n", f_fac)
            end
            
            # apply operators
            dudt[:,:,k] = @timeit "eval residual" apply_operators!(
                dudt[:,:,k], operators[k], F, f_fac, strategy, s)
        end
    end
    return nothing
end