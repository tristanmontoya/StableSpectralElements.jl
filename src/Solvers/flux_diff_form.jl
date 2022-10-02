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

#TODO add flux differencing operators (old code below does not work)

#=
"""
Compute the flux-differencing term (D ⊙ F)1 (note: this currently is not supported in this version)
"""
    function flux_diff(D::LinearMaps.WrappedMap, F::AbstractArray{Float64,3})
        N_p = size(D,1)
        N_eq = size(F,3)

        y = Matrix{Float64}(undef,N_p,N_eq)
        
        for l in 1:N_eq, i in 1:N_p
            y[i,l] = dot(D.lmap[i,:], F[i,:,l])
        end
        return 2.0*y
    end

    function apply_operators!(residual::Matrix{Float64},
        operators::DiscretizationOperators{d},
        F::NTuple{d,Array{Float64,3}}, 
        f_fac::Matrix{Float64}, 
        ::ReferenceOperator,
        s::Union{Matrix{Float64},Nothing}=nothing) where {d}
        

        @timeit thread_timer() "volume terms" begin
            volume_terms = zero(residual)
            @inbounds for m in 1:d
                volume_terms += flux_diff(operators.VOL[m], F[m])
            end
        end

        @timeit thread_timer() "facet terms" begin
            facet_terms = mul!(residual, operators.FAC, f_fac)
        end

        rhs = volume_terms + facet_terms
 
        if !isnothing(s)
            @timeit thread_timer() "source terms" begin
                source_terms = mul!(residual, operators.SRC, s)
            end
            rhs = rhs + source_terms
        end

        @timeit thread_timer() "mass matrix solve" begin
            residual = operators.M \ rhs
        end
        
        return residual
    end

    function apply_operators!(residual::Matrix{Float64},
        operators::DiscretizationOperators{d},
        F::NTuple{d,Array{Float64,3}}, 
        f_fac::Matrix{Float64}, 
        ::PhysicalOperator,
        s::Union{Matrix{Float64},Nothing}=nothing) where {d}
        

        @timeit thread_timer() "volume terms" begin
            # compute volume terms
            volume_terms = zero(residual)
            for m in 1:d
                volume_terms += flux_diff(operators.VOL[m], F[m])
            end
        end

        @timeit thread_timer() "facet terms" begin
            # compute facet terms
            facet_terms = mul!(residual, operators.FAC, f_fac)
        end

        rhs = volume_terms + facet_terms

        if !isnothing(s)
            @timeit thread_timer() "source terms" begin
                source_terms = mul!(residual, operators.SRC, s)
            end
            rhs = rhs + source_terms
        end

        return rhs
        return residual
    end
=#

"""
Evaluate semi-discrete residual for strong flux-differencing form
"""
function rhs!(dudt::AbstractArray{Float64,3}, u::AbstractArray{Float64,3}, 
    solver::Solver{StrongFluxDiffForm, <:AbstractDiscretizationOperators, d, N_eq}, t::Float64; print::Bool=false) where {d, N_eq}

    @timeit thread_timer() "rhs!" begin

        @unpack conservation_law, operators, x_q, connectivity, form, strategy = solver

        N_el = size(operators)[1]
        N_f = size(operators[1].Vf)[1]
        u_facet = Array{Float64}(undef, N_f, N_eq, N_el)

        # get all facet state values
        for k in 1:N_el
            u_facet[:,:,k] = 
                @timeit thread_timer() "extrapolate solution" convert(
                    Matrix, operators[k].Vf * u[:,:,k])
        end

        # evaluate all local residuals
        for k in 1:N_el
            # gather external state to element
            u_out = Matrix{Float64}(undef, N_f, N_eq)

            for e in 1:N_eq
                u_out[:,e] = @timeit thread_timer() "gather external state" u_facet[
                    :,e,:][connectivity[:,k]]
            end
            
            f = @timeit thread_timer() "eval flux" physical_flux(
                conservation_law.inviscid_flux, 
                convert(Matrix, operators[k].V * u[:,:,k]))

            F = @timeit thread_timer() "eval volume two-point flux" two_point_flux(
                conservation_law.two_point_flux, 
                convert(Matrix, operators[k].V * u[:,:,k]))

            f_star = @timeit thread_timer() "eval numerical flux" numerical_flux(
                conservation_law.inviscid_numerical_flux,
                u_facet[:,:,k], u_out, operators[k].scaled_normal)

            f_fac = @timeit thread_timer() "eval facet flux diff" f_star - 
                sum(convert(Matrix,operators[k].NTR[m] * f[m]) 
                    for m in 1:d)

            if isnothing(conservation_law.source_term)
                s = nothing
            else
                s = @timeit thread_timer() "eval source term" evaluate(
                    conservation_law.source_term, 
                    Tuple(x_q[m][:,k] for m in 1:d), t)
            end

            if print
                println("numerical flux: \n", f_star)
                println("normal trace difference: \n", f_fac)
            end
            
            # apply operators
            dudt[:,:,k] = @timeit thread_timer() "eval residual" apply_operators!(
                dudt[:,:,k], operators[k], F, f_fac, strategy, s)
        end
    end
    return nothing
end