struct ErrorAnalysis{d} <: AbstractAnalysis
    N_c::Int
    N_e::Int
    WJ_err::Vector{<:AbstractMatrix{Float64}}
    V_err::LinearMap
    x_err::NTuple{d, Matrix{Float64}}
    total_volume::Float64
    results_path::String
end

function ErrorAnalysis(conservation_law::AbstractConservationLaw,
    spatial_discretization::SpatialDiscretization,
    error_quadrature_rule=nothing)
    return ErrorAnalysis("./", conservation_law,
        spatial_discretization,
        error_quadrature_rule)
end

function ErrorAnalysis(results_path::String, 
    conservation_law::AbstractConservationLaw,
    spatial_discretization::SpatialDiscretization{d},
    error_quadrature_rule=nothing) where {d}

    (; N_e) = spatial_discretization
    (; xyzq) = spatial_discretization.mesh
    (; V,reference_element,approx_type) = spatial_discretization.reference_approximation
    (; J_q) = spatial_discretization.geometric_factors

    if isnothing(error_quadrature_rule)
        (W_err, V_err) = (spatial_discretization.reference_approximation.W, V)
        WJ_err = [W_err * Diagonal(J_q[:,k]) for k in 1:N_e]
        x_err = xyzq
    else 
        # Note: this introduces an additional approximation if the mapping and # Jacobian determinant are over degree p.
     
        # Otherwise we have to recompute the Jacobian rather than just
        # interpolate, which I haven't done here.

        (; wq, rstq, element_type) = reference_element
        error_quad = quadrature(element_type, error_quadrature_rule)
        r_err = error_quad[1:d]
        w_err = error_quad[d+1]
        V_modes_to_errq = vandermonde(element_type, approx_type.p, r_err...)
        VDM = vandermonde(element_type, approx_type.p, rstq...)
        P_volq_to_modes = inv(VDM' * Diagonal(wq) * VDM) * VDM' * Diagonal(wq)
        volq_to_err = V_modes_to_errq * P_volq_to_modes
        V_err = volq_to_err * V

        WJ_err = [Diagonal(w_err .* volq_to_err*J_q[:,k]) for k in 1:N_e]
        x_err = Tuple(volq_to_err * xyzq[m] for m in 1:d)
    end

    _, N_c, N_e = get_dof(spatial_discretization, conservation_law)
        
    return ErrorAnalysis(N_c, N_e, WJ_err, V_err, x_err,
        sum(sum.(WJ_err)), results_path)
end

function analyze(analysis::ErrorAnalysis{d}, 
    sol::Array{Float64,3}, 
    exact_solution::AbstractGridFunction{d},
    t::Float64=0.0; normalize=false, write_to_file=true) where {d}

    (; N_c, N_e, WJ_err, V_err, x_err, total_volume, results_path) = analysis 

    u_exact = evaluate(exact_solution, x_err, t)
    nodal_error = Tuple(u_exact[:,e,:] - convert(Matrix, V_err * sol[:,e,:])
        for e in 1:N_c)

    squared_error = [sum(dot(nodal_error[e][:,k], WJ_err[k]*nodal_error[e][:,
        k]) for k in 1:N_e) for e in 1:N_c]
    
    if normalize
        error = sqrt.(squared_error ./ total_volume)
    else
        error = sqrt.(squared_error)
    end

    if write_to_file
        save(string(results_path, "error.jld2"), 
            Dict("error_analysis" => analysis, "error" => error))
    end

    return error
end