struct ErrorAnalysis{d, V_err_type, Volq_to_Err_type} <: AbstractAnalysis
    N_c::Int
    N_e::Int
    V_err::V_err_type
    w_err::Vector{Float64}
    volq_to_err::Volq_to_Err_type
    xyzq::NTuple{d, Matrix{Float64}}
    J_q::Matrix{Float64}
    total_volume::Float64
    results_path::String
end

function ErrorAnalysis(results_path::String,
        conservation_law::AbstractConservationLaw,
        spatial_discretization::SpatialDiscretization{d},
        error_quadrature_rule = nothing) where {d}
    (; N_e, reference_approximation) = spatial_discretization
    (; xyzq) = spatial_discretization.mesh
    (; V, reference_element, approx_type) = reference_approximation
    (; J_q) = spatial_discretization.geometric_factors

    if isnothing(error_quadrature_rule)
        (w_err, V_err) = (diag(reference_approximation.W), V)
        volq_to_err = diagm(ones(length(w_err)))
    else
        # Note: this introduces an additional approximation if the mapping and 
        # Jacobian determinant are over degree p.
        # Otherwise we have to recompute the Jacobian rather than just
        # interpolate, which I haven't done here.
        (; wq, rstq, element_type) = reference_element
        error_quad = quadrature(element_type, error_quadrature_rule)
        r_err = error_quad[1:d]
        w_err = error_quad[d + 1]
        V_modes_to_errq = vandermonde(element_type, approx_type.p, r_err...)
        VDM = vandermonde(element_type, approx_type.p, rstq...)
        P_volq_to_modes = inv(VDM' * Diagonal(wq) * VDM) * VDM' * Diagonal(wq)

        volq_to_err = V_modes_to_errq * P_volq_to_modes
        V_err = volq_to_err * V
    end
    total_volume = 0.0
    for k in 1:N_e
        total_volume += sum(w_err .* volq_to_err * J_q[:, k])
    end
    _, N_c, N_e = get_dof(spatial_discretization, conservation_law)

    return ErrorAnalysis(N_c,
        N_e,
        V_err,
        w_err,
        volq_to_err,
        xyzq,
        J_q,
        total_volume,
        results_path)
end

function analyze(analysis::ErrorAnalysis{d},
        sol::Array{Float64, 3},
        exact_solution,
        t::Float64 = 0.0;
        normalize = false,
        write_to_file = true) where {d}
    (; N_c, N_e, J_q, w_err, V_err, volq_to_err, xyzq, total_volume, results_path) = analysis

    u_approx = Matrix{Float64}(undef, size(V_err, 1), N_c)
    error = zeros(N_c)
    @inbounds @views for k in 1:N_e
        u_exact = evaluate(exact_solution, Tuple(volq_to_err * xyzq[m][:, k] for m in 1:d),
            t)
        u_approx = V_err * sol[:, :, k]

        for e in 1:N_c
            error_nodal = u_exact[:, e] .- u_approx[:, e]
            error[e] += dot(error_nodal, (w_err .* volq_to_err * J_q[:, k]) .* error_nodal)
        end
    end

    if normalize
        error = sqrt.(error ./ total_volume)
    else
        error = sqrt.(error)
    end

    if write_to_file
        save(string(results_path, "error.jld2"),
            Dict("error_analysis" => analysis, "error" => error))
    end

    return error
end
