struct ErrorAnalysis{d,V_err_type} <: AbstractAnalysis
    N_c::Int
    N_e::Int
    WJ_err::Vector{Diagonal{Float64, Vector{Float64}}}
    V_err::V_err_type
    x_err::NTuple{d, Matrix{Float64}}
    total_volume::Float64
    results_path::String
end

function ErrorAnalysis(results_path::String, 
    conservation_law::AbstractConservationLaw,
    spatial_discretization::SpatialDiscretization{d},
    error_quadrature_rule=nothing) where {d}

    (; N_e, reference_approximation) = spatial_discretization
    (; xyzq) = spatial_discretization.mesh
    (; V,reference_element,approx_type) = reference_approximation
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

        print("\nmaking WJ_err...")
        WJ_err = [Diagonal(w_err .* volq_to_err*J_q[:,k]) for k in 1:N_e]
        print(" ...done!\n")
        print("\nmaking x_err...")
        x_err = Tuple(volq_to_err * xyzq[m] for m in 1:d)
        print("... done!")
    end

    _, N_c, N_e = get_dof(spatial_discretization, conservation_law)
        
    return ErrorAnalysis(N_c, N_e, WJ_err, V_err, x_err,
        sum(sum.(WJ_err)), results_path)
end

function analyze(analysis::ErrorAnalysis{d}, sol::Array{Float64,3},
    exact_solution, t::Float64=0.0; 
    normalize=false, write_to_file=true) where {d}

    (; N_c, N_e, WJ_err, V_err, x_err, total_volume, results_path) = analysis 

    u_approx = Matrix{Float64}(undef, size(V_err,1), N_c)
    error = zeros(N_c)
    @inbounds @views for k in 1:N_e
        u_exact = evaluate(exact_solution, 
            Tuple(x_err[m][:,k] for m in 1:d), t)
        mul!(u_approx, V_err, sol[:,:,k])
        for e in 1:N_c
            error_nodal = u_exact[:,e] .- u_approx[:,e]
            error[e] += dot(error_nodal, WJ_err[k]*error_nodal)
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