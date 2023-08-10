"""
Make operators for weak conservation form
"""
function make_operators(spatial_discretization::SpatialDiscretization{1}, 
    ::StandardForm{StandardMapping},
    alg::AbstractOperatorAlgorithm,
    mass_solver::AbstractMassMatrixSolver)

    (; N_e, reference_approximation) = spatial_discretization
    (; D, V, R, W, B) = reference_approximation
    (; nJf) = spatial_discretization.geometric_factors

    VOL = Vector{NTuple{1,LinearMap}}(undef,N_e)
    FAC = Vector{LinearMap}(undef,N_e)
    V_ar = Vector{LinearMap}(undef,N_e)
    R_ar = Vector{LinearMap}(undef,N_e)
    n_f = Vector{NTuple{1, Vector{Float64}}}(undef,N_e)

    Threads.@threads for k in 1:N_e
        M⁻¹ = mass_matrix_inverse(mass_solver, k)
        VOL[k] = (make_operator(M⁻¹ * Matrix(V' * D[1]' * W), alg),)
        FAC[k] = make_operator(-M⁻¹ * Matrix(V' * R' * B), alg)
        V_ar[k] = make_operator(reference_approximation.V, alg)
        R_ar[k] = make_operator(reference_approximation.R, alg)
        n_f[k] = (nJf[1,:,k],)
    end
    return PhysicalOperators{1}(VOL, FAC, V_ar, R_ar, n_f)
end

function make_operators(spatial_discretization::SpatialDiscretization{d}, 
    ::StandardForm{StandardMapping},
    alg::AbstractOperatorAlgorithm,
    mass_solver::AbstractMassMatrixSolver) where {d}

    (; N_e, reference_approximation) = spatial_discretization
    (; V, R, W, B, D) = reference_approximation
    (; Λ_q, J_f, nJf) = spatial_discretization.geometric_factors
    
    VOL = Vector{NTuple{d,LinearMap}}(undef,N_e)
    FAC = Vector{LinearMap}(undef,N_e)
    V_ar = Vector{LinearMap}(undef,N_e)
    R_ar = Vector{LinearMap}(undef,N_e)
    n_f = Vector{NTuple{d, Vector{Float64}}}(undef,N_e)

    Threads.@threads for k in 1:N_e 
        M⁻¹ = mass_matrix_inverse(mass_solver, k)
        VOL[k] = Tuple(make_operator(M⁻¹ * Matrix(V' * 
            sum(D[m]' * Diagonal(W * Λ_q[:,m,n,k]) for m in 1:d)), alg) 
            for n in 1:d)
        FAC[k] = make_operator(-M⁻¹* 
            Matrix(V' * R' * Diagonal(B * J_f[:,k])), alg)
        V_ar[k] = make_operator(reference_approximation.V, alg)
        R_ar[k] = make_operator(reference_approximation.R, alg)
        n_f[k] = Tuple(nJf[m,:,k] ./ J_f[:,k] for m in 1:d)
    end

    return PhysicalOperators{d}(VOL, FAC, V_ar, R_ar, n_f)
end

function make_operators(spatial_discretization::SpatialDiscretization{d}, 
    ::StandardForm{SkewSymmetricMapping},
    alg::AbstractOperatorAlgorithm,
    mass_solver::AbstractMassMatrixSolver) where {d}

    (; N_e, reference_approximation) = spatial_discretization
    (; V, R, W, B, D) = reference_approximation
    (; Λ_q, J_f, nJf) = spatial_discretization.geometric_factors
 
    VOL = Vector{NTuple{d,LinearMap}}(undef,N_e)
    FAC = Vector{LinearMap}(undef,N_e)
    V_ar = Vector{LinearMap}(undef,N_e)
    R_ar = Vector{LinearMap}(undef,N_e)
    n_f = Vector{NTuple{d, Vector{Float64}}}(undef,N_e)

    Threads.@threads for k in 1:N_e
        M⁻¹ = mass_matrix_inverse(mass_solver, k)
        VOL[k] = Tuple(make_operator(M⁻¹ * Matrix(V' * 
            (sum(D[m]' * Diagonal(0.5 * W * Λ_q[:,m,n,k]) -
            Diagonal(0.5 * W * Λ_q[:,m,n,k]) * D[m] for m in 1:d) +
            R' * Diagonal(0.5 * B * nJf[n,:,k]) * R)), alg) for n in 1:d)
        FAC[k] = make_operator(-M⁻¹ * 
            Matrix(V' * R' * Diagonal(B * J_f[:,k])), alg)
        V_ar[k] = make_operator(reference_approximation.V, alg)
        R_ar[k] = make_operator(reference_approximation.R, alg)
        n_f[k] = Tuple(nJf[m,:,k] ./ J_f[:,k] for m in 1:d)
    end

    return PhysicalOperators{d}(VOL, FAC, V_ar, R_ar, n_f)
end