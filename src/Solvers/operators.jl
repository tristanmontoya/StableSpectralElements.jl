function ReferenceOperators(
    reference_approximation::ReferenceApproximation{<:RefElemData{d}},
    alg::AbstractOperatorAlgorithm, Λ_q::Array{Float64,4}, 
    nJf::Array{Float64,3}, J_f::Matrix{Float64}) where {d}

    (; D, W, V, R, B) = reference_approximation
    (N_f, N_e) = size(J_f)

    halfWΛ = Array{Diagonal{Float64, Vector{Float64}},3}(undef, d, d, N_e)
    halfN = Matrix{Diagonal{Float64, Vector{Float64}}}(undef, d, N_e)
    BJf = Vector{Diagonal{Float64, Vector{Float64}}}(undef, N_e)
    n_f = Array{Float64,3}(undef, d, N_f, N_e)

    @inbounds for k in 1:N_e
        for m in 1:d
            halfWΛ[m,:,k] .= [Diagonal(0.5 * W * Λ_q[:,m,n,k]) for n in 1:d]
            n_f[m,:,k] .= nJf[m,:,k] ./ J_f[:,k]
            halfN[m,k] = Diagonal(0.5 * n_f[m,:,k])
        end
        BJf[k] = Diagonal(B .* J_f[:,k])
    end

    return ReferenceOperators(Tuple(make_operator(D[m], alg) for m in 1:d), 
        Tuple(transpose(make_operator(D[m], alg)) for m in 1:d), 
        make_operator(V, alg), transpose(make_operator(V, alg)),
        make_operator(R, alg), transpose(make_operator(R, alg)),
        W, B, halfWΛ, halfN, BJf, n_f)
end

function FluxDifferencingOperators(
    reference_approximation::ReferenceApproximation{<:RefElemData{d}},
    alg::AbstractOperatorAlgorithm, J_q::Matrix{Float64}, 
    Λ_q::Array{Float64,4}, nJq::Array{Float64,4}, nJf::Array{Float64,3}, 
    J_f::Matrix{Float64}) where {d}

    (; W, V, R, B, reference_element) = reference_approximation
    (N_f, N_e) = size(J_f)

    WJ = Vector{Diagonal{Float64, Vector{Float64}}}(undef, N_e)
    BJf = Vector{Diagonal{Float64, Vector{Float64}}}(undef, N_e)
    n_f = Array{Float64,3}(undef, d, N_f, N_e)
    
    @inbounds for k in 1:N_e
        WJ[k] = Diagonal(W .* J_q[:,k])
        BJf[k] = Diagonal(B .* J_f[:,k])
        @inbounds for m in 1:d
            n_f[m,:,k] = nJf[m,:,k] ./ J_f[:,k]
        end
    end
    
    S, C = flux_differencing_operators(reference_approximation)

    return FluxDifferencingOperators(S, C, make_operator(V, alg),
        transpose(make_operator(V, alg)), make_operator(R, alg), 
        transpose(make_operator(R, alg)), W, B, WJ, Λ_q, BJf, n_f, 0.5*nJf, 
        0.5*nJq, N_f÷num_faces(reference_element.element_type))
end

function PhysicalOperators(spatial_discretization::SpatialDiscretization{1}, 
    ::StandardForm{StandardMapping}, alg::AbstractOperatorAlgorithm,
    mass_solver::AbstractMassMatrixSolver)

    (; N_e, reference_approximation) = spatial_discretization
    (; D, V, R, W, B) = reference_approximation
    (; nJf) = spatial_discretization.geometric_factors

    VOL = Vector{NTuple{1,LinearMap}}(undef,N_e)
    FAC = Vector{LinearMap}(undef,N_e)

    @inbounds for k in 1:N_e
        M⁻¹ = mass_matrix_inverse(mass_solver, k)
        VOL[k] = (make_operator(M⁻¹ * Matrix(V' * D[1]' * W), alg),)
        FAC[k] = make_operator(-M⁻¹ * Matrix(V' * R' * B), alg)
    end
    return PhysicalOperators(
        VOL, FAC, make_operator(V, alg), make_operator(R, alg), nJf)
end

function PhysicalOperators(spatial_discretization::SpatialDiscretization{d}, 
    ::StandardForm{StandardMapping}, alg::AbstractOperatorAlgorithm,
    mass_solver::AbstractMassMatrixSolver) where {d}

    (; N_e, reference_approximation, geometric_factors) = spatial_discretization
    (; V, R, W, B, D) = reference_approximation

    (; Λ_q, nJf, J_f) = apply_reference_mapping(geometric_factors,
        reference_approximation.reference_mapping)
    
    VOL = Vector{NTuple{d,LinearMap}}(undef,N_e)
    FAC = Vector{LinearMap}(undef,N_e)
    n_f = Array{Float64,3}(undef, d, N_f, N_e)

    @inbounds for k in 1:N_e 
        M⁻¹ = mass_matrix_inverse(mass_solver, k)
        VOL[k] = Tuple(make_operator(M⁻¹ * Matrix(V' * 
            sum(D[m]' * Diagonal(W * Λ_q[:,m,n,k]) for m in 1:d)), alg) 
            for n in 1:d)
        FAC[k] = make_operator(-M⁻¹ * 
            Matrix(V' * R' * Diagonal(B * J_f[:,k])), alg)
        @inbounds for m in 1:d
            n_f[m,:,k] = nJf[m,:,k] ./ J_f[:,k]
        end
    end

    return PhysicalOperators(
        VOL, FAC, make_operator(V, alg), make_operator(R, alg), n_f)
end

function PhysicalOperators(spatial_discretization::SpatialDiscretization{d}, 
    ::StandardForm{SkewSymmetricMapping},
    alg::AbstractOperatorAlgorithm,
    mass_solver::AbstractMassMatrixSolver) where {d}

    (; N_e, reference_approximation, geometric_factors) = spatial_discretization
    (; V, R, W, B, D, N_f) = reference_approximation
    (; Λ_q, nJf, J_f) = apply_reference_mapping(geometric_factors,
        reference_approximation.reference_mapping)
 
    VOL = Vector{NTuple{d,LinearMap}}(undef,N_e)
    FAC = Vector{LinearMap}(undef,N_e)
    n_f = Array{Float64,3}(undef, d, N_f, N_e)

    @inbounds for k in 1:N_e
        M⁻¹ = mass_matrix_inverse(mass_solver, k)
        VOL[k] = Tuple(make_operator(M⁻¹ * Matrix(V' * 
            (sum(D[m]' * Diagonal(0.5 * W * Λ_q[:,m,n,k]) -
            Diagonal(0.5 * W * Λ_q[:,m,n,k]) * D[m] for m in 1:d) +
            R' * Diagonal(0.5 * B * nJf[n,:,k]) * R)), alg) for n in 1:d)
        FAC[k] = make_operator(-M⁻¹ * 
            Matrix(V' * R' * Diagonal(B * J_f[:,k])), alg)
        @inbounds for m in 1:d
            n_f[m,:,k] = nJf[m,:,k] ./ J_f[:,k]
        end
    end

    return PhysicalOperators(
        VOL, FAC, make_operator(V, alg), make_operator(R, alg), n_f)
end


function flux_differencing_operators(
    reference_approximation::ReferenceApproximation{<:RefElemData{1}, 
    <:AbstractTensorProduct})

    (; D, W, R, B) = reference_approximation

    S = (0.5*Matrix(W*D[1] - D[1]'*W),)
    C = Matrix(R')*Matrix(B)

    return S, C
end

function flux_differencing_operators(
    reference_approximation::ReferenceApproximation{<:RefElemData{d}, 
    <:AbstractTensorProduct}) where {d}

    (; D, W, R, B, reference_mapping) = reference_approximation

    D_ξ = reference_derivative_operators(D, reference_mapping)

    S = Tuple(0.5*Matrix(W*D_ξ[m] - D_ξ[m]'*W) for m in 1:d)
    C = Matrix(R')*Matrix(B)
    
    return Tuple(sparse(S[m]) for m in 1:d), sparse(C)
end

function flux_differencing_operators(
    reference_approximation::ReferenceApproximation{<:RefElemData{d}, 
    <:AbstractMultidimensional}) where {d}

    (; D, W, R, B, reference_mapping) = reference_approximation

    D_ξ = reference_derivative_operators(D, reference_mapping)
    S = Tuple(0.5*Matrix(W*D_ξ[m] - D_ξ[m]'*W) for m in 1:d)
    C = Matrix(R')*Matrix(B)
    
    return S, C
end