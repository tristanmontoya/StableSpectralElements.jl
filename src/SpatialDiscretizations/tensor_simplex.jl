# Duffy transform from the square to triangle
@inline function χ(::Tri, η::Union{NTuple{2, Float64}, NTuple{2, Vector{Float64}}})
    return (0.5 .* (1.0 .+ η[1]) .* (1.0 .- η[2]) .- 1.0, η[2])
end

# Duffy transform from the cube to tetrahedron
@inline function χ(::Tet, η::Union{NTuple{3, Float64}, NTuple{3, Vector{Float64}}})
    ξ_pri = (0.5 .* (1.0 .+ η[1]) .* (1.0 .- η[3]) .- 1.0, η[2], η[3])
    ξ_pyr = (ξ_pri[1], 0.5 .* (1.0 .+ η[2]) .* (1.0 .- η[3]) .- 1.0, ξ_pri[3])
    return (0.5 .* (1.0 .+ ξ_pri[1]) .* (1.0 .- η[2]) .- 1.0, ξ_pyr[2], ξ_pyr[3])
end

function reference_geometric_factors(::Tri,
        quadrature_rule::NTuple{2, AbstractQuadratureRule})
    η = quadrature(Quad(), quadrature_rule)
    N = size(η[1], 1)
    Λ_ref = Array{Float64, 3}(undef, N, 2, 2)

    if ((quadrature_rule[1].a, quadrature_rule[1].b) == (0, 0) &&
        (quadrature_rule[2].a, quadrature_rule[2].b) == (0, 0))
        J_ref = (x -> 0.5 * (1.0 - x)).(η[2])
        Λ_ref[:, 1, 1] = ones(N) # Jdη1/dξ1
        Λ_ref[:, 1, 2] = (x -> 0.5 * (1.0 + x)).(η[1]) # Jdη1/dξ2
        Λ_ref[:, 2, 1] = zeros(N) # Jdη2/dξ1
        Λ_ref[:, 2, 2] = (x -> 0.5 * (1.0 - x)).(η[2]) # Jdη2/dξ2

    elseif ((quadrature_rule[1].a, quadrature_rule[1].b) == (0, 0) &&
            (quadrature_rule[2].a, quadrature_rule[2].b) == (1, 0))
        J_ref = 0.5 * ones(N)
        Λ_ref[:, 1, 1] = (x -> 1.0 / (1.0 - x)).(η[2]) # Jdη1/dξ1
        Λ_ref[:, 1, 2] = (x -> 0.5 * (1.0 + x)).(η[1]) .* (x -> 1.0 / (1.0 - x)).(η[2]) # Jdη1/dξ2
        Λ_ref[:, 2, 1] = zeros(N) # Jdη2/dξ1
        Λ_ref[:, 2, 2] = 0.5 * ones(N) # Jdη2/dξ2

    else
        @error "Chosen Jacobi weight not supported"
    end

    return J_ref, Λ_ref
end

function reference_geometric_factors(::Tet,
        quadrature_rule::NTuple{3, AbstractQuadratureRule})
    η = quadrature(Hex(), quadrature_rule)
    N = size(η[1], 1)
    Λ_ref = Array{Float64, 3}(undef, N, 3, 3)

    if ((quadrature_rule[1].a, quadrature_rule[1].b) == (0, 0) &&
        (quadrature_rule[2].a, quadrature_rule[2].b) == (0, 0) &&
        (quadrature_rule[3].a, quadrature_rule[3].b) == (0, 0))
        J_ref = (x -> 0.5 * (1.0 - x)).(η[2]) .* (x -> (0.5 * (1.0 - x))^2).(η[3])
        Λ_ref[:, 1, 1] = (x -> 0.5 * (1.0 - x)).(η[3]) # Jdη1/dξ1
        Λ_ref[:, 1, 2] = (x -> 0.5 * (1.0 + x)).(η[1]) .* (x -> 0.5 * (1.0 - x)).(η[3])  # Jdη1/dξ2
        Λ_ref[:, 1, 3] = (x -> 0.5 * (1.0 + x)).(η[1]) .* (x -> 0.5 * (1.0 - x)).(η[3])  # Jdη1/dξ3
        Λ_ref[:, 2, 1] = zeros(N) # Jdη2/dξ1
        Λ_ref[:, 2, 2] = (x -> 0.5 * (1.0 - x)).(η[2]) .* (x -> 0.5 * (1.0 - x)).(η[3]) # Jdη2/dξ2
        Λ_ref[:, 2, 3] = (x -> 0.5 * (1.0 + x)).(η[2]) .* (x -> 0.5 * (1.0 - x)).(η[2]) .*
                         (x -> 0.5 * (1.0 - x)).(η[3]) # Jdη2/dξ3
        Λ_ref[:, 3, 1] = zeros(N) # Jdη3/dξ1
        Λ_ref[:, 3, 2] = zeros(N) # Jdη3/dξ2
        Λ_ref[:, 3, 3] = (x -> 0.5 * (1.0 - x)).(η[2]) .* (x -> (0.5 * (1.0 - x))^2).(η[3])# Jdη3/dξ3

    elseif ((quadrature_rule[1].a, quadrature_rule[1].b) == (0, 0) &&
            (quadrature_rule[2].a, quadrature_rule[2].b) == (0, 0) &&
            (quadrature_rule[3].a, quadrature_rule[3].b) == (1, 0))
        J_ref = 0.125 * (x -> (1.0 - x)).(η[2]) .* (x -> (1.0 - x)).(η[3])
        Λ_ref[:, 1, 1] = 0.5 * ones(N) # Jdη1/dξ1
        Λ_ref[:, 1, 2] = 0.25 * (x -> (1.0 + x)).(η[1])  # Jdη1/dξ2
        Λ_ref[:, 1, 3] = 0.25 * (x -> (1.0 + x)).(η[1]) # Jdη1/dξ3
        Λ_ref[:, 2, 1] = zeros(N) # Jdη2/dξ1
        Λ_ref[:, 2, 2] = 0.25 * (x -> (1.0 - x)).(η[2]) # Jdη2/dξ2
        Λ_ref[:, 2, 3] = 0.125 * (x -> (1.0 + x)).(η[2]) .* (x -> (1.0 - x)).(η[2])  # Jdη2/dξ3
        Λ_ref[:, 3, 1] = zeros(N) # Jdη3/dξ1
        Λ_ref[:, 3, 2] = zeros(N) # Jdη3/dξ2
        Λ_ref[:, 3, 3] = 0.125 * (x -> (1.0 - x)).(η[2]) .* (x -> (1.0 - x)).(η[3])# Jdη3/dξ3

    else
        @error "Chosen Jacobi weight not supported"
    end

    return J_ref, Λ_ref
end

function warped_product(::Tri, p::Int, η1D::NTuple{2, Vector{Float64}})
    (M1, M2) = (length(η1D[1]), length(η1D[2]))
    σₒ = [M2 * (i - 1) + j for i in 1:M1, j in 1:M2]
    σᵢ = zeros(Int, p + 1, p + 1)
    A = zeros(M1, p + 1)
    B = zeros(M2, p + 1, p + 1)

    k = 1
    @inbounds for i in 0:p
        for j in 0:(p - i)
            σᵢ[i + 1, j + 1] = k
            k = k + 1
            for α1 in 1:M1, α2 in 1:M2
                A[α1, i + 1] = sqrt(2) * jacobiP(η1D[1][α1], 0, 0, i)
                B[α2, i + 1, j + 1] = (1 - η1D[2][α2])^i * jacobiP(η1D[2][α2], 2i + 1, 0, j)
            end
        end
    end

    return WarpedTensorProductMap2D(SArray{Tuple{M1, p + 1}}(A),
        SArray{Tuple{M1, p + 1, p + 1}}(B),
        SArray{Tuple{M1, M2}}(σᵢ),
        SArray{Tuple{p + 1, p + 1}}(σₒ))
end

function warped_product(::Tet, p::Int, η1D::NTuple{3, Vector{Float64}})
    (M1, M2, M3) = (length(η1D[1]), length(η1D[2]), length(η1D[3]))
    σₒ = [M2 * M3 * (i - 1) + M3 * (j - 1) + k for i in 1:M1, j in 1:M2, k in 1:M3]
    σᵢ = zeros(Int, p + 1, p + 1, p + 1)
    A = zeros(M1, p + 1)
    B = zeros(M2, p + 1, p + 1)
    C = zeros(M3, p + 1, p + 1, p + 1)

    l = 1
    @inbounds for i in 0:p
        for j in 0:(p - i)
            for k in 0:(p - i - j)
                σᵢ[i + 1, j + 1, k + 1] = l
                l = l + 1
                for α1 in 1:M1, α2 in 1:M2, α3 in 1:M3
                    A[α1, i + 1] = sqrt(2) * jacobiP(η1D[1][α1], 0, 0, i)
                    B[α2, i + 1, j + 1] = (1 - η1D[2][α2])^i *
                                          jacobiP(η1D[2][α2], 2i + 1, 0, j)
                    C[α3, i + 1, j + 1, k + 1] = 2 *
                                                 (1 - η1D[3][α3])^(i + j) *
                                                 jacobiP(η1D[3][α3], 2i + 2j + 2, 0, k)
                end
            end
        end
    end

    return WarpedTensorProductMap3D(SArray{Tuple{M1, p + 1}}(A),
        SArray{Tuple{M1, p + 1, p + 1}}(B),
        SArray{Tuple{M1, p + 1, p + 1, p + 1}}(C),
        SArray{Tuple{M1, M2, M3}}(σᵢ),
        SArray{Tuple{p + 1, p + 1, p + 1}}(σₒ))
end

function operators_1d(quadrature_rule::NTuple{d, AbstractQuadratureRule}) where {d}
    η_1D, q, V_1D, D_1D, I_1D, R_L, R_R = fill((), 7)
    for m in 1:d
        η, _ = quadrature(Line(), quadrature_rule[m])
        η_1D = (η_1D..., η)
        q = (q..., length(η_1D[m]) - 1)
        V_1D = (V_1D..., vandermonde(Line(), q[m], η_1D[m]))
        D_1D = (D_1D..., OctavianMap(grad_vandermonde(Line(), q[m], η_1D[m]) / V_1D[m]))
        I_1D = (I_1D..., LinearMap(I, q[m] + 1))
        R_L = (R_L..., OctavianMap(vandermonde(Line(), q[m], [-1.0]) / V_1D[m]))
        R_R = (R_R..., OctavianMap(vandermonde(Line(), q[m], [1.0]) / V_1D[m]))
    end

    return η_1D, q, V_1D, D_1D, I_1D, R_L, R_R
end

function operators_1d(quadrature_rule::AbstractQuadratureRule)
    η_1D, q, V_1D, D_1D, I_1D, R_L, R_R = fill((), 7)
    η, _ = quadrature(Line(), quadrature_rule)
    η_1D = η
    q = length(η_1D) - 1
    V_1D = vandermonde(Line(), q, η_1D)
    D_1D = OctavianMap(grad_vandermonde(Line(), q, η_1D) / V_1D)
    I_1D = LinearMap(I, q + 1)
    R_L = OctavianMap(vandermonde(Line(), q, [-1.0]) / V_1D)
    R_R = OctavianMap(vandermonde(Line(), q, [1.0]) / V_1D)

    return η_1D, q, V_1D, D_1D, I_1D, R_L, R_R
end

function ReferenceApproximation(approx_type::AbstractTensorProduct,
        ::Tri;
        mapping_degree::Int = 1,
        N_plot::Int = 10,
        volume_quadrature_rule = (LGQuadrature(approx_type.p),
            LGQuadrature(approx_type.p)),
        facet_quadrature_rule = LGQuadrature(approx_type.p),
        sum_factorize_vandermonde = true)

    # one-dimensional operators
    η_1D, q, V_1D, D_1D, I_1D, R_L, R_R = operators_1d(volume_quadrature_rule)

    # geometric factors for collapsed coordinate transformation
    J_ref, Λ_ref = reference_geometric_factors(Tri(), volume_quadrature_rule)

    # one-dimensional facet quadrature rule
    η_f, _ = quadrature(Line(), facet_quadrature_rule)

    # interpolation/extrapolation operators
    if volume_quadrature_rule[1] == facet_quadrature_rule
        η1_to_ηf = LinearMap(I, q[1] + 1)
    else
        η1_to_ηf = OctavianMap(vandermonde(Line(), q[1], η_f) / V_1D[1])
    end
    if volume_quadrature_rule[2] == facet_quadrature_rule
        η2_to_ηf = LinearMap(I, q[2] + 1)
    else
        η2_to_ηf = OctavianMap(vandermonde(Line(), q[2], η_f) / V_1D[2])
    end
    R = [η1_to_ηf ⊗ R_L[2]; R_R[1] ⊗ η2_to_ηf; R_L[1] ⊗ η2_to_ηf]

    # reference element data (mainly used for mapping, normals, etc.)
    reference_element = RefElemData(Tri(),
        approx_type,
        mapping_degree,
        volume_quadrature_rule = volume_quadrature_rule,
        facet_quadrature_rule = facet_quadrature_rule,
        Nplot = N_plot)
    
    (; rstq, rstf, rstp, wq) = reference_element

    # construct nodal or modal scheme (different Vandermonde matrix)
    if approx_type isa ModalTensor
        if sum_factorize_vandermonde
            V = warped_product(Tri(), approx_type.p, η_1D)
        else
            V = OctavianMap(vandermonde(Tri(), approx_type.p, reference_element.rstq...))
        end
        V_plot = OctavianMap(vandermonde(Tri(), approx_type.p, reference_element.rstp...))
    else
        V = LinearMap(I, (q[1] + 1) * (q[2] + 1))
        V_plot = (vandermonde(Line(), q[1], equi_nodes(Line(), N_plot)) / V_1D[1]) ⊗
                 (vandermonde(Line(), q[2], equi_nodes(Line(), N_plot)) / V_1D[2])
    end

    return ReferenceApproximation(approx_type,
        reference_element,
        (D_1D[1] ⊗ I_1D[2], I_1D[1] ⊗ D_1D[2]),
        V,
        R * V,
        R,
        V_plot,
        ReferenceMapping(J_ref, Λ_ref))
end

function ReferenceApproximation(approx_type::AbstractTensorProduct,
        ::Tet;
        mapping_degree::Int = 1,
        N_plot::Int = 10,
        volume_quadrature_rule = (LGQuadrature(approx_type.p),
            LGQuadrature(approx_type.p),
            GaussQuadrature(approx_type.p, 1,
                0)),
        facet_quadrature_rule = (LGQuadrature(approx_type.p),
            GaussQuadrature(approx_type.p, 1,
                0)),
        sum_factorize_vandermonde = true)

    # one-dimensional operators
    η_1D, q, V_1D, D_1D, I_1D, R_L, R_R = operators_1d(volume_quadrature_rule)

    # reference geometric factors for cube-to-tetrahedron mapping
    J_ref, Λ_ref = reference_geometric_factors(Tet(), volume_quadrature_rule)

    # two-dimensional facet quadrature nodes and weights
    η_f1, _ = quadrature(Line(), facet_quadrature_rule[1])
    η_f2, _ = quadrature(Line(), facet_quadrature_rule[2])

    # interpolation/extrapolation operators
    if volume_quadrature_rule[1] == facet_quadrature_rule[1]
        η1_to_ηf1 = LinearMap(I, q[1] + 1)
    else
        η1_to_ηf1 = OctavianMap(vandermonde(Line(), q[1], η_f1) / V_1D[1])
    end
    if volume_quadrature_rule[2] == facet_quadrature_rule[1]
        η2_to_ηf1 = LinearMap(I, q[2] + 1)
    else
        η2_to_ηf1 = OctavianMap(vandermonde(Line(), q[2], η_f1) / V_1D[2])
    end
    if volume_quadrature_rule[2] == facet_quadrature_rule[2]
        η2_to_ηf2 = LinearMap(I, q[2] + 1)
    else
        η2_to_ηf2 = OctavianMap(vandermonde(Line(), q[2], η_f2) / V_1D[2])
    end
    if volume_quadrature_rule[3] == facet_quadrature_rule[2]
        η3_to_ηf2 = LinearMap(I, q[3] + 1)
    else
        η3_to_ηf2 = OctavianMap(vandermonde(Line(), q[3], η_f2) / V_1D[3])
    end
    R = [η1_to_ηf1 ⊗ R_L[2] ⊗ η3_to_ηf2
         R_R[1] ⊗ η2_to_ηf1 ⊗ η3_to_ηf2
         R_L[1] ⊗ η2_to_ηf1 ⊗ η3_to_ηf2
         η1_to_ηf1 ⊗ η2_to_ηf2 ⊗ R_L[3]]

    # reference element data (mainly used for mapping, normals, etc.)
    reference_element = RefElemData(Tet(),
        approx_type,
        mapping_degree,
        volume_quadrature_rule = volume_quadrature_rule,
        facet_quadrature_rule = facet_quadrature_rule,
        Nplot = N_plot)

    # construct nodal or modal scheme
    if approx_type isa ModalTensor
        if sum_factorize_vandermonde
            V = warped_product(Tet(), approx_type.p, η_1D)
        else
            V = OctavianMap(vandermonde(Tet(), approx_type.p, reference_element.rstq...))
        end
        V_plot = OctavianMap(vandermonde(Tet(), approx_type.p, reference_element.rstp...))
    else
        V = LinearMap(I, (q[1] + 1) * (q[2] + 1) * (q[3] + 1))
        V_plot = OctavianMap(vandermonde(Line(), q[1], equi_nodes(Line(), N_plot)) /
                             V_1D[1]) ⊗
                 OctavianMap(vandermonde(Line(), q[2], equi_nodes(Line(), N_plot)) /
                             V_1D[2]) ⊗
                 OctavianMap(vandermonde(Line(), q[3], equi_nodes(Line(), N_plot)) /
                             V_1D[3])
    end

    return ReferenceApproximation(approx_type,
        reference_element,
        (D_1D[1] ⊗ I_1D[2] ⊗ I_1D[3],
            I_1D[1] ⊗ D_1D[2] ⊗ I_1D[3],
            I_1D[1] ⊗ I_1D[2] ⊗ D_1D[3]),
        V,
        R * V,
        R,
        V_plot,
        ReferenceMapping(J_ref, Λ_ref))
end
