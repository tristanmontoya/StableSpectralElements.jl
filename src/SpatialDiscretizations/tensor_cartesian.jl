function ReferenceApproximation(approx_type::NodalTensor,
        element_type::Line;
        mapping_degree::Int = 1,
        N_plot::Int = 10,
        volume_quadrature_rule = LGLQuadrature(approx_type.p))
    reference_element = RefElemData(Line(),
        Polynomial{MultidimensionalQuadrature}(),
        mapping_degree,
        quad_rule_vol = quadrature(Line(),
            volume_quadrature_rule),
        Nplot = N_plot)

    (; rp, rq, rstq, rf, rstf) = reference_element
    q = length(rq) - 1
    VDM, ∇VDM = basis(Line(), q, rq)

    if volume_quadrature_rule isa GaussLobattoQuadrature
        R = SelectionMap(match_coordinate_vectors(rstf, rstq), q + 1)
    else
        R = OctavianMap(vandermonde(element_type, q, rf) / VDM)
    end

    V_plot = OctavianMap(vandermonde(element_type, q, rp) / VDM)

    return ReferenceApproximation(NodalTensor(q),
        reference_element,
        (OctavianMap(∇VDM / VDM),),
        LinearMap(I, q + 1),
        R,
        R,
        V_plot)
end

function ReferenceApproximation(approx_type::NodalTensor,
        element_type::Quad;
        mapping_degree::Int = 1,
        N_plot::Int = 10,
        volume_quadrature_rule = LGLQuadrature(approx_type.p),
        facet_quadrature_rule = LGLQuadrature(approx_type.p))

    # one-dimensional operators
    nodes_1D = quadrature(Line(), volume_quadrature_rule)[1]
    q = length(nodes_1D) - 1
    VDM_1D, ∇VDM_1D = basis(Line(), q, nodes_1D)
    D_1D = OctavianMap(∇VDM_1D / VDM_1D)
    I_1D = LinearMap(I, q + 1)
    R_L = OctavianMap(vandermonde(Line(), q, [-1.0]) / VDM_1D)
    R_R = OctavianMap(vandermonde(Line(), q, [1.0]) / VDM_1D)

    # reference element data
    reference_element = RefElemData(element_type,
        Polynomial{MultidimensionalQuadrature}(),
        mapping_degree,
        quad_rule_vol = quadrature(Quad(),
            volume_quadrature_rule),
        quad_rule_face = quadrature(Line(),
            facet_quadrature_rule),
        Nplot = N_plot)

    (; rstp, rstq, rstf) = reference_element

    # extrapolation operators
    if volume_quadrature_rule == facet_quadrature_rule
        if volume_quadrature_rule isa GaussLobattoQuadrature
            R = SelectionMap(match_coordinate_vectors(rstf, rstq), (q + 1)^2)
        else
            R = [R_L ⊗ I_1D; R_R ⊗ I_1D; I_1D ⊗ R_L; I_1D ⊗ R_R]
        end
    else
        R = OctavianMap(vandermonde(element_type, q, rstf...) /
                        vandermonde(element_type, q, rstq...))
    end

    V_plot = OctavianMap(vandermonde(element_type, q, rstp...) /
                         vandermonde(element_type, q, rstq...))

    return ReferenceApproximation(NodalTensor(q),
        reference_element,
        (D_1D ⊗ I_1D, I_1D ⊗ D_1D),
        LinearMap(I, (q + 1)^2),
        R,
        R,
        V_plot)
end

function ReferenceApproximation(approx_type::NodalTensor,
        ::Hex;
        mapping_degree::Int = 1,
        N_plot::Int = 10,
        volume_quadrature_rule = LGLQuadrature(approx_type.p),
        facet_quadrature_rule = LGLQuadrature(approx_type.p))

    # one-dimensional operators
    nodes_1D = quadrature(Line(), volume_quadrature_rule)[1]
    q = length(nodes_1D) - 1
    VDM_1D, ∇VDM_1D = basis(Line(), q, nodes_1D)
    D_1D = OctavianMap(∇VDM_1D / VDM_1D)
    I_1D = LinearMap(I, q + 1)

    # reference element data
    reference_element = RefElemData(Hex(),
        Polynomial{MultidimensionalQuadrature}(),
        mapping_degree,
        quad_rule_vol = quadrature(Hex(),
            volume_quadrature_rule),
        quad_rule_face = quadrature(Quad(),
            facet_quadrature_rule),
        Nplot = N_plot)

    (; rstp, rstq, rstf) = reference_element

    # extrapolation operators
    if (volume_quadrature_rule == facet_quadrature_rule) &&
       (volume_quadrature_rule isa GaussLobattoQuadrature)
        R = SelectionMap(match_coordinate_vectors(rstf, rstq), (q + 1)^3)
    else
        R = OctavianMap(vandermonde(Hex(), q, rstf...) / vandermonde(Hex(), q, rstq...))
    end

    # mapping from solution/quadrature to plotting nodes
    V_plot = OctavianMap(vandermonde(Hex(), q, rstp...) / vandermonde(Hex(), q, rstq...))

    return ReferenceApproximation(NodalTensor(q),
        reference_element,
        (D_1D ⊗ I_1D ⊗ I_1D, I_1D ⊗ D_1D ⊗ I_1D,
            I_1D ⊗ I_1D ⊗ D_1D),
        LinearMap(I, (q + 1)^3),
        R,
        R,
        V_plot)
end
