function ReferenceApproximation(
    approx_type::NodalTensor, element_type::Line,
    mapping_degree::Int=1, N_plot::Int=10,
    volume_quadrature_rule=LGQuadrature(approx_type.p))

    reference_element = RefElemData(Line(), mapping_degree,
        quad_rule_vol=quadrature(Line(), 
        volume_quadrature_rule), Nplot=N_plot)

    @unpack rp, rq, rf, wq, wf = reference_element
    q = length(rq)-1
    VDM, ∇VDM = basis(Line(), q, rq)

    if volume_quadrature_rule isa LGLQuadrature
        R = SelectionMap(facet_node_ids(Line(),q+1),q+1)
    else R = LinearMap(vandermonde(element_type, q, rf) / VDM) end

    V_plot = LinearMap(vandermonde(element_type, q, rp) / VDM)

    return ReferenceApproximation(NodalTensor(q), q+1, q+1, 2, 
        reference_element, (LinearMap(∇VDM / VDM),), LinearMap(I, p+1), R, R,
        Diagonal(wq), Diagonal(wf), V_plot, NoMapping())
end

function ReferenceApproximation(approx_type::NodalTensor, 
    element_type::Quad; mapping_degree::Int=1, N_plot::Int=10,
    volume_quadrature_rule=LGQuadrature(approx_type.p),
    facet_quadrature_rule=LGQuadrature(approx_type.p))

    # one-dimensional operators
    nodes_1D = quadrature(Line(),volume_quadrature_rule)[1]
    q = length(nodes_1D)-1
    VDM_1D, ∇VDM_1D = basis(Line(), q, nodes_1D)
    R_L = vandermonde(Line(), q, [-1.0]) / VDM_1D
    R_R = vandermonde(Line(), q, [1.0]) / VDM_1D

    # differentiation matrix
    σ = [(q+1)*(i-1) + j for i in 1:q+1, j in 1:q+1]
    D = (TensorProductMap2D(∇VDM_1D / VDM_1D, I, σ, σ),
            TensorProductMap2D(I, ∇VDM_1D / VDM_1D, σ, σ))

    reference_element = RefElemData(element_type, mapping_degree,
        quad_rule_vol=quadrature(Quad(), volume_quadrature_rule),
        quad_rule_face=quadrature(Line(), facet_quadrature_rule), Nplot=N_plot)

    @unpack rstp, rstq, rstf, wq, wf = reference_element

    # extrapolation operators
    if volume_quadrature_rule == facet_quadrature_rule
        if volume_quadrature_rule isa LGLQuadrature
            R = SelectionMap(match_coordinate_vectors(rstf, rstq), (q+1)^2)
        else
            R =[TensorProductMap2D(R_L, I, σ, [j for i in 1:1, j in 1:q+1]); #L
                TensorProductMap2D(R_R, I, σ, [j for i in 1:1, j in 1:q+1]); #R
                TensorProductMap2D(I, R_L, σ, [i for i in 1:q+1, j in 1:1]); #B
                TensorProductMap2D(I, R_R ,σ, [i for i in 1:q+1, j in 1:1])] #T
        end
    else
        R = LinearMap(vandermonde(element_type,q,rstf...) / 
            vandermonde(element_type,q,rstq...))
    end
    
    V_plot = LinearMap(vandermonde(element_type, q, rstp...) / 
        vandermonde(element_type, q, rstq...))

    return ReferenceApproximation(NodalTensor(q), (q+1)^2, (q+1)^2, 4*(q+1), 
        reference_element, D, LinearMap(I, (q+1)^2), R, R,
        Diagonal(wq), Diagonal(wf), V_plot, NoMapping())
end

function ReferenceApproximation(approx_type::NodalTensor, ::Hex;
    mapping_degree::Int=1, N_plot::Int=10,
    volume_quadrature_rule=LGQuadrature(approx_type.p), facet_quadrature_rule=LGQuadrature(approx_type.p))

    # one-dimensional operators
    nodes_1D = quadrature(Line(),volume_quadrature_rule)[1]
    q = length(nodes_1D)-1
    VDM_1D, ∇VDM_1D = basis(Line(), q, nodes_1D)
    D_1D = ∇VDM_1D / VDM_1D

    # differentiation matrix
    σ = permutedims(reshape(collect(1:(q+1)^3),q+1,q+1,q+1), [3,2,1])
    D = (TensorProductMap3D(D_1D, I, I, σ, σ),
         TensorProductMap3D(I, D_1D, I, σ, σ),
         TensorProductMap3D(I, I, D_1D, σ, σ))

    reference_element = RefElemData(Hex(), mapping_degree,
        quad_rule_vol=quadrature(Hex(), volume_quadrature_rule),
        quad_rule_face=quadrature(Quad(), facet_quadrature_rule),
        Nplot=N_plot)

    @unpack rstp, rstq, rstf, wq, wf = reference_element

    # extrapolation operators
    if (volume_quadrature_rule isa LGLQuadrature && 
            facet_quadrature_rule isa LGLQuadrature)
        R = SelectionMap(match_coordinate_vectors(rstf,rstq),(q+1)^3)
    else 
        R = LinearMap(vandermonde(Hex(),q,rstf...) / 
            vandermonde(Hex(),q,rstq...))
    end
    
    V_plot = LinearMap(vandermonde(Hex(), q, rstp...) / 
        vandermonde(Hex(), q, rstq...))

    return ReferenceApproximation(approx_type, (q+1)^3, (q+1)^3, 6*(q+1)^2, 
        reference_element, D, LinearMap(I, (q+1)^3), R, R,
        Diagonal(wq), Diagonal(wf), V_plot, NoMapping())
end