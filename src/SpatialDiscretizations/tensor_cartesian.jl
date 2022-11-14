function ReferenceApproximation(
    approx_type::NodalTensor, element_type::Line; 
    volume_quadrature_rule::AbstractQuadratureRule=LGLQuadrature(),
    mapping_degree::Int=1, N_plot::Int=10)

    @unpack p = approx_type

    reference_element = RefElemData(Line(), mapping_degree,
        quad_rule_vol=quadrature(Line(), 
        volume_quadrature_rule, p+1), Nplot=N_plot)

    @unpack rstp, rstq, rstf, wq, wf = reference_element

    VDM, ∇VDM = basis(Line(), p, rstq[1])

    if volume_quadrature_rule isa LGLQuadrature
        R = SelectionMap(facet_node_ids(Line(),p+1),p+1)
    else R = LinearMap(vandermonde(element_type,p,rstf[1]) / VDM) end

    V_plot = LinearMap(vandermonde(element_type, p, rstp[1]) / VDM)

    return ReferenceApproximation(approx_type, p+1, p+1, 2, reference_element,
    (LinearMap(∇VDM / VDM),), LinearMap(I, (p+1)), R, R, Diagonal(wq),
        Diagonal(wf), V_plot, NoMapping())
end

function ReferenceApproximation(approx_type::NodalTensor, 
    element_type::Quad;
    volume_quadrature_rule::AbstractQuadratureRule=LGLQuadrature(),
    facet_quadrature_rule::AbstractQuadratureRule=LGLQuadrature(),
    mapping_degree::Int=1, N_plot::Int=10)

    @unpack p = approx_type

    # one-dimensional operators
    nodes_1D = quadrature(Line(),volume_quadrature_rule,p+1)[1]
    VDM_1D, ∇VDM_1D = basis(Line(), p, nodes_1D)
    R_L = vandermonde(Line(),p, [-1.0]) / VDM_1D
    R_R = vandermonde(Line(),p, [1.0]) / VDM_1D

    # differentiation matrix
    σ = [(p+1)*(i-1) + j for i in 1:p+1, j in 1:p+1]
    D = (TensorProductMap2D(∇VDM_1D / VDM_1D, I, σ, σ),
            TensorProductMap2D(I, ∇VDM_1D / VDM_1D, σ, σ))

    reference_element = RefElemData(element_type, mapping_degree,
        quad_rule_vol=quadrature(element_type, volume_quadrature_rule, p+1),
        quad_rule_face=quadrature(face_type(element_type), 
            facet_quadrature_rule, p+1), Nplot=N_plot)
    @unpack rstp, rstq, rstf, wq, wf = reference_element

    # extrapolation operators
    if (volume_quadrature_rule isa LGLQuadrature && 
            facet_quadrature_rule isa LGLQuadrature)
        R = SelectionMap(facet_node_ids(Quad(),(p+1,p+1)),(p+1)^2)
    elseif typeof(volume_quadrature_rule) == typeof(facet_quadrature_rule)
        R =[TensorProductMap2D(R_L, I, σ, [j for i in 1:1, j in 1:p+1]); #L
            TensorProductMap2D(R_R, I, σ, [j for i in 1:1, j in 1:p+1]); #R
            TensorProductMap2D(I, R_L, σ, [i for i in 1:p+1, j in 1:1]); #B
            TensorProductMap2D(I, R_R ,σ, [i for i in 1:p+1, j in 1:1])] #T
    else
        R = LinearMap(vandermonde(element_type,p,rstf...) / 
            vandermonde(element_type,p,rstq...))
    end
    
    V_plot = LinearMap(vandermonde(element_type, p, rstp...) / 
        vandermonde(element_type, p, rstq...))

    return ReferenceApproximation(approx_type, (p+1)^2, (p+1)^2, 4*(p+1), 
        reference_element, D, LinearMap(I, (p+1)^2), R, R,
        Diagonal(wq), Diagonal(wf), V_plot, NoMapping())
end