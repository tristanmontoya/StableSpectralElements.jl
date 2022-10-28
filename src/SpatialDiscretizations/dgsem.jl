struct DGSEM <:AbstractApproximationType
    p::Int  # polynomial degree
end

function ReferenceApproximation(
    approx_type::DGSEM, element_type::Line; 
    volume_quadrature_rule::AbstractQuadratureRule=LGLQuadrature(),
    mapping_degree::Int=1, N_plot::Int=10,
    operator_algorithm::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm())

    @unpack p = approx_type
    N_p = p+1
    N_q = p+1
    N_f = 2

    reference_element = RefElemData(Line(), mapping_degree,
        quad_rule_vol=quadrature(Line(), 
        volume_quadrature_rule, p+1), Nplot=N_plot)
    @unpack rstp, rstq, rstf, wq, wf = reference_element
    VDM, ∇VDM = basis(Line(), p, rstq[1])
    D = (make_operator(LinearMap(∇VDM / VDM), operator_algorithm),)

    if volume_quadrature_rule isa LGLQuadrature
        Vf = SelectionMap(facet_node_ids(Line(),p+1),p+1)
    else
        Vf = make_operator(LinearMap(vandermonde(element_type,p,rstf[1]) / VDM),
            operator_algorithm)
    end

    V_plot = LinearMap(vandermonde(element_type, p, rstp[1]) / VDM)
    V = LinearMap(I, N_q)
    R = Vf
    W = Diagonal(wq)
    B = Diagonal(wf)

    return ReferenceApproximation(approx_type, N_p, N_q, N_f, 
        reference_element, D, V, Vf, R, W, B, V_plot, NoMapping())
end

function ReferenceApproximation(approx_type::DGSEM, 
    element_type::Quad;
    volume_quadrature_rule::AbstractQuadratureRule=LGLQuadrature(),
    facet_quadrature_rule::AbstractQuadratureRule=LGLQuadrature(),
    mapping_degree::Int=1, N_plot::Int=10,
    operator_algorithm::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm())

    @unpack p = approx_type

    # dimensions of operators
    N_p = (p+1)^2
    N_q = (p+1)^2
    N_f = 4*(p+1)

    reference_element = RefElemData(element_type, mapping_degree,
        quad_rule_vol=quadrature(element_type, volume_quadrature_rule, p+1),
        quad_rule_face=quadrature(face_type(element_type), 
            facet_quadrature_rule, p+1), 
        Nplot=N_plot)

    @unpack rstp, rstq, rstf, wq, wf = reference_element

    # one-dimensional operators
    ref_elem_1D = RefElemData(Line(), mapping_degree,
        quad_rule_vol=quadrature(Line(), 
            volume_quadrature_rule, p+1), Nplot=N_plot)
    VDM_1D, ∇VDM_1D = basis(Line(), p, ref_elem_1D.rstq[1])
    D_1D = ∇VDM_1D / VDM_1D
    R_1D = vandermonde(Line(),p, ref_elem_1D.rstf[1]) / VDM_1D
    R_L = R_1D[1:1,:]
    R_R = R_1D[2:2,:]

    # scalar ordering of multi-indices
    sigma = [(p+1)*(i-1) + j for i in 1:p+1, j in 1:p+1]

    # extrapolation operators
    if (volume_quadrature_rule isa LGLQuadrature && 
            facet_quadrature_rule isa LGLQuadrature)
        Vf = SelectionMap(facet_node_ids(Quad(),(p+1,p+1)),N_p)
    elseif typeof(volume_quadrature_rule) == typeof(facet_quadrature_rule)
        Vf = make_operator([
            TensorProductMap(R_L, I, sigma, [j for i in 1:1, j in 1:p+1]) ; #L
            TensorProductMap(R_R, I, sigma, [j for i in 1:1, j in 1:p+1]) ; #R
            TensorProductMap(I, R_L, sigma, [i for i in 1:p+1, j in 1:1]) ; #B
            TensorProductMap(I, R_R ,sigma, [i for i in 1:p+1, j in 1:1])], #T
            operator_algorithm)
    else
        Vf = make_operator(LinearMap(vandermonde(element_type,p,rstf...) / 
            vandermonde(element_type,p,rstq...)), operator_algorithm)
    end

    V_plot = LinearMap(vandermonde(element_type, p, rstp...) / 
        vandermonde(element_type, p, rstq...))
    
    D = (make_operator(TensorProductMap(D_1D, I, sigma, sigma),             
            operator_algorithm),
        make_operator(TensorProductMap(I, D_1D, sigma, sigma),
            operator_algorithm))

    V = LinearMap(I, N_q)
    R = Vf
    W =Diagonal(wq)
    B = Diagonal(wf)

    return ReferenceApproximation(approx_type, N_p, N_q, N_f, 
        reference_element, D, V, Vf, R, W, B, V_plot, NoMapping())
end