struct DGSEM <:AbstractCollocatedApproximation
    p::Int  # polynomial degree
end

function ReferenceApproximation(approx_type::DGSEM, 
    elem_type::Union{Line,Quad,Hex},
    quadrature_rule::AbstractQuadratureRule=LGLQuadrature(),
    mapping_degree::Int=1)

    # get spatial dimension
    if elem_type isa Line
        d = 1
    elseif elem_type isa Quad
        d = 2
    elseif elem_type isa Hex
        d = 3
    end

    @unpack p = approx_type

    # get reference element data
    reference_element = RefElemData(elem_type, mapping_degree,
        quad_rule_vol=volume_quadrature(elem_type, 
        quadrature_rule, p+1), Nplot=20)
    @unpack rstp, rstq, rstf, wq, wf = reference_element

    # dimensions of operators
    N_p = (p+1)^d
    N_q = (p+1)^d
    N_f = 2*d*(p+1)^(d-1)

    if reference_element.elementType isa Line
        V_tilde, grad_V_tilde = basis(elem_type, p, rstq[1])
        D = (LinearMap(grad_V_tilde / V_tilde),)
        R = LinearMap(vandermonde(elem_type,p,rstf[1]) / V_tilde) 
        V_plot = LinearMap(vandermonde(elem_type, p, rstp[1]) / V_tilde)
    else
        V_tilde, grad_V_tilde... = basis(elem_type, p, rstq...)
        D = Tuple(LinearMap(grad_V_tilde[m] / V_tilde) for m in 1:d)
        R = LinearMap(vandermonde(elem_type,p,rstf...) / V_tilde) 
        V_plot = LinearMap(vandermonde(elem_type, p, rstp...) / V_tilde)
    end

    V = LinearMap(I, N_q)
    P = LinearMap(I, N_q)
    W = LinearMap(Diagonal(wq))
    B = LinearMap(Diagonal(wf))

    # strong-form reference advection operator (no mass matrix)
    ADVs = Tuple(W * D[m] * P for m in 1:d)

    # weak-form reference advection operator (no mass matrix)
    ADVw = Tuple(transpose(D[m]) * W for m in 1:d)

    return ReferenceApproximation{d}(approx_type, N_p, N_q, N_f, 
        reference_element, D, V, R, P, W, B, ADVs, ADVw, V_plot)
end