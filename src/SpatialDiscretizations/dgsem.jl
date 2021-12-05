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

    # dimensions of operators
    N_p = (approx_type.p+1)^d
    N_q = N_p
    N_f = 2*d*(approx_type.p+1)^(d-1)

    # get reference element data
    reference_element = RefElemData(elem_type, mapping_degree,
        quad_rule_vol=volume_quadrature(elem_type, 
        quadrature_rule, approx_type.p+1))

    # modal basis to nodal basis
    V_tilde, grad_V_tilde = basis(
        reference_element.elementType, 
        approx_type.p,reference_element.rq)

    # fundamental operators
    V = LinearMap(I, N_q)
    P = LinearMap(I, N_q)
    R = LinearMap(vandermonde(reference_element.elementType,
        approx_type.p, reference_element.rf) / V_tilde)
    W = LinearMap(Diagonal(reference_element.wq))
    B = LinearMap(Diagonal(reference_element.wf))

    # mass matrix inverse
    invM = LinearMap(inv(Diagonal(reference_element.wq)))

    # strong-form derivative operator
    if reference_element.elementType isa Line
        D = (LinearMap(grad_V_tilde / V_tilde),)
    else
        D = Tuple(LinearMap(grad_V_tilde[m] / V_tilde) 
            for m in 1:d)
    end

    # weak-form advection operator
    ADV = Tuple(transpose(D[m]) * W for m in 1:d)

    # solution to plotting nodes
    V_plot = LinearMap(vandermonde(
        reference_element.elementType, approx_type.p, 
        reference_element.rp) / V_tilde)

    return ReferenceApproximation{d}(approx_type, N_p, N_q, N_f, 
        reference_element, D, ADV, V, V_plot, R, 
        invM, P, W, B)
end