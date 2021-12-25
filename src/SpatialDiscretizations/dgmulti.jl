struct DGMulti <: AbstractApproximationType
    p::Int  # polynomial degree
    #q::Int  # volume_quadrature degree
    #qf::Int  # facet quadrature degree 
end

function ReferenceApproximation(approx_type::DGMulti, 
    elem_type::Union{Line,Tri,Tet};
    mapping_degree::Int=1, N_plot=10)

    # get spatial dimension
    if elem_type isa Line
        d = 1
    elseif elem_type isa Tri
        d = 2
    elseif elem_type isa Tet
        d = 3
    end
   
    @unpack p = approx_type

    if elem_type isa Line
        reference_element = RefElemData(elem_type, 
            mapping_degree, quad_rule_vol=quad_nodes(elem_type, p),Nplot=N_plot)

        @unpack rstp, rstq, rstf, wq, wf = reference_element    

        V_tilde, grad_V_tilde = basis(elem_type, p, rstq[1])     
        grad_V = (LinearMap(grad_V_tilde),)
        R = LinearMap(vandermonde(elem_type,p,rstf[1]))
        V_plot = LinearMap(vandermonde(elem_type, p, rstp[1]))
    else
        reference_element = RefElemData(elem_type, 
            mapping_degree, quad_rule_vol=quad_nodes(elem_type, p),
            quad_rule_face=quad_nodes(face_type(elem_type), p), Nplot=N_plot)

        @unpack rstp, rstq, rstf, wq, wf = reference_element

        V_tilde, grad_V_tilde... = basis(elem_type, p, rstq...) 
        grad_V = Tuple(LinearMap(grad_V_tilde[m]) for m in 1:d)
        R = LinearMap(vandermonde(elem_type,p,rstf...))
        V_plot = LinearMap(vandermonde(elem_type, p, rstp...))
    end

    N_p = binomial(p+d, d)
    N_q = length(wq)
    N_f = length(wf)
    
    V = LinearMap(V_tilde)
    inv_M = LinearMap(inv(transpose(V_tilde) * Diagonal(wq) * V_tilde))
    W = LinearMap(Diagonal(wq))
    B = LinearMap(Diagonal(wf))
    P = inv_M * transpose(V) * W
    D = Tuple(inv_M * transpose(V) * W * grad_V[m] 
        for m in 1:d)

    # strong-form reference advection operator (no mass matrix)
    ADVs = Tuple(-transpose(V) * W * grad_V[m] * P for m in 1:d)

    # weak-form reference advection operator (no mass matrix)
    ADVw = Tuple(transpose(grad_V[m]) * W for m in 1:d)


    return ReferenceApproximation{d}(approx_type, N_p, N_q, N_f, 
        reference_element, D, V, R, P, W, B, ADVs, ADVw, V_plot)
end