function ReferenceApproximation(
    approx_type::ModalMulti, ::Line; 
    mapping_degree::Int=1, N_plot::Int=10)

    @unpack p, q = approx_type
    N_p = p+1
    N_q = q+1
    N_f = 2

    reference_element = RefElemData(Line(), 
        mapping_degree, quad_rule_vol=quad_nodes(Line(), q), 
        Nplot=N_plot)
    @unpack rstp, rstq, rstf, wq, wf = reference_element    

    VDM, ∇VDM = basis(Line(), p, rstq[1])     
    ∇V = (LinearMap(∇VDM),)
    Vf = LinearMap(vandermonde(Line(),p,rstf[1]))
    V = LinearMap(VDM)
    V_plot = LinearMap(vandermonde(Line(), p, rstp[1]))
    W = Diagonal(wq)
    B = Diagonal(wf)
    P = inv(VDM' * Diagonal(wq) * VDM) * V' * W
    R = Vf * P
    D = (∇V[1] * P,)

    return ReferenceApproximation(approx_type, N_p, N_q, N_f, 
        reference_element, D, V, Vf, R, W, B, V_plot, NoMapping())
end

function ReferenceApproximation(
    approx_type::ModalMulti, element_type::AbstractElemShape;
    mapping_degree::Int=1, N_plot::Int=10)

    @unpack p,q,q_f = approx_type
    d = dim(element_type)
    N_p = binomial(p+d, d)
    
    reference_element = RefElemData(element_type, 
        mapping_degree, quad_rule_vol=quad_nodes(element_type, q),
        quad_rule_face=quad_nodes(face_type(element_type), q_f), Nplot=N_plot)

    @unpack rstq, rstf, rstp, wq, wf = reference_element
    N_q = length(wq)
    N_f = length(wf)
    
    VDM, ∇VDM... = basis(element_type, p, rstq...) 
    ∇V = Tuple(LinearMap(∇VDM[m]) for m in 1:d)
    V = LinearMap(VDM)
    Vf = LinearMap(vandermonde(element_type,p,rstf...))
    V_plot = LinearMap(vandermonde(element_type, p, rstp...))
    W = Diagonal(wq)
    B = Diagonal(wf)
    P = inv(VDM' * Diagonal(wq) * VDM) * V' * W
    R = Vf * P
    D = Tuple(∇V[m] * P for m in 1:d)

    return ReferenceApproximation(approx_type, N_p, N_q, N_f, 
        reference_element, D, V, Vf, R, W, B, V_plot, NoMapping())
end