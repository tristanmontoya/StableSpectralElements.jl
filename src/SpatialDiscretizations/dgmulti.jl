struct DGMulti <: AbstractApproximationType
    p::Int  # polynomial degree
#    q::Int # quadrature parameter (left out for backwards compatibility)
#    q_f::Int # quadrature parameter (left out for backwards compatibility)
end

function ReferenceApproximation(
    approx_type::DGMulti, ::Line; 
    mapping_degree::Int=1, N_plot::Int=10)

    @unpack p = approx_type
    q = p
    N_p = p+1
    N_q = q+1
    N_f = 2

    reference_element = RefElemData(Line(), 
        mapping_degree, quad_rule_vol=quad_nodes(Line(), q), 
        Nplot=N_plot)
    @unpack rstp, rstq, rstf, wq, wf = reference_element    

    VDM, ∇VDM = basis(Line(), p, rstq[1])     
    ∇V = (LinearMap(∇VDM),)
    R = LinearMap(vandermonde(Line(),p,rstf[1]))
    V_plot = LinearMap(vandermonde(Line(), p, rstp[1]))
    V = LinearMap(VDM)
    inv_M = LinearMap(inv(VDM' * Diagonal(wq) * VDM))
    W = LinearMap(Diagonal(wq))
    B = LinearMap(Diagonal(wf))
    P = inv_M * V' * W
    R = Vf * P
    D = (∇V[m] * P,)
    ADVw = (∇V[m]' * W,)

    return ReferenceApproximation{1}(approx_type, N_p, N_q, N_f, 
        reference_element, D, V, Vf, R, P, W, B, ADVw, V_plot, 
        NoMapping())
end

function ReferenceApproximation(
    approx_type::DGMulti, ::Tri;
    mapping_degree::Int=1, N_plot::Int=10)

    #@unpack p, q, q_f = approx_type
    @unpack p = approx_type
    q = p
    q_f = p

    N_p = binomial(p+2, 2)
    reference_element = RefElemData(Tri(), 
        mapping_degree, quad_rule_vol=quad_nodes(Tri(), q),
        quad_rule_face=quad_nodes(face_type(Tri()), q_f), Nplot=N_plot)
    @unpack rstp, rstq, rstf, wq, wf = reference_element

    VDM, ∇VDM... = basis(Tri(), p, rstq...) 
    ∇V = Tuple(LinearMap(∇VDM[m]) for m in 1:2)
    Vf = LinearMap(vandermonde(Tri(),p,rstf...))
    V_plot = LinearMap(vandermonde(Tri(), p, rstp...))
    N_q = length(wq)
    N_f = length(wf)
    V = LinearMap(VDM)
    inv_M = LinearMap(inv(VDM' * Diagonal(wq) * VDM))
    W = LinearMap(Diagonal(wq))
    B = LinearMap(Diagonal(wf))
    P = inv_M * V' * W
    R = Vf * P
    D = Tuple(∇V[m] * P for m in 1:2)
    ADVw = Tuple(∇V[m]' * W for m in 1:2)

    return ReferenceApproximation{2}(approx_type, N_p, N_q, N_f, 
        reference_element, D, V, Vf, R, P, W, B, ADVw, V_plot, 
        NoMapping())
end