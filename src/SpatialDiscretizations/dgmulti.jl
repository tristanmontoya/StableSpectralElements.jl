struct DGMulti <: AbstractApproximationType
    p::Int  # polynomial degree
    q::Int # volume quadrature parameter 
    q_f::Int # facet quadrature parameter 
end

function DGMulti(p::Int; q=nothing, q_f=nothing)
    if isnothing(q)
        q = p
    end
    if isnothing(q_f)
        q_f = p
    end
    return DGMulti(p,q,q_f)
end

function ReferenceApproximation(
    approx_type::DGMulti, ::Line; 
    mapping_degree::Int=1, N_plot::Int=10, 
    operator_algorithm::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm())

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

    Vf = make_operator(LinearMap(vandermonde(Line(),p,rstf[1])),
        operator_algorithm)
    V = make_operator(LinearMap(VDM), operator_algorithm)

    V_plot = LinearMap(vandermonde(Line(), p, rstp[1]))
    inv_M = LinearMap(inv(VDM' * Diagonal(wq) * VDM))
    W = Diagonal(wq)
    B = Diagonal(wf)
    P = inv_M * V' * W

    R = make_operator(Vf * P, operator_algorithm)
    D = (make_operator(∇V[1] * P, operator_algorithm),)

    return ReferenceApproximation(approx_type, N_p, N_q, N_f, 
        reference_element, D, V, Vf, R, W, B, V_plot, NoMapping())
end

function ReferenceApproximation(
    approx_type::DGMulti, element_type::AbstractElemShape;
    mapping_degree::Int=1, N_plot::Int=10, 
    operator_algorithm::AbstractOperatorAlgorithm=BLASAlgorithm())

    @unpack p,q,q_f = approx_type
    d = dim(element_type)
    N_p = binomial(p+d, d)
    reference_element = RefElemData(element_type, 
        mapping_degree, quad_rule_vol=quad_nodes(element_type, q),
        quad_rule_face=quad_nodes(face_type(element_type), q_f), Nplot=N_plot)

        
    @unpack rstq, rstf, rstp, wq, wf = reference_element

    VDM, ∇VDM... = basis(element_type, p, rstq...) 
    ∇V = Tuple(LinearMap(∇VDM[m]) for m in 1:d)

    V = make_operator(LinearMap(VDM), operator_algorithm)
    Vf = make_operator(LinearMap(vandermonde(element_type,p,rstf...)), operator_algorithm)

    V_plot = LinearMap(vandermonde(element_type, p, rstp...))
    N_q = length(wq)
    N_f = length(wf)
    inv_M = LinearMap(inv(VDM' * Diagonal(wq) * VDM))
    W = Diagonal(wq)
    B = Diagonal(wf)
    P = inv_M * V' * W

    R = make_operator(Vf * P, operator_algorithm)
    D = Tuple(make_operator(∇V[m] * P, operator_algorithm) for m in 1:d)

    return ReferenceApproximation(approx_type, N_p, N_q, N_f, 
        reference_element, D, V, Vf, R, W, B, V_plot, NoMapping())
end