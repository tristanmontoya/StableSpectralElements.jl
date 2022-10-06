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
    W = LinearMap(Diagonal(wq))
    B = LinearMap(Diagonal(wf))
    P = inv_M * V' * W

    R = make_operator(Vf * P, operator_algorithm)
    D = (make_operator(∇V[1] * P, operator_algorithm),)
    ADVw = (make_operator(∇V[1]' * Matrix(W), operator_algorithm),)

    return ReferenceApproximation(approx_type, N_p, N_q, N_f, 
        reference_element, D, V, Vf, R, W, B, ADVw, V_plot, NoMapping())
end

function ReferenceApproximation(
    approx_type::DGMulti, ::Tri;
    mapping_degree::Int=1, N_plot::Int=10, 
    operator_algorithm::AbstractOperatorAlgorithm=GenericMatrixAlgorithm())

    @unpack p,q,q_f = approx_type
    N_p = binomial(p+2, 2)
    reference_element = RefElemData(Tri(), 
        mapping_degree, quad_rule_vol=quad_nodes(Tri(), q),
        quad_rule_face=quad_nodes(face_type(Tri()), q_f), Nplot=N_plot)
    @unpack rstp, rstq, rstf, wq, wf = reference_element

    VDM, ∇VDM... = basis(Tri(), p, rstq...) 
    ∇V = Tuple(LinearMap(∇VDM[m]) for m in 1:2)

    V = make_operator(LinearMap(VDM), operator_algorithm)
    Vf = make_operator(LinearMap(vandermonde(Tri(),p,rstf...)), operator_algorithm)

    V_plot = LinearMap(vandermonde(Tri(), p, rstp...))
    N_q = length(wq)
    N_f = length(wf)
    inv_M = LinearMap(inv(VDM' * Diagonal(wq) * VDM))
    W = LinearMap(Diagonal(wq))
    B = LinearMap(Diagonal(wf))
    P = inv_M * V' * W

    R = make_operator(Vf * P, operator_algorithm)
    D = Tuple(make_operator(∇V[m] * P, operator_algorithm) for m in 1:2)
    ADVw = Tuple(make_operator(∇V[m]' * Matrix(W), operator_algorithm) 
        for m in 1:2)

    return ReferenceApproximation(approx_type, N_p, N_q, N_f, 
        reference_element, D, V, Vf, R, W, B, ADVw, V_plot, NoMapping())
end