function χ(::DuffyTri, ξ::Union{NTuple{2,Float64},NTuple{2,Vector{Float64}}})
    return (0.5.*(1.0 .+ ξ[1]).*(1.0 .- ξ[2]) .- 1.0, ξ[2])
end

function ReferenceApproximation(approx_type::DGSEM, 
    elem_type::DuffyTri;
    quadrature_rule::NTuple{2,AbstractQuadratureRule}=(
        LGQuadrature(),LGQuadrature()),
    mapping_degree::Int=1, N_plot::Int=10)

    @unpack p = approx_type
    d = 2

    reference_element = RefElemData(Tri(), 
        mapping_degree, quad_rule_vol=quadrature(
            elem_type,quadrature_rule, (p+1, p+1)),
        quad_rule_face=quad_nodes(face_type(Tri()), p),
        Nplot=N_plot)

    @unpack rstp, rstq, rstf, wq, wf = reference_element

    V_tilde, grad_V_tilde... = basis(Tri(), p, rstq...) 
    grad_V = Tuple(LinearMap(grad_V_tilde[m]) for m in 1:d)
    R = LinearMap(vandermonde(Tri(),p,rstf...))
    V_plot = LinearMap(vandermonde(Tri(), p, rstp...))

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