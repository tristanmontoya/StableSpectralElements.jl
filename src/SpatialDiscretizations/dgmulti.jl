struct DGMulti <: AbstractApproximationType
    p::Int  # polynomial degree
end

function ReferenceApproximation(approx_type::DGMulti, 
    elem_type::Union{Line,Tri,Tet};
    mapping_degree::Int=1, N_plot::Int=10)

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
            mapping_degree, quad_rule_vol=quad_nodes(elem_type, p), Nplot=N_plot)

        @unpack rstp, rstq, rstf, wq, wf = reference_element    

        VDM, ∇VDM = basis(elem_type, p, rstq[1])     
        ∇V = (LinearMap(∇VDM),)
        R = LinearMap(vandermonde(elem_type,p,rstf[1]))
        V_plot = LinearMap(vandermonde(elem_type, p, rstp[1]))
    else
        reference_element = RefElemData(elem_type, 
            mapping_degree, quad_rule_vol=quad_nodes(elem_type, p),
            quad_rule_face=quad_nodes(face_type(elem_type), p), Nplot=N_plot)

        @unpack rstp, rstq, rstf, wq, wf = reference_element

        VDM, ∇VDM... = basis(elem_type, p, rstq...) 
        ∇V = Tuple(LinearMap(∇VDM[m]) for m in 1:d)
        Vf = LinearMap(vandermonde(elem_type,p,rstf...))
        V_plot = LinearMap(vandermonde(elem_type, p, rstp...))
    end

    N_p = binomial(p+d, d)
    N_q = length(wq)
    N_f = length(wf)
    V = LinearMap(VDM)
    inv_M = LinearMap(inv(VDM' * Diagonal(wq) * VDM))
    W = LinearMap(Diagonal(wq))
    B = LinearMap(Diagonal(wf))
    P = inv_M * V' * W
    R = Vf * P
    
    # D here is the nodal derivative operator
    D = Tuple(∇V[m] * P for m in 1:d)

    # weak-form reference advection operator (no mass matrix)
    ADVw = Tuple(∇V[m]' * W for m in 1:d)

    return ReferenceApproximation{d}(approx_type, N_p, N_q, N_f, 
        reference_element, D, V, Vf, R, P, W, B, ADVw, V_plot, 
        NoMapping())
end