function ReferenceApproximation(
    approx_type::ModalMulti, ::Line; 
    mapping_degree::Int=1, N_plot::Int=10,
    volume_quadrature_rule=LGQuadrature(approx_type.p))

    reference_element = RefElemData(Line(), mapping_degree,
        quad_rule_vol=quadrature(Line(),volume_quadrature_rule), Nplot=N_plot)

    (; rstp, rstq, rstf, wq) = reference_element    

    VDM, ∇VDM = basis(Line(), approx_type.p, rstq[1])     
    ∇V = (LinearMap(∇VDM),)
    Vf = LinearMap(vandermonde(Line(),approx_type.p,rstf[1]))
    V = LinearMap(VDM)
    V_plot = LinearMap(vandermonde(Line(), approx_type.p, rstp[1]))
    P = inv(VDM' * Diagonal(wq) * VDM) * V' * Diagonal(wq)

    return ReferenceApproximation(approx_type, reference_element,
         (∇V[1] * P,), V, Vf, Vf * P, V_plot)
end

function ReferenceApproximation(
    approx_type::ModalMulti, element_type::AbstractElemShape;
    mapping_degree::Int=1, N_plot::Int=10, volume_quadrature_rule=DefaultQuadrature(2*approx_type.p),
    facet_quadrature_rule=DefaultQuadrature(2*approx_type.p))

    d = dim(element_type)
    
    reference_element = RefElemData(element_type, mapping_degree, 
        quad_rule_vol=quadrature(element_type, volume_quadrature_rule),
        quad_rule_face=quadrature(face_type(element_type),
            facet_quadrature_rule), Nplot=N_plot)

    (; rstq, rstf, rstp, wq) = reference_element
    
    VDM, ∇VDM... = basis(element_type, approx_type.p, rstq...) 
    ∇V = Tuple(LinearMap(∇VDM[m]) for m in 1:d)
    V = LinearMap(VDM)
    Vf = LinearMap(vandermonde(element_type,approx_type.p,rstf...))
    V_plot = LinearMap(vandermonde(element_type, approx_type.p, rstp...))
    P = inv(VDM' * Diagonal(wq) * VDM) * V' * Diagonal(wq)

    return ReferenceApproximation(approx_type, reference_element, 
        Tuple(∇V[m] * P for m in 1:d), V, Vf, Vf * P, V_plot)
end

function ReferenceApproximation(
    approx_type::NodalMulti, element_type::AbstractElemShape;
    mapping_degree::Int=1, N_plot::Int=10, volume_quadrature_rule=DefaultQuadrature(2*approx_type.p),
    facet_quadrature_rule=DefaultQuadrature(2*approx_type.p))

    d = dim(element_type)

    reference_element = RefElemData(element_type, mapping_degree, 
        quad_rule_vol=quadrature(element_type, volume_quadrature_rule),
        quad_rule_face=quadrature(face_type(element_type),
            facet_quadrature_rule), Nplot=N_plot)

    (; rstq, rstf, rstp, wq) = reference_element

    VDM, ∇VDM... = basis(element_type, approx_type.p, rstq...) 
    ∇V = Tuple(LinearMap(∇VDM[m]) for m in 1:d)
    V = LinearMap(I, length(wq))
    V_plot = LinearMap(vandermonde(element_type, approx_type.p, rstp...)) 
    P = inv(VDM' * Diagonal(wq) * VDM) * VDM' * Diagonal(wq)
    R = LinearMap(vandermonde(element_type,approx_type.p,rstf...) * P)

    return ReferenceApproximation(approx_type, 
        reference_element, Tuple(∇V[m] * P for m in 1:d), V, R, R, V_plot * P)
end

function ReferenceApproximation(
    approx_type::NodalMultiDiagE, 
    element_type::AbstractElemShape;
    sbp_type::SBP=SBP{Hicken}(),
    mapping_degree::Int=1, N_plot::Int=10)

    d = dim(element_type)

    volume_quadrature_rule, facet_quadrature_rule = diagE_sbp_nodes(
        element_type, sbp_type, approx_type.p)
    
    reference_element = RefElemData(element_type, mapping_degree, 
        quad_rule_vol=volume_quadrature_rule,
        quad_rule_face=facet_quadrature_rule, Nplot=N_plot)

    sbp_element = RefElemData(element_type, sbp_type, approx_type.p)

    (; rstq, rstp, wq) = reference_element

    VDM = vandermonde(element_type, approx_type.p, rstq...) 
    V = LinearMap(I, length(wq))
    V_plot = LinearMap(vandermonde(element_type, approx_type.p, rstp...)) 
    P = inv(VDM' * Diagonal(wq) * VDM) * VDM' * Diagonal(wq)
    R = LinearMap(sbp_element.Vf)

    return ReferenceApproximation(approx_type, 
        reference_element, Tuple(LinearMap(sbp_element.Drst[m]) for m in 1:d), 
        V, R, R, V_plot * P)
end