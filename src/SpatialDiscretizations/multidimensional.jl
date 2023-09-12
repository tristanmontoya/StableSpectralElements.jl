function ReferenceApproximation(
    approx_type::ModalMulti, element_type::AbstractElemShape;
    mapping_degree::Int=1, N_plot::Int=10, volume_quadrature_rule=DefaultQuadrature(2*approx_type.p),
    facet_quadrature_rule=DefaultQuadrature(2*approx_type.p))

    d = dim(element_type)
    (; p) = approx_type
    
    if d > 1
        reference_element = RefElemData(element_type, mapping_degree, 
            quad_rule_vol=quadrature(element_type, volume_quadrature_rule),
            quad_rule_face=quadrature(face_type(element_type),
                facet_quadrature_rule), Nplot=N_plot)
    else
        reference_element = RefElemData(element_type, mapping_degree, 
            quad_rule_vol=quadrature(element_type, volume_quadrature_rule), Nplot=N_plot)
    end

    (; rstq, rstf, rstp, wq,) = reference_element  

    VDM, ∇VDM... = basis(element_type, p, rstq...)
    Vf = vandermonde(element_type, p, rstf...)
    V_plot = vandermonde(element_type, p, rstp...)
    P = Matrix(inv(VDM' * Diagonal(wq) * VDM) * VDM' * Diagonal(wq))

    return ReferenceApproximation(approx_type, reference_element, 
        Tuple(OctavianMap(∇VDM[m] * P) for m in 1:d), OctavianMap(VDM),
        OctavianMap(Vf), OctavianMap(Vf * P), OctavianMap(V_plot))
end

function ReferenceApproximation(
    approx_type::NodalMulti, element_type::AbstractElemShape;
    mapping_degree::Int=1, N_plot::Int=10, volume_quadrature_rule=DefaultQuadrature(2*approx_type.p),
    facet_quadrature_rule=DefaultQuadrature(2*approx_type.p))

    d = dim(element_type)
    (; p) = approx_type

    reference_element = RefElemData(element_type, mapping_degree, 
        quad_rule_vol=quadrature(element_type, volume_quadrature_rule),
        quad_rule_face=quadrature(face_type(element_type),
            facet_quadrature_rule), Nplot=N_plot)

    (; rstq, rstf, rstp, wq) = reference_element

    VDM, ∇VDM... = basis(element_type, p, rstq...)     
    Vf = vandermonde(element_type, p, rstf...)
    V_plot = vandermonde(element_type, p, rstp...)
    P = Matrix(inv(VDM' * Diagonal(wq) * VDM) * VDM' * Diagonal(wq))
    
    return ReferenceApproximation(approx_type, reference_element, 
        Tuple(OctavianMap(∇VDM[m] * P) for m in 1:d), LinearMap(I, length(wq)),
        OctavianMap(Vf * P), OctavianMap(Vf * P), OctavianMap(V_plot*P))
end

function ReferenceApproximation(
    approx_type::ModalMultiDiagE, 
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

    (; rstq, rstf, rstp, wq) = reference_element

    V = OctavianMap(vandermonde(element_type, approx_type.p, rstq...))
    V_plot = OctavianMap(vandermonde(element_type, approx_type.p, rstp...)) 
    #R = LinearMap(sbp_element.Vf) # SparseMatrixCSC
    R = SelectionMap(match_coordinate_vectors(rstf, rstq), length(wq))

    return ReferenceApproximation(approx_type, reference_element, 
        Tuple(OctavianMap(sbp_element.Drst[m]) for m in 1:d), V, 
        OctavianMap(Matrix(R*V)), R, V_plot)
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

    (; rstq, rstf, rstp, wq) = reference_element

    VDM = vandermonde(element_type, approx_type.p, rstq...) 
    V = LinearMap(I, length(wq))
    V_plot = OctavianMap(vandermonde(element_type, approx_type.p, rstp...) *
        inv(VDM' * Diagonal(wq) * VDM) * VDM' * Diagonal(wq))
    #R = LinearMap(sbp_element.Vf) # SparseMatrixCSC
    R = SelectionMap(match_coordinate_vectors(rstf, rstq), length(wq))

    return ReferenceApproximation(approx_type, reference_element,
        Tuple(OctavianMap(sbp_element.Drst[m]) for m in 1:d), V, R, R, V_plot)
end