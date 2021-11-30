abstract type AbstractCollocatedApproximation <: AbstractApproximationType end

struct DGSEM <:AbstractCollocatedApproximation
    p::Int  # polynomial degree
end

# collocation on optimized nodes, exact mass matrix
struct DGMulti <: AbstractCollocatedApproximation 
    p::Int  # polynomial degree
end

function SpatialDiscretization(
    mesh::MeshData{d},
    reference_element::RefElemData, 
    approx_type::DGSEM) where {d}

    # dimension names
    N_p = size(mesh.xyzq[1])[1]  # solution DOF
    N_q = N_p  # volume quadrature nodes
    N_f = size(mesh.xyzf[1])[1]  # facet quadrature nodes 
    N_el =  size(mesh.xyzq[1])[2]  # mesh elemenrts

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
        D = Tuple(LinearMap(grad_V_tilde[m] / V_tilde) for m in 1:d)
    end

    # weak-form advection operator
    ADV = Tuple(transpose(D[m]) * W for m in 1:d)

    # mapping to plotting nodes
    V_plot = LinearMap(vandermonde(
        reference_element.elementType, approx_type.p, reference_element.rp) / V_tilde)
    x_plot = Tuple(reference_element.Vp * mesh.xyz[m] for m in 1:d)

    # create reference operators
    reference_operators = ReferenceOperators{d}(D, 
        ADV, V, V_plot, R, invM, P, W, B)

    # evaluate metrics and compute physical mass and projection matrices
    geometric_factors = GeometricFactors(mesh,reference_element)
    physical_mass_inverse = [
        LinearMap(inv(Diagonal(reference_element.wq)*Diagonal(geometric_factors.J[:,k]))) for k in 1:N_el]
    physical_projection = fill(LinearMap(I,N_q),N_el)
    jacobian_inverse = [
        LinearMap(inv(Diagonal(geometric_factors.J[:,k]))) for k in 1:N_el]

    return SpatialDiscretization{d}(
        mesh,
        N_p, 
        N_q,
        N_f,
        N_el,
        reference_element,
        reference_operators,
        geometric_factors,
        physical_mass_inverse,
        physical_projection,
        jacobian_inverse,
        x_plot)
        
end