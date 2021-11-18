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
    approx_type::DGSEM,
    form::StrongConservationForm) where {d}

    V_tilde, grad_V_tilde = basis(
        reference_element.elementType, 
        approx_type.p,reference_element.rq)
    V = LinearMap(I, size(V_tilde,2))
    P = LinearMap(I, size(V_tilde,2))
    R = LinearMap(vandermonde(reference_element.elementType, 
        approx_type.p, reference_element.rf) / V_tilde)
    M = diagm(reference_element.wq)

    if reference_element.elementType isa Line
        D_strong = (LinearMap(grad_V_tilde / V_tilde),)
        D_weak = (LinearMap(inv(M) * transpose(grad_V_tilde / V_tilde) * M),)
    else
        D_strong = Tuple(LinearMap(grad_V_tilde[m] / V_tilde) for m in 1:d)
        D_weak = Tuple(
            LinearMap(inv(M) * transpose(grad_V_tilde[m] / V_tilde) * M)
            for m in 1:d)
    end

    L = inv(M) * transpose(R) * diagm(reference_element.wf)

    V_plot = LinearMap(vandermonde(
        reference_element.elementType, approx_type.p, reference_element.rp) / V_tilde)
    x_plot = Tuple(reference_element.Vp * mesh.xyz[m] for m in 1:d)

    reference_operators = ReferenceOperators{d}(D_strong, 
        D_weak, P, V, V_plot, R, L)

    return SpatialDiscretization{d}(
        mesh, 
        form, 
        reference_operators,
        x_plot)
        
end