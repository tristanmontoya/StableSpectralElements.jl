abstract type AbstractCollocatedApproximation <: AbstractApproximationType end

struct DGSEM <:AbstractCollocatedApproximation
    p::Int
end

# collocation on optimized nodes, exact mass matrix
struct DGMulti <: AbstractCollocatedApproximation end


function SpatialDiscretization(
    mesh::MeshData,
    reference_element::RefElemData, 
    approx_type::DGSEM,
    form::StrongConservationForm)

    # create refrence element operators

    if reference_element.elementType isa Line

        V_tilde, grad_V_tilde = basis(
            Line(), approx_type.p,reference_element.rq)
        V = LinearMap(I, size(V_tilde,2))
        P = LinearMap(I, size(V_tilde,2))
        R = LinearMap(vandermonde(
            Line(), approx_type.p, reference_element.rf) / V_tilde)
        M = diagm(reference_element.wq)
        B = diagm(reference_element.wf)

        # reference strong and weak differentiation matrices
        D_strong = LinearMap(grad_V_tilde / V_tilde)
        D_weak = inv(M) * transpose(D_strong) * M

        # reference lifting matrix
        L = inv(M) * transpose(R) * B

    else
        return nothing
    end

    reference_operators = ReferenceOperators{1}(D_strong, D_weak, P, V, R, L)

    return SpatialDiscretization(
        mesh, 
        form, 
        reference_operators)
        
end