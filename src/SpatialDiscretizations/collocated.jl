abstract type AbstractCollocatedApproximation <: AbstractApproximationType end

struct DGSEM <:AbstractCollocatedApproximation
    p::Int
end

# collocation on optimized nodes, exact mass matrix
struct DGMulti <: AbstractCollocatedApproximation end


function SpatialDiscretization(
    conservation_law::ConservationLaw,
    mesh::MeshData,
    reference_element::RefElemData, 
    approx_type::DGSEM,
    form::StrongConservationForm)

    # create refrence element operators

    if reference_element.elementType == Line

        V_tilde, grad_V_tilde = basis(
            Line(), approx_type.p,reference_element.rq)
        V = LinearMap(I, size(V_tilde,2))
        R = LinearMap(vandermonde(
            Line(), approx_type.p, reference_element.rf) / V_tilde)
        M = LinearMap(diagm(reference_element.wq))
        B = LinearMap(diagm(reference_element.wf))

        # reference strong and weak differentiation matrices
        D_strong = LinearMap(grad_V_tilde / V_tilde)
        D_weak = inv(M) * transpose(D_strong) * M

        # reference lifting matrix
        L = inv(M) * transpose(R) * B

        # geometric factors
        J = repeat(transpose(diff(mesh.VX)/2),length(reference_element.rq),1)
        rxJ = one.(J)
        nxJ = repeat([-1.0; 1.0],1,mesh.K)
        sJ = abs.(nxJ)  
    else
        return nothing
    end

    # for first-order flux, need to store reference operators 
    # (D_strong, D_weak, P, V, R, L)

    reference_operators = nothing
    geometric_factors = nothing

    return SpatialDiscretization(
        conservation_law,
        mesh, 
        form, 
        reference_operators,
        geometric_factors)
end