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
    approx_type::AbstractCollocatedApproximation,
    form::StrongConservationForm)

    volume_operator = nothing # d-tuple, volume nodes to DOF
    facet_operator = nothing # all facet nodes to DOF
    solution_to_volume_nodes = LinearMap(I, size(reference_element.Vq,2))
    solution_to_facet_nodes =  LinearMap(
        reference_element.Vf * reference_element.Pq)

    return SpatialDiscretization(
        conservation_law,
        mesh, 
        reference_element,
        approx_type, 
        form, 
        volume_operator,
        facet_operator,
        solution_to_volume_nodes,
        solution_to_facet_nodes)
end