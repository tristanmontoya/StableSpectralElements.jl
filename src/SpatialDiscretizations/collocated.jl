abstract type AbstractCollocatedApproximation <: AbstractApproximationType end

# collocation on Legendre-Gauss-Lobatto quadrature nodes
struct CollocatedLGL <:AbstractCollocatedApproximation
    p::Int
end

# collocation on Legendre-Gauss quadrature nodes
struct CollocatedLG <:AbstractCollocatedApproximation
    p::Int
end

function SpatialDiscretization1D(
    conservation_law::AbstractConservationLaw,
    mesh::MeshData,
    reference_element::RefElemData, 
    approx_type::AbstractCollocatedApproximation,
    form::AbstractResidualForm)
    # placeholder -- make operators
    operators = nothing

    return SpatialDiscretization1D(conservation_law, mesh, reference_element,
        approx_type, form, operators)
end