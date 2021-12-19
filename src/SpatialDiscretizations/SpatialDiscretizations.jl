module SpatialDiscretizations

    using UnPack
    using LinearAlgebra: I, inv, transpose, Diagonal
    using LinearMaps: LinearMap, UniformScalingMap
    using StartUpDG: MeshData, RefElemData, AbstractElemShape, face_type,basis, vandermonde, quad_nodes, gauss_quad, gauss_lobatto_quad

    using ..Mesh: GeometricFactors

    using Reexport
    @reexport using StartUpDG: Line, Quad, Tri, Tet, Hex, Pyr

    export AbstractApproximationType, AbstractCollocatedApproximation, ReferenceApproximation, GeometricFactors, SpatialDiscretization
    
    abstract type AbstractApproximationType end
    abstract type AbstractCollocatedApproximation <: AbstractApproximationType end
    
    struct ReferenceApproximation{d}
        approx_type::AbstractApproximationType
        N_p::Int
        N_q::Int
        N_f::Int
        reference_element::RefElemData{d}
        D::NTuple{d, LinearMap}
        V::LinearMap
        R::LinearMap
        P::LinearMap
        W::LinearMap
        B::LinearMap
        ADVs::NTuple{d, LinearMap}
        ADVw::NTuple{d, LinearMap}
        V_plot::LinearMap 
    end
    
    struct SpatialDiscretization{d}
        mesh::MeshData{d}
        N_el::Int
        reference_approximation::ReferenceApproximation{d}
        geometric_factors::GeometricFactors{d}
        M::Vector{AbstractMatrix}
        x_plot::NTuple{d, Matrix{Float64}}
    end

    function SpatialDiscretization(mesh::MeshData{d},
        reference_approximation::ReferenceApproximation{d}) where {d}

        @unpack reference_element = reference_approximation

        N_el = size(mesh.xyz[1])[2]
        geometric_factors = GeometricFactors(mesh,
            reference_approximation.reference_element)

        if reference_approximation.approx_type isa AbstractCollocatedApproximation
            
            return SpatialDiscretization{d}(
                mesh,
                N_el,
                reference_approximation,
                geometric_factors,
                [Diagonal(reference_element.wq) *
                    Diagonal(geometric_factors.J[:,k]) for k in 1:N_el],
                Tuple(reference_element.Vp * mesh.xyz[m] for m in 1:d))

        else 

            return SpatialDiscretization{d}(
                mesh,
                N_el,
                reference_approximation,
                geometric_factors,
                [convert(Matrix, 
                transpose(reference_approximation.V) * 
                    Diagonal(reference_element.wq) *
                    Diagonal(geometric_factors.J[:,k]) * 
                    reference_approximation.V) for k in 1:N_el],
                Tuple(reference_element.Vp * mesh.xyz[m] for m in 1:d))
        end
    end

    export AbstractQuadratureRule, LGLQuadrature, LGQuadrature, volume_quadrature
    include("quadrature.jl")
    
    export DGSEM
    include("dgsem.jl")

    export DGMulti
    include("dgmulti.jl")

end