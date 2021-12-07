module SpatialDiscretizations

    using UnPack
    using LinearAlgebra: I, inv, transpose, Diagonal
    using LinearMaps: LinearMap, UniformScalingMap
    using StartUpDG: MeshData, RefElemData, AbstractElemShape, basis, vandermonde, gauss_quad, gauss_lobatto_quad

    using ..Mesh: GeometricFactors

    using Reexport
    @reexport using StartUpDG: Line, Quad, Tri, Tet, Hex, Pyr

    export AbstractApproximationType, AbstractCollocatedApproximation, AbstractQuadratureRule, ReferenceApproximation, GeometricFactors, LGLQuadrature, LGQuadrature, SpatialDiscretization, ReferenceApproximation, volume_quadrature, apply_to_all_nodes, apply_to_all_dof
    
    abstract type AbstractApproximationType end
    abstract type AbstractQuadratureRule end
    abstract type AbstractCollocatedApproximation <: AbstractApproximationType end
    
    struct LGLQuadrature <: AbstractQuadratureRule end
    struct LGQuadrature <: AbstractQuadratureRule end

    struct ReferenceApproximation{d}
        approx_type::AbstractApproximationType
        N_p::Int
        N_q::Int
        N_f::Int
        reference_element::RefElemData{d}
        D::NTuple{d, LinearMap{Float64}}
        V::LinearMap
        R::LinearMap{Float64}
        P::LinearMap
        W::LinearMap
        B::LinearMap
        ADVs::NTuple{d, LinearMap{Float64}}
        ADVw::NTuple{d, LinearMap{Float64}}
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

    function volume_quadrature(::Line,
        quadrature_rule::LGQuadrature,
        num_quad_nodes::Int)
            return gauss_quad(0,0,num_quad_nodes-1) 
    end

    function volume_quadrature(::Line, 
        ::LGLQuadrature,
        num_quad_nodes::Int)
            return gauss_lobatto_quad(0,0,num_quad_nodes-1)
    end

    function apply_to_all_nodes(f::Function,
        x::NTuple{d, Matrix{Float64}}, N_eq::Int=1) where {d}
        # f maps tuple of length d to tuple of length N_eq
        
        N_nodes = size(x[1])[1]
        N_el = size(x[1])[2]
        nodal_values = Array{Float64}(undef, N_nodes, N_eq, N_el)
        
        for k in 1:N_el
            for i in 1:N_nodes
                # get tuple of length N_eq at node i of elem k
                vector_at_node = f(Tuple(x[m][i,k] for m in 1:d)) 
                for e in 1:N_eq
                    nodal_values[i,e,k] = vector_at_node[e]
                end
            end
        end
        return nodal_values
    end

    function apply_to_all_dof(f::Vector{<:LinearMap},
        dof::Array{Float64,3})
        # dof may be volume/facet/solution nodal values 

        N_eq = size(dof)[2]
        N_el = size(dof)[3]

        # assumes matrix for f same size for all elements
        output = Array{Float64}(undef, size(f[1],1), N_eq, N_el)
        
        for k in 1:N_el
            for e in 1:N_eq
                output[:,e,k] = f[k] * dof[:,e,k]
            end
        end
        return output
    end
    
    export DGSEM
    include("dgsem.jl")

end