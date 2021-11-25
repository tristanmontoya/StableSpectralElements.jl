module SpatialDiscretizations

    using LinearAlgebra: I, inv, transpose, diagm
    using LinearMaps: LinearMap, UniformScalingMap
    using StartUpDG: basis, vandermonde, gauss_quad, gauss_lobatto_quad

    using Reexport
    @reexport using StartUpDG: MeshData, RefElemData, Line, Quad, Tri, Tet, Hex, Pyr

    export AbstractApproximationType, AbstractQuadratureRule, ReferenceOperators, GeometricFactors, LGLQuadrature, LGQuadrature, SpatialDiscretization, ReferenceOperators, volume_quadrature, reference_element, apply_to_all_nodes, apply_to_all_dof
    
    abstract type AbstractApproximationType end
    abstract type AbstractQuadratureRule end
    
    struct LGLQuadrature <: AbstractQuadratureRule end
    struct LGQuadrature <: AbstractQuadratureRule end

    struct ReferenceOperators{d}
        D_strong::NTuple{d, LinearMap{Float64}}
        D_weak::NTuple{d, LinearMap{Float64}}
        P::LinearMap
        V::LinearMap
        V_plot::LinearMap
        R::LinearMap{Float64}
        L::LinearMap{Float64}
    end
    
    struct SpatialDiscretization{d}
        mesh::MeshData
        N_p::Int
        N_q::Int
        N_f::Int
        N_el::Int
        reference_operators::ReferenceOperators
        projection::Vector{LinearMap}
        x_plot::NTuple{d, Matrix{Float64}}
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

    function reference_element(elem_type::Line, quadrature_rule, num_quad_nodes)
        return RefElemData(Line(),1,quad_rule_vol=volume_quadrature(elem_type, quadrature_rule, num_quad_nodes))
    end

    function apply_to_all_nodes(f::Function,
        x::NTuple{d, Matrix{Float64}}, N_eq::Int=1) where {d}
        # f maps tuple of length d to tuple of length N_eq
        
        N_vol_nodes = size(x[1])[1]
        N_el = size(x[1])[2]
        nodal_values = Array{Float64}(undef, N_vol_nodes, N_eq, N_el)
        
        for k in 1:N_el
            for i in 1:N_vol_nodes
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
    
    export AbstractCollocatedApproximation, DGSEM, DGMulti
    include("collocated.jl")

end