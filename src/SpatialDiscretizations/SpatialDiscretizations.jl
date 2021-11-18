module SpatialDiscretizations

    using LinearAlgebra: I, inv, transpose, diagm
    using LinearMaps: LinearMap, UniformScalingMap
    using StartUpDG: basis, vandermonde, gauss_quad, gauss_lobatto_quad

    using Reexport
    @reexport using StartUpDG: MeshData, RefElemData, Line, Quad, Tri, Tet, Hex, Pyr

    export AbstractApproximationType, AbstractResidualForm, AbstractQuadratureRule, StrongConservationForm, WeakConservationForm, ReferenceOperators, GeometricFactors, LGLQuadrature, LGQuadrature, SpatialDiscretization, ReferenceOperators, volume_quadrature, l2_projection
    
    abstract type AbstractApproximationType end
    abstract type AbstractResidualForm end
    abstract type AbstractQuadratureRule end

    struct StrongConservationForm <: AbstractResidualForm end
    struct WeakConservationForm <: AbstractResidualForm end
    
    struct LGLQuadrature <: AbstractQuadratureRule end
    struct LGQuadrature <: AbstractQuadratureRule end

    struct ReferenceOperators{d}
        D_strong::NTuple{d, LinearMap{Float64}}
        D_weak::NTuple{d, LinearMap{Float64}}
        V::LinearMap
        V_plot::LinearMap
        P::LinearMap
        R::LinearMap{Float64}
        L::LinearMap{Float64}
    end

    struct SpatialDiscretization{d}
        mesh::MeshData
        form::AbstractResidualForm
        reference_operators::ReferenceOperators
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

    function l2_projection(spatial_discretization::SpatialDiscretization,
        u0::Function)

        nodal_values = @. u0(spatial_discretization.mesh.xyzq)

        #as test don't weight projection

        return nothing

    end
    
    export AbstractCollocatedApproximation, DGSEM, DGMulti
    include("collocated.jl")

end