module SpatialDiscretizations

    using UnPack
    using LinearAlgebra: I, inv, transpose, Diagonal, UniformScaling
    using LinearMaps: LinearMap, UniformScalingMap, TransposeMap
    using StartUpDG: MeshData, RefElemData, AbstractElemShape, basis, vandermonde, quad_nodes, gauss_quad, gauss_lobatto_quad
    using Jacobi: zgrjm, wgrjm

    import StartUpDG: face_type

    using ..Mesh: GeometricFactors
    using ..Operators: TensorProductMap, SelectionMap
    using Reexport
    @reexport using StartUpDG: Line, Quad, Tri, Tet, Hex, Pyr

    export AbstractApproximationType, AbstractCollocatedApproximation, NonsymmetricElemShape, ReferenceApproximation, GeometricFactors, SpatialDiscretization, check_normals, check_facet_nodes, centroids
    
    abstract type AbstractApproximationType end
    abstract type AbstractCollocatedApproximation <: AbstractApproximationType end
    abstract type NonsymmetricElemShape <: AbstractElemShape end

    struct DuffyTri <: NonsymmetricElemShape end
    struct OneToOneTri <: NonsymmetricElemShape end
    @inline face_type(::Union{DuffyTri,OneToOneTri}) = Line()

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
                    Diagonal(geometric_factors.J_q[:,k]) for k in 1:N_el],
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
                    Diagonal(geometric_factors.J_q[:,k]) * 
                    reference_approximation.V) for k in 1:N_el],
                Tuple(reference_element.Vp * mesh.xyz[m] for m in 1:d))
        end
    end

    function check_normals(
        spatial_discretization::SpatialDiscretization{d}) where {d}
        @unpack geometric_factors, mesh, N_el = spatial_discretization
        return Tuple([maximum(abs.(geometric_factors.nJf[m][:,k] + 
                geometric_factors.nJf[m][mesh.mapP[:,k]])) for k in 1:N_el]
                for m in 1:d)
    end

    function check_facet_nodes(
        spatial_discretization::SpatialDiscretization{d}) where {d}
        @unpack geometric_factors, mesh, N_el = spatial_discretization
        return Tuple([maximum(abs.(mesh.xyzf[m][:,k] -
                mesh.xyzf[m][mesh.mapP[:,k]])) for k in 1:N_el]
                for m in 1:d)
    end

    function meshgrid(x::Vector{Float64}, y::Vector{Float64})
        return ([x[i] for i in 1:length(x), j in 1:length(y)],
            [y[j] for i in 1:length(x), j in 1:length(y)])
    end

    function centroids(
        spatial_discretization::SpatialDiscretization{d}) where {d}
        
        vertices = Tuple(spatial_discretization.mesh.VXYZ[m][
            spatial_discretization.mesh.EToV] for m in 1:d)

        return [Tuple(sum(vertices[m][k,:])/length(vertices[m][k,:]) 
            for m in 1:d)
            for k in 1:spatial_discretization.N_el]
    end

    export AbstractQuadratureRule, LGLQuadrature, LGQuadrature, JGRQuadrature, quadrature, facet_node_ids
    include("quadrature.jl")

    export DGSEM
    include("dgsem.jl")

    export DGMulti
    include("dgmulti.jl")

    export OneToOneTri, DuffyTri
    include("duffy.jl")

end