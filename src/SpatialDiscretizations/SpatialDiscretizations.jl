module SpatialDiscretizations

    using UnPack
    using LinearAlgebra: I, inv, Diagonal, diagm
    using LinearMaps: LinearMap
    using StartUpDG: MeshData, RefElemData, AbstractElemShape, basis, vandermonde, quad_nodes, gauss_quad, gauss_lobatto_quad, face_vertices, nodes, find_face_nodes, init_face_data, equi_nodes, face_type, Polynomial

    using Jacobi: zgrjm, wgrjm, zgj, wgj
    import StartUpDG: face_type, init_face_data

    using ..Mesh: GeometricFactors
    using ..Operators: TensorProductMap, SelectionMap

    using Reexport
    @reexport using StartUpDG: Line, Quad, Tri, Tet, Hex, Pyr

    export AbstractApproximationType, AbstractCollocatedApproximation, NonsymmetricElemShape, AbstractReferenceMapping, NoMapping, ReferenceMapping, ReferenceApproximation, GeometricFactors, SpatialDiscretization, CollapsedTri, check_normals, check_facet_nodes, check_sbp_property, centroids, make_sbp_operator, χ
    
    abstract type AbstractApproximationType end
    abstract type AbstractCollocatedApproximation <: AbstractApproximationType end
    abstract type NonsymmetricElemShape <: AbstractElemShape end
    abstract type AbstractReferenceMapping end
    struct NoMapping <: AbstractReferenceMapping end

    struct ReferenceMapping <: AbstractReferenceMapping 
        J_ref::Vector{Float64}
        Λ_ref::Array{Float64, 3}
    end

    struct CollapsedTri <: NonsymmetricElemShape end
    @inline face_type(::CollapsedTri) = Line()

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
        reference_mapping::AbstractReferenceMapping
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
        geometric_factors = apply_reference_mapping(GeometricFactors(mesh,
            reference_approximation.reference_element), 
            reference_approximation.reference_mapping)

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
                [convert(Matrix, reference_approximation.V' * 
                    Diagonal(reference_element.wq) *
                    Diagonal(geometric_factors.J_q[:,k]) * 
                    reference_approximation.V) for k in 1:N_el],
                Tuple(reference_element.Vp * mesh.xyz[m] for m in 1:d))
        end
    end

    @inline apply_reference_mapping(geometric_factors::GeometricFactors, ::NoMapping) = geometric_factors
    
    function apply_reference_mapping(geometric_factors::GeometricFactors,
        reference_mapping::ReferenceMapping)
        @unpack J_q, Λ_q, nJf = geometric_factors
        @unpack J_ref, Λ_ref = reference_mapping
        (N_q,N_el) = size(J_q)
        d = size(Λ_q, 3)
        J_qJ_ref = Matrix{Float64}(undef, N_q, N_el)
        Λ_qΛ_ref = zeros(N_q, d, d, N_el)
        for k in 1:N_el
            J_qJ_ref[:,k] = J_q[:,k] .* J_ref
            for i in 1:N_q
                Λ_qΛ_ref[i,:,:,k] = Λ_q[i,:,:,k] * Λ_ref[i,:,:]
            end

        end
        return GeometricFactors{d}(J_qJ_ref, Λ_qΛ_ref, nJf)
    end

    """
    Check if the normals are equal and opposite under the mapping
    """
    function check_normals(
        spatial_discretization::SpatialDiscretization{d}) where {d}
        @unpack geometric_factors, mesh, N_el = spatial_discretization
        return Tuple([maximum(abs.(geometric_factors.nJf[m][:,k] + 
                geometric_factors.nJf[m][mesh.mapP[:,k]])) for k in 1:N_el]
                for m in 1:d)
    end

    """
    Check if the facet nodes are conforming
    """
    function check_facet_nodes(
        spatial_discretization::SpatialDiscretization{d}) where {d}
        @unpack geometric_factors, mesh, N_el = spatial_discretization
        return Tuple([maximum(abs.(mesh.xyzf[m][:,k] -
                mesh.xyzf[m][mesh.mapP[:,k]])) for k in 1:N_el]
                for m in 1:d)
    end

    """
    Check if the SBP property is satisfied on the reference element
    """
    function check_sbp_property(
        reference_approximation::ReferenceApproximation{d}) where {d}       
        @unpack W, V, D, R, B = reference_approximation
        @unpack nrstJ = reference_approximation.reference_element
        
        return Tuple(maximum(abs.(convert(Matrix,
            V'*W*V*D[m] + D[m]'*V'*W*V - R'*B*Diagonal(nrstJ[m])*R  
            ))) for m in 1:d)
    end

    function meshgrid(x::Vector{Float64}, y::Vector{Float64})
        return ([x[i] for i in 1:length(x), j in 1:length(y)],
            [y[j] for i in 1:length(x), j in 1:length(y)])
    end

    """
    Average of vertex positions (not necessarily actual centroid).
    Use only for plotting.
    """
    function centroids(
        spatial_discretization::SpatialDiscretization{d}) where {d}
        
        vertices = Tuple(spatial_discretization.mesh.VXYZ[m][
            spatial_discretization.mesh.EToV] for m in 1:d)

        return [Tuple(sum(vertices[m][k,:])/length(vertices[m][k,:]) 
            for m in 1:d)
            for k in 1:spatial_discretization.N_el]
    end

    export AbstractQuadratureRule, LGLQuadrature, LGQuadrature, LGRQuadrature, JGLQuadrature, JGRQuadrature, JGQuadrature, JacobiQuadrature, LegendreQuadrature, quadrature, facet_node_ids
    include("quadrature.jl")

    export DGSEM
    include("dgsem.jl")

    export DGMulti
    include("dgmulti.jl")

    export reference_geometric_factors
    include("collapsed.jl")

end