module SpatialDiscretizations

    using UnPack
    using LinearAlgebra: I, inv, Diagonal, diagm, kron
    using LinearMaps: LinearMap
    using StartUpDG: MeshData, RefElemData, basis, vandermonde, quad_nodes, gauss_quad, gauss_lobatto_quad, face_vertices, nodes, find_face_nodes, init_face_data, equi_nodes, face_type, Polynomial, jacobiP

    using Jacobi: zgrjm, wgrjm, zgj, wgj
    import StartUpDG: face_type, init_face_data

    using ..Mesh: GeometricFactors
    using ..Operators: TensorProductMap, WarpedTensorProductMap, SelectionMap

    using Reexport
    @reexport using StartUpDG: AbstractElemShape, Line, Quad, Tri, Tet, Hex, Pyr

    export AbstractApproximationType, AbstractReferenceMapping, NoMapping, CollapsedMapping, ReferenceApproximation, GeometricFactors, SpatialDiscretization, check_normals, check_facet_nodes, check_sbp_property, centroids, make_sbp_operator, χ, warped_product
    
    abstract type AbstractApproximationType end

    """Collapsed coordinate mapping χ: [-1,1]ᵈ → Ωᵣ"""
    abstract type AbstractReferenceMapping end
    struct NoMapping <: AbstractReferenceMapping end
    struct ReferenceMapping <: AbstractReferenceMapping 
        J_ref::Vector{Float64}
        Λ_ref::Array{Float64, 3}
    end

    """Operators for local approximation on reference element"""
    struct ReferenceApproximation{d}
        approx_type::AbstractApproximationType
        N_p::Int
        N_q::Int
        N_f::Int
        reference_element::RefElemData{d}
        D::NTuple{d, LinearMap}
        V::LinearMap
        Vf::LinearMap
        R::LinearMap
        W::LinearMap
        B::LinearMap
        ADVw::NTuple{d, LinearMap}
        V_plot::LinearMap
        reference_mapping::AbstractReferenceMapping
    end
    
    """Data for constructing the global spatial discretization"""
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

        @unpack reference_element, reference_mapping, W = reference_approximation

        N_el = size(mesh.xyz[1])[2]
        geometric_factors = apply_reference_mapping(GeometricFactors(mesh,
            reference_element), reference_mapping)

        return SpatialDiscretization{d}(
            mesh,
            N_el,
            reference_approximation,
            geometric_factors,
            [convert(Matrix, reference_approximation.V' * W *
                Diagonal(geometric_factors.J_q[:,k]) * 
                reference_approximation.V) for k in 1:N_el],
            Tuple(reference_element.Vp * mesh.xyz[m] for m in 1:d))
    end

    """Use this when there are no collapsed coordinates"""
    @inline apply_reference_mapping(geometric_factors::GeometricFactors, ::NoMapping) = geometric_factors
    
    """Express all metric terms in terms of collapsed coordinates"""
    function apply_reference_mapping(geometric_factors::GeometricFactors,
        reference_mapping::ReferenceMapping)
        @unpack J_q, Λ_q, nJf = geometric_factors
        @unpack J_ref, Λ_ref = reference_mapping

        (N_q, N_el) = size(J_q)
        d = size(Λ_q, 2)
        Λ_η = similar(Λ_q)
        J_η = similar(J_q)

        for k in 1:N_el
            for i in 1:N_q
                for m in 1:d, n in 1:d
                    Λ_η[i,m,n,k] = sum( Λ_ref[i,m,l] * Λ_q[i,l,n,k] 
                        for l in 1:d)
                end
                J_η[i,k] = J_ref[i] * J_q[i,k]
            end
        end
        
        return GeometricFactors{d}(J_η, Λ_η, nJf)
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
        @unpack W, V, D, Vf, B = reference_approximation
        @unpack nrstJ = reference_approximation.reference_element
        
        return Tuple(maximum(abs.(convert(Matrix,
            V'*W*D[m]*V + V' * D[m]'*W*V - Vf'*B*Diagonal(nrstJ[m])*Vf  
            ))) for m in 1:d)
    end

    """
    Check if the SBP property is satisfied on the physical element
    """
    function check_sbp_property(
        spatial_discretization::SpatialDiscretization{d}, k::Int=1) where {d}

        @unpack ADVw, V, Vf, D, B = spatial_discretization.reference_approximation
        @unpack Λ_q, nJf = spatial_discretization.geometric_factors

        S = Tuple((sum(0.5 * D[m]' * W * Diagonal(Λ_q[:,m,n,k]) * V -
                0.5 * V' * Diagonal(Λ_q[:,m,n,k]) * W * D[m] * V
                for m in 1:d) + 
                0.5 * Vf' * B * Diagonal(nJf[n][:,k]) * Vf) for n in 1:d)

        E = Tuple(Vf' * B * Diagonal(nJf[n][:,k]) * Vf for n in 1:d)
            
        return Tuple(maximum(abs.(convert(Matrix,
            S[n] + S[n]' - E[n]))) for n in 1:d)
    end

    """
    Average of vertex positions (not necessarily actual centroid).
    Use only for plotting.
    """
    function centroids(
        spatial_discretization::SpatialDiscretization{d}) where {d}

        @unpack xyz = spatial_discretization.mesh
        return [Tuple(sum(xyz[m][:,k])/length(xyz[m][:,k]) 
            for m in 1:d) for k in 1:spatial_discretization.N_el]
    end

    export AbstractQuadratureRule, LGLQuadrature, LGQuadrature, LGRQuadrature, JGLQuadrature, JGRQuadrature, JGQuadrature, JacobiQuadrature, LegendreQuadrature, quadrature, facet_node_ids
    include("quadrature.jl")

    export DGSEM
    include("dgsem.jl")

    export DGMulti, QuadratureDG
    include("dgmulti.jl")

    export CollapsedSEM, CollapsedModal, reference_geometric_factors
    include("collapsed.jl")

end