module SpatialDiscretizations

    using UnPack
    using LinearAlgebra: I, inv, Diagonal, diagm, kron
    using LinearMaps: LinearMap
    using StartUpDG: MeshData, basis, vandermonde, grad_vandermonde, quad_nodes, NodesAndModes.quad_nodes_tri, NodesAndModes.quad_nodes_tet, face_vertices, nodes, find_face_nodes, init_face_data, equi_nodes, face_type, Polynomial, jacobiP, match_coordinate_vectors

    using Jacobi: zgrjm, wgrjm, zgj, wgj, zglj, wglj
    import StartUpDG: face_type, init_face_data

    using ..Mesh: GeometricFactors
    using ..MatrixFreeOperators: TensorProductMap2D, TensorProductMap3D, WarpedTensorProductMap2D, WarpedTensorProductMap3D, SelectionMap

    using Reexport
    @reexport using StartUpDG: RefElemData, AbstractElemShape, Line, Quad, Tri, Tet, Hex

    export AbstractApproximationType, NodalTensor, ModalTensor, ModalMulti, NodalMulti, AbstractReferenceMapping, NoMapping, ReferenceApproximation, GeometricFactors, SpatialDiscretization, check_normals, check_facet_nodes, check_sbp_property, centroids, dim, χ, warped_product
    
    abstract type AbstractApproximationType end

    struct NodalTensor <: AbstractApproximationType 
        p::Int
    end

    struct ModalTensor <: AbstractApproximationType
        p::Int
    end

    struct ModalMulti <: AbstractApproximationType
        p::Int
    end

    struct NodalMulti <: AbstractApproximationType
        p::Int
    end

    """Collapsed coordinate mapping χ: [-1,1]ᵈ → Ωᵣ"""
    abstract type AbstractReferenceMapping end
    struct NoMapping <: AbstractReferenceMapping end
    struct ReferenceMapping <: AbstractReferenceMapping 
        J_ref::Vector{Float64}
        Λ_ref::Array{Float64, 3}
    end

    """Operators for local approximation on reference element"""
    struct ReferenceApproximation{d, ElemShape, ApproxType}
        approx_type::ApproxType
        N_p::Int
        N_q::Int
        N_f::Int
        reference_element::RefElemData{d, ElemShape}
        D::NTuple{d, LinearMap}
        V::LinearMap
        Vf::LinearMap
        R::LinearMap
        W::Diagonal
        B::Diagonal
        V_plot::LinearMap
        reference_mapping::AbstractReferenceMapping
    end
    
    """Data for constructing the global spatial discretization"""
    struct SpatialDiscretization{d}
        mesh::MeshData{d}
        N_e::Int
        reference_approximation::ReferenceApproximation{d}
        geometric_factors::GeometricFactors{d}
        M::Vector{AbstractMatrix}
        x_plot::NTuple{d, Matrix{Float64}}
    end

    function SpatialDiscretization(mesh::MeshData{d},
        reference_approximation::ReferenceApproximation{d};
        project_jacobian::Bool=false) where {d}

        @unpack reference_element, reference_mapping, W, V = reference_approximation

        N_e = size(mesh.xyz[1])[2]
        geometric_factors = apply_reference_mapping(GeometricFactors(mesh,
            reference_element), reference_mapping)
        @unpack J_q, Λ_q, J_f, nJf = geometric_factors

        if project_jacobian
            J_proj = similar(J_q)
            for k in 1:N_e J_proj[:,k] = V * inv(Matrix(V'*W*V)) * V' * W * J_q[:,k] end
        else J_proj = J_q end

        return SpatialDiscretization{d}(mesh, N_e, reference_approximation, 
            GeometricFactors(J_proj, Λ_q, J_f, nJf),
            [Matrix(V' * Diagonal(W * J_proj[:,k]) * V) for k in 1:N_e],
            Tuple(reference_element.Vp * mesh.xyz[m] for m in 1:d))
    end

    """Use this when there are no collapsed coordinates"""
    @inline apply_reference_mapping(geometric_factors::GeometricFactors, ::NoMapping) = geometric_factors
    
    """Express all metric terms in terms of collapsed coordinates"""
    function apply_reference_mapping(geometric_factors::GeometricFactors,
        reference_mapping::ReferenceMapping)
        @unpack J_q, Λ_q, J_f, nJf = geometric_factors
        @unpack J_ref, Λ_ref = reference_mapping

        (N_q, N_e) = size(J_q)
        d = size(Λ_q, 2)
        Λ_η = similar(Λ_q)

        for k in 1:N_e, i in 1:N_q, m in 1:d, n in 1:d
            Λ_η[i,m,n,k] = sum(Λ_ref[i,m,l] * Λ_q[i,l,n,k] ./ J_ref[i] 
                for l in 1:d)
        end
        
        return GeometricFactors{d}(J_q, Λ_η, J_f, nJf)
    end

    """
    Check if the normals are equal and opposite under the mapping
    """
    function check_normals(
        spatial_discretization::SpatialDiscretization{d}) where {d}
        @unpack geometric_factors, mesh, N_e = spatial_discretization
        return Tuple([maximum(abs.(geometric_factors.nJf[m][:,k] + 
                geometric_factors.nJf[m][mesh.mapP[:,k]])) for k in 1:N_e]
                for m in 1:d)
    end

    """
    Check if the facet nodes are conforming
    """
    function check_facet_nodes(
        spatial_discretization::SpatialDiscretization{d}) where {d}
        @unpack geometric_factors, mesh, N_e = spatial_discretization
        return Tuple([maximum(abs.(mesh.xyzf[m][:,k] -
                mesh.xyzf[m][mesh.mapP[:,k]])) for k in 1:N_e]
                for m in 1:d)
    end

    """
    Check if the SBP property is satisfied on the reference element
    """
    function check_sbp_property(
        reference_approximation::ReferenceApproximation{d}) where {d}       
        @unpack W, D, R, B = reference_approximation
        @unpack reference_mapping = reference_approximation
        @unpack nrstJ = reference_approximation.reference_element

        if reference_mapping isa NoMapping
            return Tuple(maximum(abs.(Matrix(W*D[m] + D[m]'*W - 
                R'*B*Diagonal(nrstJ[m])*R))) for m in 1:d)
        else
            @unpack Λ_ref, J_ref = reference_mapping
            D_ξ = Tuple(sum(Diagonal(Λ_ref[:,l,m]./J_ref) * D[l]
                for l in 1:d) for m in 1:d)
            return Tuple(maximum(abs.(Matrix(W*D_ξ[m] + D_ξ[m]'*W - 
                R'*B*Diagonal(nrstJ[m])*R))) for m in 1:d)
        end
    end

    """
    Check if the SBP property is satisfied on the physical element
    """
    function check_sbp_property(
        spatial_discretization::SpatialDiscretization{d}, k::Int=1) where {d}

        @unpack V, Vf, D, B = spatial_discretization.reference_approximation
        @unpack Λ_q, nJf = spatial_discretization.geometric_factors

        S = Tuple((sum( 0.5 * D[m]' * W * Diagonal(Λ_q[:,m,n,k]) -
                        0.5 * Diagonal(Λ_q[:,m,n,k]) * W * D[m] for m in 1:d) + 
                0.5 * R' * B * Diagonal(nJf[n][:,k]) * R) for n in 1:d)

        E = Tuple(R' * B * Diagonal(nJf[n][:,k]) * R for n in 1:d)
            
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
            for m in 1:d) for k in 1:spatial_discretization.N_e]
    end

    dim(::Line) = 1
    dim(::Union{Tri,Quad}) = 2
    dim(::Union{Tet,Hex}) = 3

    export AbstractQuadratureRule, DefaultQuadrature, LGLQuadrature, LGQuadrature, LGRQuadrature, GaussLobattoQuadrature, GaussRadauQuadrature, GaussQuadrature, quadrature
    include("quadrature_rules.jl")

    # new constructors for RefElemData from StartUpDG
    include("ref_elem_data.jl")

    include("multidimensional.jl")
    include("tensor_cartesian.jl")

    export reference_geometric_factors, operators_1d
    include("tensor_simplex.jl")

end