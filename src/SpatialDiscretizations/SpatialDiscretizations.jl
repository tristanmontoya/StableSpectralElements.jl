module SpatialDiscretizations

using LinearAlgebra: I, inv, Diagonal, diagm, kron, transpose, det, eigvals, mul!
using Random: rand, shuffle
using LinearMaps: LinearMap, ⊗
using StartUpDG:
                 MeshData,
                 basis,
                 vandermonde,
                 grad_vandermonde,
                 diagE_sbp_nodes,
                 quad_nodes,
                 NodesAndModes.quad_nodes_tri,
                 NodesAndModes.quad_nodes_tet,
                 face_vertices,
                 nodes,
                 num_faces,
                 find_face_nodes,
                 init_face_data,
                 equi_nodes,
                 face_type,
                 Polynomial,
                 jacobiP,
                 match_coordinate_vectors,
                 uniform_mesh,
                 make_periodic,
                 jaskowiec_sukumar_quad_nodes,
                 Hicken,
                 geometric_factors,
                 MultidimensionalQuadrature
using Jacobi: zgrjm, wgrjm, zgj, wgj, zglj, wglj

using ..MatrixFreeOperators

using Reexport
@reexport using StartUpDG: RefElemData, AbstractElemShape, Line, Quad, Tri, Tet, Hex, SBP
@reexport using StaticArrays: SArray, SMatrix, SVector

export AbstractApproximationType,
       AbstractTensorProduct,
       AbstractMultidimensional,
       NodalTensor,
       ModalTensor,
       ModalMulti,
       NodalMulti,
       ModalMultiDiagE,
       NodalMultiDiagE,
       AbstractReferenceMapping,
       AbstractMetrics,
       ExactMetrics,
       ConservativeCurlMetrics,
       ChanWilcoxMetrics,
       NoMapping,
       ReferenceApproximation,
       GeometricFactors,
       SpatialDiscretization,
       apply_reference_mapping,
       reference_derivative_operators,
       check_normals,
       check_facet_nodes,
       check_sbp_property,
       centroids,
       trace_constant,
       dim,
       χ,
       warped_product

abstract type AbstractApproximationType end
abstract type AbstractTensorProduct <: AbstractApproximationType end
abstract type AbstractMultidimensional <: AbstractApproximationType end
@doc raw"""
    NodalTensor(p::Int)

Approximation type for a nodal formulation of polynomial degree $p$ based on
tensor-product volume and facet quadrature rules (generalized Vandermonde matrix is
identity, derivative and interpolation/extrapolation operators have tensor-product
structure). Currently supports `Line`, `Tri`, `Tet`, `Quad`, and `Hex` element types.
"""
struct NodalTensor <: AbstractTensorProduct
    p::Int
end

@doc raw"""
    ModalTensor(p::Int)

Approximation type for a modal formulation of polynomial degree $p$ based on tensor-product
volume and facet quadrature rules (generalized Vandermonde matrix is not necessarily
identity, derivative and interpolation/extrapolation operators have tensor-product
structure). Currently supports `Tri` and `Tet` element types.
"""
struct ModalTensor <: AbstractTensorProduct
    p::Int
end

@doc raw"""
    NodalMulti(p::Int)

Approximation type for a nodal formulation based on multidimensional volume and facet 
quadrature rules (generalized Vandermonde matrix is identity, derivative and 
interpolation/extrapolation operators are dense). Currently supports `Tri` and `Tet` element
types.
"""
struct NodalMulti <: AbstractMultidimensional
    p::Int
end

@doc raw"""
    ModalMulti(p::Int)

Approximation type for a modal formulation of polynomial degree $p$ based on 
multidimensional volume and facet quadrature rules (generalized Vandermonde, derivative and
interpolation/extrapolation operators are all dense). Currently supports `Tri` and `Tet`
element types.
"""
struct ModalMulti <: AbstractMultidimensional
    p::Int
end

@doc raw"""
    NodalMultiDiagE(p::Int)

Approximation type for a nodal formulation of polynomial degree $p$ based on a 
multidimensional volume quadrature rule including nodes collocated with those used for
facet integration (generalized Vandermonde matrix is identity, derivative operator is 
dense, interpolation/extrapolation operator picks out values at facet quadrature nodes).
Currently supports only the `Tri` element type.
"""
struct NodalMultiDiagE <: AbstractMultidimensional
    p::Int
end

@doc raw"""
    ModalMultiDiagE(p::Int)

Approximation type for a modal formulation based on a multidimensional volume quadrature
rule of polynomial degree $p$ including nodes collocated with those used for facet
integration (generalized Vandermonde and derivative operators are dense, interpolation/
extrapolation operator picks out values at facet quadrature nodes). Currently supports only
the `Tri` element type.
"""
struct ModalMultiDiagE <: AbstractMultidimensional
    p::Int
end

# Collapsed coordinate mapping
abstract type AbstractReferenceMapping end
struct NoMapping <: AbstractReferenceMapping end
struct ReferenceMapping <: AbstractReferenceMapping
    J_ref::Vector{Float64}
    Λ_ref::Array{Float64, 3}
end

abstract type AbstractMetrics end
struct ExactMetrics <: AbstractMetrics end
struct ConservativeCurlMetrics <: AbstractMetrics end
const ChanWilcoxMetrics = ConservativeCurlMetrics

@doc raw"""
    ReferenceApproximation(approx_type::AbstractReferenceMapping,
                           element_type::StartUpDG.AbstractElemShape, kwargs...)

Data structure defining the discretization on the reference element, containing the
following fields, which are defined according to the approximation type, element type, and 
other parameters passed into the outer constructor:
- `approx_type::AbstractApproximationType`: Type of operators used for the discretization
  on the reference element ([`NodalTensor`](@ref), [`ModalTensor`](@ref), [`NodalMulti`]
  (@ref), [`ModalMulti`](@ref), [`NodalMultiDiagE`](@ref), or [`ModalMultiDiagE`](@ref))
- `reference_element::StartUpDG.RefElemData`: Data structure containing quadrature node
  positions and operators used for defining the mapping from reference to physical space;
  contains the field `element_type::StartUpDG.AbstractElemShape` which determines the shape
  of the reference element (currently, StableSpectralElements.jl supports the options
  `Line`, `Quad`, `Hex`, `Tri`, and `Tet`)
- `D::NTuple{d, <:LinearMap}`: Tuple of operators of size `N_q` by `N_q` approximating each
  partial derivative at the volume quadrature nodes
- `V::LinearMap`: Generalized Vandermonde matrix of size `N_q` by `N_p` mapping solution
  degrees of freedom to values at volume quadrature nodes
- `Vf::LinearMap`: Generalized Vandermonde matrix of size `N_f` by `N_p` mapping solution
  degrees of freedom to values at facet quadrature nodes
- `R::LinearMap`: Interpolation/extrapolation operator of size `N_f` by `N_q` which maps
  nodal data from volume quadrature nodes to facet quadrature nodes 
- `W::Diagonal`: Volume quadrature weight matrix of size `N_q` by `N_q`
- `B::Diagonal`: Facet quadrature weight matrix of size `N_f` by `N_f`
- `V_plot::LinearMap`: Generalized Vandermonde matrix mapping solution degrees of freedom
  to plotting nodes
- `reference_mapping::AbstractReferenceMapping`: Optional collapsed coordinate
  transformation (either `ReferenceMapping` or `NoMapping`); if such a mapping is used (i.e. not `NoMapping`), the discrete derivative operators approximate partial derivatives 
  with respect to components of the collapsed coordinate system
Outer constructors are provided to construct the discrete operators by dispatching on each
combination of subtypes of `AbstractApproximationType` and `StartUpDG.AbstractElemShape`. 
"""
struct ReferenceApproximation{RefElemType,
    ApproxType,
    D_type,
    V_type,
    Vf_type,
    R_type,
    V_plot_type,
    ReferenceMappingType}
    approx_type::ApproxType
    N_p::Int
    N_q::Int
    N_f::Int
    reference_element::RefElemType
    D::D_type
    V::V_type
    Vf::Vf_type
    R::R_type
    W::Diagonal{Float64, Vector{Float64}}
    B::Diagonal{Float64, Vector{Float64}}
    V_plot::V_plot_type
    reference_mapping::ReferenceMappingType

    function ReferenceApproximation(approx_type::ApproxType,
            reference_element::RefElemType,
            D::D_type,
            V::V_type,
            Vf::Vf_type,
            R::R_type,
            V_plot::V_plot_type,
            reference_mapping::ReferenceMappingType = NoMapping()) where {
            RefElemType,
            ApproxType,
            D_type,
            V_type,
            Vf_type,
            R_type,
            V_plot_type,
            ReferenceMappingType
    }
        return new{RefElemType,
            ApproxType,
            D_type,
            V_type,
            Vf_type,
            R_type,
            V_plot_type,
            ReferenceMappingType}(approx_type,
            size(V, 2),
            size(V, 1),
            size(R, 1),
            reference_element,
            D,
            V,
            Vf,
            R,
            Diagonal(reference_element.wq),
            Diagonal(reference_element.wf),
            V_plot,
            reference_mapping)
    end
end

@doc raw"""
    GeometricFactors(J_q::Matrix{Float64}, 
                     Λ_q::Array{Float64, 4}, 
                     J_f::Matrix{Float64},
                     nJf::Array{Float64, 3},
                     nJq::Array{Float64, 4})
Nodal values of geometric factors used by the solver to construct discretizations on the physical element. Contains the following fields:
- `J_q::Matrix{Float64}`: Jacobian determinant $J$ of the mapping from reference 
  coordinates $\bm{\xi} \in \hat{\Omega}$ to physical coordinates 
  $\bm{x} \in \Omega^{(\kappa)}$ at volume quadrature nodes; first dimension is node index 
  (size `N_q`), second is element index (size `N_e`)
- `Λ_q::Array{Float64, 4}`: Metric terms $J \partial \xi_l / \partial x_m$ at volume
  quadrature nodes; first index is node index (size `N_q`), next two are $l$ and $m$ (size 
  `d`), last is element index (size `N_e`)
- `J_f::Matrix{Float64}`: Facet area element at facet quadrature nodes;
  first index is node index (size `N_f`), second is element index (size `N_e`)
- `nJf::Array{Float64, 3}`: Scaled surface normal vector at facet quadrature nodes; first
  index is component of normal vector (size `d`), second is node index (size `N_f`), third 
  is element index (size `N_e`)
- `nJq::Array{Float64, 4}`: Scaled surface normal vector to a given facet computed using
  the volume metrics (used in flux differencing); first index is component of normal vector (size `d`), second component is reference facet index, third is volume quadrature node index (size `N_q`), last is element index (size `N_e`)

!!! note 
    When using sum-factorization algorithms in collapsed coordinates with a `StandardForm
    ` solver and a `ReferenceOperator` strategy, `apply_reference_mapping!` overwrites
    `Λ_q` to contain the metrics associated with the composite mapping from $[-1,1]^d$ to $\Omega^{(\kappa)}$. See (6.2) and (6.3) in the following paper:
    - T. Montoya and D. W. Zingg (2024). Efficient tensor-product spectral-element
      operators with the summation-by-parts property on curved triangles and tetrahedra. 
      *SIAM Journal on Scientific Computing* 46(4):A2270-A2297.
"""
struct GeometricFactors
    J_q::Matrix{Float64} # N_q x N_e
    Λ_q::Array{Float64, 4} # N_q x d x d x N_e
    J_f::Matrix{Float64} # N_f x N_e
    nJf::Array{Float64, 3} # d x N_f x N_e
    nJq::Array{Float64, 4} # d x num_faces x N_q x N_e
end

@doc raw"""
    SpatialDiscretization(mesh::StartUpDG.MeshData,
                          reference_approximation::ReferenceApproximation,
                          metric_type::AbstractMetrics, kwargs...)

Composite type containing data for constructing the discretization on the reference element
as well as the mesh and associated metric terms.
"""
struct SpatialDiscretization{d, MeshType, ReferenceApproximationType}
    mesh::MeshType
    N_e::Int
    reference_approximation::ReferenceApproximationType
    geometric_factors::GeometricFactors
    M::Vector{Matrix{Float64}}
    x_plot::NTuple{d, Matrix{Float64}}
end

dim(::SpatialDiscretization{d}) where {d} = d

function project_jacobian!(J_q::Matrix{Float64}, V::LinearMap, W::Diagonal, ::Val{true})
    VDM = Matrix(V)
    proj = VDM * inv(VDM' * W * VDM) * VDM' * W
    @inbounds for k in axes(J_q, 2)
        J_qk = copy(J_q[:, k])
        mul!(view(J_q, :, k), proj, J_qk)
    end
end

function project_jacobian!(::Matrix{Float64}, ::LinearMap, ::Diagonal, ::Val{false})
    return
end

function physical_mass_matrix(J_q::Matrix{Float64}, V::LinearMap, W::Diagonal)
    N_e = size(J_q, 2)
    VDM = Matrix(V)
    M = Vector{Matrix{Float64}}(undef, N_e)
    @inbounds for k in 1:N_e
        M[k] = VDM' * W * Diagonal(J_q[:, k]) * VDM
    end
    return M
end

function SpatialDiscretization(mesh::MeshType,
        reference_approximation::ReferenceApproximationType,
        metric_type::ExactMetrics = ExactMetrics();
        project_jacobian::Bool = true) where {
        d,
        MeshType <:
        MeshData{d},
        ReferenceApproximationType <:
        ReferenceApproximation{<:RefElemData{d}}
}
    (; reference_element, W, V) = reference_approximation

    geometric_factors = GeometricFactors(mesh, reference_element, metric_type)
    (; J_q, Λ_q, J_f, nJf, nJq) = geometric_factors

    project_jacobian!(J_q, V, W, Val(project_jacobian))

    return SpatialDiscretization{d, MeshType, ReferenceApproximationType}(mesh,
        size(J_q, 2),
        reference_approximation,
        GeometricFactors(J_q,
            Λ_q,
            J_f,
            nJf,
            nJq),
        physical_mass_matrix(J_q,
            V,
            W),
        Tuple(reference_element.Vp *
              mesh.xyz[m]
        for m in 1:d))
end

function SpatialDiscretization(mesh::MeshType,
        reference_approximation::ReferenceApproximationType,
        metric_type::ChanWilcoxMetrics) where {
        d,
        MeshType <:
        MeshData{d},
        ReferenceApproximationType <:
        ReferenceApproximation{<:RefElemData{d}}
}
    (; reference_element, W, V) = reference_approximation
    (; J_q, Λ_q, J_f, nJf, nJq) = GeometricFactors(mesh, reference_element, metric_type)

    return SpatialDiscretization{d, MeshType, ReferenceApproximationType}(mesh,
        size(J_q, 2),
        reference_approximation,
        GeometricFactors(J_q,
            Λ_q,
            J_f,
            nJf,
            nJq),
        physical_mass_matrix(J_q,
            V,
            W),
        Tuple(reference_element.Vp *
              mesh.xyz[m]
        for m in 1:d))
end

# Use this when there are no collapsed coordinates
@inline apply_reference_mapping(geometric_factors::GeometricFactors, ::NoMapping) = geometric_factors

# Express all metric terms in terms of collapsed coordinates
function apply_reference_mapping(geometric_factors::GeometricFactors,
        reference_mapping::ReferenceMapping)
    (; J_q, Λ_q, J_f, nJf, nJq) = geometric_factors
    (; J_ref, Λ_ref) = reference_mapping
    (N_q, N_e) = size(J_q)
    d = size(Λ_q, 2)
    Λ_η = similar(Λ_q)

    @inbounds for k in 1:N_e, i in 1:N_q, m in 1:d, n in 1:d
        Λ_η[i, m, n, k] = sum(Λ_ref[i, m, l] * Λ_q[i, l, n, k] / J_ref[i] for l in 1:d)
    end

    return GeometricFactors(J_q, Λ_η, J_f, nJf, nJq)
end

# Get derivative operators in reference coordinates from collapsed coordinates
function reference_derivative_operators(D_η::NTuple{d, LinearMap},
        reference_mapping::ReferenceMapping) where {d}
    (; Λ_ref, J_ref) = reference_mapping
    return Tuple(sum(Diagonal(Λ_ref[:, l, m] ./ J_ref) * D_η[l] for l in 1:d) for m in 1:d)
end

function reference_derivative_operators(D_η::NTuple{d, LinearMap}, ::NoMapping) where {d}
    return D_η
end

# Check if the normals are equal and opposite under the mapping (i.e. watertight mesh)
function check_normals(spatial_discretization::SpatialDiscretization{d}) where {d}
    (; geometric_factors, mesh, N_e) = spatial_discretization
    return Tuple([maximum(abs.(geometric_factors.nJf[m, :, k] +
                               geometric_factors.nJf[m, :, :][mesh.mapP[:, k]]))
                  for k in 1:N_e] for m in 1:d)
end

# Check if the facet nodes are conforming
function check_facet_nodes(spatial_discretization::SpatialDiscretization{d}) where {d}
    (; mesh, N_e) = spatial_discretization
    return Tuple([maximum(abs.(mesh.xyzf[m][:, k] - mesh.xyzf[m][mesh.mapP[:, k]]))
                  for k in 1:N_e]
    for m in 1:d)
end

# Check if the SBP property is satisfied on the reference element
function check_sbp_property(reference_approximation::ReferenceApproximation{
        <:RefElemData{d},
}) where {
        d,
}
    (; W, D, R, B) = reference_approximation
    (; reference_mapping) = reference_approximation
    (; nrstJ) = reference_approximation.reference_element

    D_ξ = reference_derivative_operators(D, reference_mapping)

    return Tuple(maximum(abs.(Matrix(W * D_ξ[m] + D_ξ[m]' * W -
                                     R' * B * Diagonal(nrstJ[m]) * R)))
    for m in 1:d)
end

# Check if the SBP property is satisfied on the physical element
function check_sbp_property(spatial_discretization::SpatialDiscretization{d},
        k::Int = 1) where {d}
    (; W, D, R, B) = spatial_discretization.reference_approximation
    (; Λ_q, nJf) = spatial_discretization.geometric_factors

    Q = Tuple((sum(0.5 * D[m]' * W * Diagonal(Λ_q[:, m, n, k]) -
                   0.5 * Diagonal(Λ_q[:, m, n, k]) * W * D[m] for m in 1:d) +
               0.5 * R' * B * Diagonal(nJf[n, :, k]) * R) for n in 1:d)

    E = Tuple(R' * B * Diagonal(nJf[n, :, k]) * R for n in 1:d)

    return Tuple(maximum(abs.(convert(Matrix, Q[n] + Q[n]' - E[n]))) for n in 1:d)
end

# Average of vertex positions (not necessarily actual centroid).
function centroids(spatial_discretization::SpatialDiscretization{d}) where {d}
    (; xyz) = spatial_discretization.mesh
    return [Tuple(sum(xyz[m][:, k]) / length(xyz[m][:, k]) for m in 1:d)
            for
            k in 1:(spatial_discretization.N_e)]
end

# Trace inequality constant from Chan et al. (2016)
function trace_constant(reference_approximation::ReferenceApproximation)
    (; B, Vf, W, V) = reference_approximation
    return maximum(eigvals(Matrix(Vf' * B * Vf), Matrix(V' * W * V)))
end

@inline dim(::Line) = 1
@inline dim(::Union{Tri, Quad}) = 2
@inline dim(::Union{Tet, Hex}) = 3

export AbstractQuadratureRule,
       DefaultQuadrature,
       LGLQuadrature,
       LGQuadrature,
       LGRQuadrature,
       GaussLobattoQuadrature,
       GaussRadauQuadrature,
       GaussQuadrature,
       XiaoGimbutasQuadrature,
       JaskowiecSukumarQuadrature,
       quadrature
include("quadrature_rules.jl")

# new constructors for RefElemData from StartUpDG
include("ref_elem_data.jl")

include("multidimensional.jl")
include("tensor_cartesian.jl")

export reference_geometric_factors, operators_1d
include("tensor_simplex.jl")

export GeometricFactors,
       metrics,
       uniform_periodic_mesh,
       warp_mesh,
       cartesian_mesh,
       Uniform,
       ZigZag,
       DelReyWarping,
       ChanWarping
include("mesh.jl")

end
