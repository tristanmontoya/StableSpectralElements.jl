# Module `SpatialDiscretizations`

The `SpatialDiscretizations` module defines the discretization on the reference element and provides the geometric data relating to the mesh and the mapping from reference to physical space.

## Overview
Discretizations in StableSpectralElements.jl are constructed by first building a local approximation on a canonical reference element, denoted generically as $\hat{\Omega} \subset \mathbb{R}^d$, and using a bijective transformation $\bm{X}^{(\kappa)} : \hat{\Omega} \rightarrow \Omega^{(\kappa)}$ to construct the approximation on each physical element $\Omega^{(\kappa)} \subset \Omega$ of the mesh $\{ \Omega^{(\kappa)}\}_{\kappa \in \{1:N_e\}}$ in terms of the associated operators on the reference element. An example of such a mapping is shown below, where we also depict the collapsed coordinate transformation $\bm{\chi} : [-1,1]^d \to \hat{\Omega}$ which may be used to construct operators with a tensor-product structure on the reference simplex.

![Mesh mapping](./assets/meshmap.svg)

In order to define the different geometric reference elements, the subtypes `Line`, `Quad`, `Hex`, `Tri`, and `Tet` of `AbstractElemShape` from StartUpDG.jl are used and re-exported by StableSpectralElements.jl, representing the following reference domains:
```math
\begin{aligned}
\hat{\Omega}_{\mathrm{line}} &= [-1,1],\\
\hat{\Omega}_{\mathrm{quad}} &= [-1,1]^2,\\
\hat{\Omega}_{\mathrm{hex}} & = [-1,1]^3, \\
\hat{\Omega}_{\mathrm{tri}} &= \big\{ \bm{\xi} \in [-1,1]^2 : \xi_1 + \xi_2 \leq 0 \big\},\\
\hat{\Omega}_{\mathrm{tet}} &= \big\{ \bm{\xi} \in [-1,1]^3 : \xi_1 + \xi_2 + \xi_3 \leq -1 \big\}.
\end{aligned}
```
These element types are used in the constructor for StableSpectralElements.jl's [`ReferenceApproximation`](@ref) type, along with a subtype of `AbstractApproximationType` ([`NodalTensor`](@ref), [`ModalTensor`](@ref), [`NodalMulti`](@ref), [`ModalMulti`](@ref), [`NodalMultiDiagE`](@ref), or [`ModalMultiDiagE`](@ref)) specifying the nature of the local approximation. 

All the information used to define the spatial discretization on the physical domain $\Omega$ is contained within a [`SpatialDiscretization`](@ref) structure, which is constructed using a [`ReferenceApproximation`](@ref) and a [`MeshData`](https://jlchan.github.io/StartUpDG.jl/stable/MeshData/) from StartUpDG.jl. When the constructor for a [`SpatialDiscretization`](@ref) is called, the grid metrics are computed and stored in a field of type [`GeometricFactors`](@ref).

## Reference
```@autodocs
Modules = [SpatialDiscretizations]
Order   = [:function, :type]
```