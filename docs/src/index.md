# CLOUD.jl: Conservation Laws on Unstructured Domains

[**CLOUD.jl**](https://github.com/tristanmontoya/CLOUD.jl) is a Julia framework for the numerical solution of partial differential equations of the form
```math
\partial_t \underline{U}(\bm{x},t) + \bm{\nabla}_{\bm{x}} \cdot \underline{\bm{F}}(\underline{U}(\bm{x},t), \bm{\nabla}_{\bm{x}}\underline{U}(\bm{x},t)) = \underline{0},
```
for $t \in (0,T)$ with $T \in \mathbb{R}^+ $ and $\bm{x} \in \Omega \subset \mathbb{R}^d$, subject to appropriate initial and boundary conditions, where $\underline{U}(\bm{x},t)$ is the vector of solution variables and $\underline{\bm{F}}(\underline{U}(\bm{x},t),\bm{\nabla}_{\bm{x}}\underline{U}(\bm{x},t))$ is the flux tensor containing advective and/or diffusive contributions. 
These equations are spatially discretized on curvilinear unstructured grids using discontinuous spectral element methods in order to generate `ODEProblem` objects suitable for time integration using [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) within the [SciML](https://sciml.ai/) ecosystem. CLOUD.jl also includes postprocessing tools employing [WriteVTK.jl](https://github.com/jipolanco/WriteVTK.jl) for generating `.vtu` files, allowing for visualization of high-order numerical solutions on unstructured grids using [ParaView](https://www.paraview.org/) or other tools.

The functionality provided by [StartUpDG.jl](https://github.com/jlchan/StartUpDG.jl) for the handling of mesh data structures, polynomial basis functions, and quadrature nodes is employed throughout this package. Moreover, CLOUD.jl implements dynamically dispatched strategies for semi-discrete operator evaluation using [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl), allowing for the efficient matrix-free application of tensor-product operators, including those associated with [collapsed-coordinate formulations on triangles](https://tjbmontoya.com/papers/MontoyaZinggICCFD22.pdf) as well as tetrahedra (paper to come soon).

Discretizations employing nodal as well as modal bases are implemented, with the latter allowing for efficient and low-storage inversion of the dense elemental mass matrices arising from curvilinear meshes through the use of [weight-adjusted approximations](https://arxiv.org/abs/1608.03836).

## Installation

To install CLOUD.jl, open a `julia` session and enter:

```julia
julia> import Pkg

julia> Pkg.add(url="https://github.com/tristanmontoya/CLOUD.jl.git")
```

Currently, Julia versions 1.6 or newer are supported by CLOUD.jl, although we recommend using the [current stable release](https://julialang.org/downloads/), due to a performance issue for diagonal matrix multiplication which was fixed in [Julia PR #44651](https://github.com/JuliaLang/julia/pull/44651).

## Basic Usage

As this documentation is currently a work in progress, we recommend that users refer to the following Jupyter notebooks (included in the `examples` directory) for examples of how to use CLOUD.jl:
* [Linear advection-diffusion equation in 1D](https://github.com/tristanmontoya/CLOUD.jl/tree/main/examples/advection_diffusion_1d.ipynb)
* [Linear advection equation in 2D](https://github.com/tristanmontoya/CLOUD.jl/tree/main/examples/advection_2d.ipynb)
* [Linear advection equation in 3D](https://github.com/tristanmontoya/CLOUD.jl/tree/main/examples/advection_3d.ipynb)

Scripts for running grid refinement studies are provided in the `drivers` directory; an example is provided [here](https://github.com/tristanmontoya/CLOUD.jl/blob/main/drivers/advection_refinement.sh).

## Modules
CLOUD.jl is structured as several submodules, which are exported with the top-level module `CLOUD`; below is a list of those most important for new users to familiarize themselves with:
* [`ConservationLaws`](ConservationLaws.md)
* [`SpatialDiscretizations`](SpatialDiscretizations.md)
* `Solvers`
* `Visualize`
* `Analysis`

## License
This software is released under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).