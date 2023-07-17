# StableSpectralElements.jl

[**StableSpectralElements.jl**](https://github.com/tristanmontoya/StableSpectralElements.jl), formerly known as **CLOUD.jl** (**C**onservation **L**aws **o**n **U**nstructured **D**omains), is a Julia framework for the numerical solution of partial differential equations of the form
```math
\partial_t \underline{U}(\bm{x},t) + \bm{\nabla}_{\bm{x}} \cdot \underline{\bm{F}}(\underline{U}(\bm{x},t), \bm{\nabla}_{\bm{x}}\underline{U}(\bm{x},t)) = \underline{0},
```
for $t \in (0,T)$ with $T \in \mathbb{R}^+ $ and $\bm{x} \in \Omega \subset \mathbb{R}^d$, subject to appropriate initial and boundary conditions, where $\underline{U}(\bm{x},t)$ is the vector of solution variables and $\underline{\bm{F}}(\underline{U}(\bm{x},t),\bm{\nabla}_{\bm{x}}\underline{U}(\bm{x},t))$ is the flux tensor containing advective and/or diffusive contributions. 
These equations are spatially discretized on curvilinear unstructured grids using energy-stable and entropy-stable discontinuous spectral element methods in order to generate `ODEProblem` objects suitable for time integration using [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) within the [SciML](https://sciml.ai/) ecosystem. StableSpectralElements.jl also includes postprocessing tools employing [WriteVTK.jl](https://github.com/jipolanco/WriteVTK.jl) for generating `.vtu` files, allowing for visualization of high-order numerical solutions on unstructured grids using [ParaView](https://www.paraview.org/) or other software. Shared-memory parallelization is supported through multithreading.

The functionality provided by [StartUpDG.jl](https://github.com/jlchan/StartUpDG.jl) for the handling of mesh data structures, polynomial basis functions, and quadrature nodes is employed throughout this package. Moreover, StableSpectralElements.jl implements dispatched strategies for semi-discrete operator evaluation using [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl), allowing for the efficient matrix-free application of tensor-product operators.

Discretizations satisfying the summation-by-parts property employing nodal as well as modal bases are implemented, with the latter allowing for efficient and low-storage inversion of the dense elemental mass matrices arising from curvilinear meshes through the use of [weight-adjusted approximations](https://arxiv.org/abs/1608.03836). 

## Installation

StableSpectralElements.jl is a registered Julia package (compatible with Julia versions 1.8 and higher) and can be installed by entering the following commands within the REPL:
```julia
using Pkg; Pkg.add("StableSpectralElements")
```

## Examples

We recommend that users refer to the following Jupyter notebooks (included in the `examples` directory) for examples of how to use StableSpectralElements.jl:
* [Linear advection-diffusion equation in 1D](https://github.com/tristanmontoya/StableSpectralElements.jl/tree/main/examples/advection_diffusion_1d.ipynb)
* [Inviscid Burgers' equation in 1D with energy-conservative scheme](https://github.com/tristanmontoya/StableSpectralElements.jl/tree/main/examples/burgers_1d.ipynb)
* [Linear advection equation in 2D](https://github.com/tristanmontoya/StableSpectralElements.jl/tree/main/examples/advection_2d.ipynb)
* [Isentropic Euler vortex in 2D with entropy-conservative scheme](https://github.com/tristanmontoya/StableSpectralElements.jl/tree/main/examples/euler_vortex_2d.ipynb)
* [Linear advection equation in 3D](https://github.com/tristanmontoya/StableSpectralElements.jl/tree/main/examples/advection_3d.ipynb)

## License
This software is released under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).