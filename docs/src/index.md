# CLOUD.jl: Conservation Laws on Unstructured Domains

CLOUD.jl is a Julia framework for the numerical solution of partial differential equations of the form
```math
\frac{\partial \underline{U}(\bm{x},t)}{\partial t} + \bm{\nabla} \cdot \underline{\bm{F}}(\underline{U}(\bm{x},t), \bm{\nabla}\underline{U}(\bm{x},t)) = \underline{0},
```
subject to appropriate initial and boundary conditions, where $\underline{U}(\bm{x},t)$ is the vector of solution variables and $\underline{\bm{F}}(\underline{U}(\bm{x},t),\bm{\nabla}\underline{U}(\bm{x},t))$ is the flux tensor containing advective and/or diffusive contributions. 
These equations are spatially discretized on curvilinear unstructured grids using discontinuous spectral element methods with the summation-by-parts property in order to generate `ODEProblem` objects suitable for time integration using [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) within the [SciML](https://sciml.ai/) ecosystem. 

The functionality provided by [StartUpDG.jl](https://github.com/jlchan/StartUpDG.jl) for the handling of mesh data structures, polynomial basis functions, and quadrature nodes is employed throughout this package. Moreover, CLOUD.jl employs dynamically dispatched strategies for semi-discrete operator evaluation using [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl), allowing for the efficient matrix-free application of tensor-product operators, including those associated with collapsed-coordinate formulations on triangles.

## Installation

To install CLOUD.jl, open a `julia` session and enter:

```julia
julia> import Pkg

julia> Pkg.add("https://github.com/tristanmontoya/CLOUD.jl.git")
```

## Basic Usage

Example usage of CLOUD.jl is provided in the following Jupyter notebooks:
* [Linear advection-diffusion equation in 1D](https://github.com/tristanmontoya/CLOUD.jl/blob/main/examples/advection_diffusion_1d.ipynb)
* [Linear advection equation in 2D](https://github.com/tristanmontoya/CLOUD.jl/blob/main/examples/advection_2d.ipynb)


## License

This software is released under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).
