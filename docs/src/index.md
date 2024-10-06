# StableSpectralElements.jl

[**StableSpectralElements.jl**](https://github.com/tristanmontoya/StableSpectralElements.jl) is a Julia framework for the numerical solution of partial differential equations of the form
```math
\partial_t \underline{U}(\bm{x},t) + \bm{\nabla}_{\bm{x}} \cdot \underline{\bm{F}}(\underline{U}(\bm{x},t), \bm{\nabla}_{\bm{x}}\underline{U}(\bm{x},t)) = \underline{0},
```
for $t \in (0,T)$ with $T \in \mathbb{R}^+ $ and $\bm{x} \in \Omega \subset \mathbb{R}^d$, subject to appropriate initial and boundary conditions, where $\underline{U}(\bm{x},t)$ is the vector of solution variables and $\underline{\bm{F}}(\underline{U}(\bm{x},t),\bm{\nabla}_{\bm{x}}\underline{U}(\bm{x},t))$ is the flux tensor containing advective and/or diffusive contributions. 
These equations are spatially discretized on curvilinear unstructured grids using energy-stable and entropy-stable discontinuous spectral element methods in order to generate [`ODEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/basics/problem/) objects suitable for time integration using [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) within the [SciML](https://sciml.ai/) ecosystem. StableSpectralElements.jl also includes postprocessing tools employing [WriteVTK.jl](https://github.com/jipolanco/WriteVTK.jl) for generating `.vtu` files, allowing for visualization of high-order numerical solutions on unstructured grids using [ParaView](https://www.paraview.org/) or other software. Shared-memory parallelization is supported through multithreading.

The functionality provided by [StartUpDG.jl](https://github.com/jlchan/StartUpDG.jl) for the handling of mesh data structures, polynomial basis functions, and quadrature nodes is employed throughout this package. Moreover, StableSpectralElements.jl implements dispatched strategies for semi-discrete operator evaluation using [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl), allowing for the efficient matrix-free application of tensor-product operators.

Discretizations satisfying the summation-by-parts (SBP) property employing nodal as well as modal bases are implemented, with the latter allowing for efficient and low-storage inversion of the dense elemental mass matrices arising from curvilinear meshes through the use of [weight-adjusted approximations](https://arxiv.org/abs/1608.03836). Tensor-product formulations supporting sum factorization are available on triangles and tetrahedra through the use of [SBP operators in collapsed coordinates](https://arxiv.org/abs/2306.05975), as well as on quadrilaterals and hexahedra.

## Theory

For a comprehensive overview of the numerical methods implemented in StableSpectralElements.jl, please refer to Tristan's PhD thesis, [Provably Stable Discontinuous Spectral-Element Methods with the Summation-by-Parts Property: Unified Matrix Analysis and Efficient Tensor-Product Formulations on Curved Simplices](https://tjbmontoya.com/papers/MontoyaPhDThesis24.pdf).

## Installation

StableSpectralElements.jl is a registered Julia package (compatible with Julia versions 1.10 and higher) and can be installed by entering the following commands within the [Julia REPL (read-eval-print loop)](https://docs.julialang.org/en/v1/stdlib/REPL/#The-Julia-REPL):
```julia
using Pkg; Pkg.add("StableSpectralElements")
```
Alternatively, you can clone the repository and run your local version as follows:
```bash
git clone https://github.com/tristanmontoya/StableSpectralElements.jl.git
cd StableSpectralElements.jl
julia --project=.
```
In either case, you can then start using the package by entering `using StableSpectralElements`.

## Examples

We recommend that users refer to the following Jupyter notebooks (included in the `examples` directory) for examples of how to use StableSpectralElements.jl:
* [Linear advection-diffusion equation in 1D](https://github.com/tristanmontoya/StableSpectralElements.jl/tree/main/examples/advection_diffusion_1d.ipynb)
* [Inviscid Burgers' equation in 1D with energy-conservative scheme](https://github.com/tristanmontoya/StableSpectralElements.jl/tree/main/examples/burgers_1d.ipynb)
* [Euler equations in 1D with entropy-conservative Gauss collocation methods](https://github.com/tristanmontoya/StableSpectralElements.jl/tree/main/examples/euler_1d_gauss_collocation.ipynb)
* [Linear advection equation in 2D](https://github.com/tristanmontoya/StableSpectralElements.jl/tree/main/examples/advection_2d.ipynb)
* [Isentropic Euler vortex in 2D with entropy-stable modal scheme on triangles](https://github.com/tristanmontoya/StableSpectralElements.jl/tree/main/examples/euler_vortex_2d.ipynb)
* [Linear advection equation in 3D](https://github.com/tristanmontoya/StableSpectralElements.jl/tree/main/examples/advection_3d.ipynb)
* [3D Euler equations with entropy-stable modal scheme on tetrahedra](https://github.com/tristanmontoya/StableSpectralElements.jl/tree/main/examples/euler_3d.ipynb)

## Citing

If you use StableSpectralElements.jl in your research, please cite the following publications:
```bibtex
@article{MontoyaZinggTensorProduct24,
    title = {Efficient Tensor-Product Spectral-Element Operators with the Summation-by-Parts 
             Property on Curved Triangles and Tetrahedra},
    author = {Montoya, Tristan and Zingg, David W},
    journal = {{SIAM} Journal on Scientific Computing},
    volume = {46},
    number = {4},
    pages = {A2270--A2297},
    doi = {10.1137/23M1573963},
    year = {2024}
}

@article{MontoyaZinggEntropyStable24,
    title = {Efficient Entropy-Stable Discontinuous Spectral-Element Methods Using 
             Tensor-Product Summation-by-Parts Operators on Triangles and Tetrahedra},
    author = {Montoya, Tristan and Zingg, David W},
    journal = {Journal of Computational Physics},
    volume = {516},
    pages = {113360},
    doi = {10.1016/j.jcp.2024.113360},
    year = {2024}
}
```

The following repositories are associated with the above papers, and contain the code required to reproduce the results presented therein:
- <https://github.com/tristanmontoya/ReproduceSBPSimplex>
- <https://github.com/tristanmontoya/ReproduceEntropyStableDSEM>

## License
This software is released under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).