# CLOUD.jl: Conservation Laws on Unstructured Domains

CLOUD.jl is a Julia framework implementing several unstructured high-order methods for partial differential equations of the form
```math
\frac{\partial \underline{U}(\bm{x},t)}{\partial t} + \bm{\nabla} \cdot \underline{\bm{F}}(\underline{U}(\bm{x},t), \bm{\nabla}\underline{U}(\bm{x},t)) = \underline{0},
```
where $\underline{U}(\bm{x},t)$ is the vector of solution variables, $\underline{\bm{F}}(\underline{U}(\bm{x},t), \bm{\nabla}\underline{U}(\bm{x},t))$ is the flux tensor.

## Installation

To install CLOUD.jl, open a `julia` session and enter:

```julia
julia> import Pkg

julia> Pkg.add("https://github.com/tristanmontoya/CLOUD.jl.git")
```
## License

This software is released under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).
