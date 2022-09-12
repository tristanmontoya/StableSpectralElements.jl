# CLOUD.jl: Conservation Laws on Unstructured Domains

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tjbmontoya.com/CLOUD.jl/dev/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

CLOUD.jl is a Julia framework for high-order methods for hyperbolic and mixed hyperbolic-parabolic conservation laws on general unstructured grids using dynamically dispatched strategies for the evaluation of a broad class of discretization operators. 

## Installation

To install CLOUD.jl, open a `julia` session and enter:

```julia
julia> import Pkg

julia> Pkg.add([
    "https://github.com/tristanmontoya/CLOUD.jl.git",
    "OrdinaryDiffEq"])
```

## License

This software is released under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).