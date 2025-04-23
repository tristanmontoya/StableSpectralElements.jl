# StableSpectralElements.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://tjbmontoya.com/StableSpectralElements.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tjbmontoya.com/StableSpectralElements.jl/dev/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CI](https://github.com/tristanmontoya/StableSpectralElements.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/tristanmontoya/StableSpectralElements.jl/actions/workflows/ci.yml) [![Documenter](https://github.com/tristanmontoya/StableSpectralElements.jl/actions/workflows/documenter.yml/badge.svg)](https://github.com/tristanmontoya/StableSpectralElements.jl/actions/workflows/documenter.yml) 
<p align=center>
<img src="docs/src/assets/visualization.png" alt="drawing" style="width:300px;"/>

**StableSpectralElements.jl** is a Julia framework for the numerical solution of hyperbolic and mixed hyperbolic-parabolic conservation laws on general unstructured grids using [provably stable discontinuous spectral-element methods with the summation-by-parts property](https://tjbmontoya.com/papers/MontoyaPhDThesis24.pdf), with an emphasis on dispatched strategies for the evaluation of a broad class of discretization operators.

## Installation

StableSpectralElements.jl is a registered Julia package (compatible with Julia versions 1.10 and above) and can be installed by entering the following commands within the [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/):
```julia
using Pkg; Pkg.add("StableSpectralElements")
```

For further information on the functionality and usage of this package, please refer to the [example notebooks](https://github.com/tristanmontoya/StableSpectralElements.jl/tree/main/examples/) as well as the [documentation](https://tjbmontoya.com/StableSpectralElements.jl/dev/), or feel free to [send me an email](mailto:tristan.montoya@alumni.utoronto.ca). If you suspect something is not working properly, or have an idea for how to improve StableSpectralElements.jl, [please file an issue](https://github.com/tristanmontoya/StableSpectralElements.jl/issues). Contributions from the community are always welcome!

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

## License

This software is released under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).
