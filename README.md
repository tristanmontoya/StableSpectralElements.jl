# CLOUD.jl: Conservation Laws on Unstructured Domains

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tjbmontoya.com/CLOUD.jl/dev/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CI](https://github.com/tristanmontoya/CLOUD.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/tristanmontoya/CLOUD.jl/actions/workflows/ci.yml) [![Documenter](https://github.com/tristanmontoya/CLOUD.jl/actions/workflows/documenter.yml/badge.svg)](https://github.com/tristanmontoya/CLOUD.jl/actions/workflows/documenter.yml) 
<p align=center>
<img src="docs/src/assets/visualization.png" alt="drawing" style="width:300px;"/>

**CLOUD.jl** is a Julia framework for the numerical solution of hyperbolic and mixed hyperbolic-parabolic conservation laws on general unstructured grids using provably stable discontinuous spectral-element methods of arbitrary order, with an emphasis on dispatched strategies for the evaluation of a broad class of discretization operators. CLOUD.jl supports shared-memory parallelization through multithreading. For further information on the functionality and usage of this package, please refer to the [example notebooks](https://github.com/tristanmontoya/CLOUD.jl/tree/main/examples/) as well as the [documentation](https://tjbmontoya.com/CLOUD.jl/dev/) or [send me an email](mailto:tristan.montoya@mail.utoronto.ca).

## License

This software is released under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).