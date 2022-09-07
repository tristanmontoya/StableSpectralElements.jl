# `CLOUD.jl` - Conservation Laws on Unstructured Domains

`CLOUD.jl` is a Julia framework for high-order discretizations of conservation laws of the form
```
∂ₜu(x,t) + ∇·F(u(x,t), ∇u(x,t)) = s(x,t)
```
 on general unstructured grids using dynamically dispatched strategies for the evaluation of a broad class of discretization operators. Documentation is currently in progress.

## License

This software is released under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).
