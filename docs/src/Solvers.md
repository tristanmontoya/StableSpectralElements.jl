# Module `Solvers`

The `Solvers` module implements the algorithms which evaluate the semi-discrete residual corresponding to the discretization of an [`AbstractConservationLaw`](@ref) in space using the operators and geometric information contained within the [`SpatialDiscretization`](@ref) to create an [`ODEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/basics/problem/), which can be solved using your choice of time-marching method from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl).

## Overview

StableSpectralElements.jl is based on a semi-discrete or "method-of-lines" approach to the numerical solution of partial differential equations, in which the spatial discretization is performed first in order to obtain a system of ordinary differential equations of the form $\underline{u}'(t) = \underline{R}(\underline{u}(t),t)$,
where $\underline{u}(t) \in \mathbb{R}^{N_p \cdot N_c \cdot N_e}$ is the global solution array containing the $N_p$ coefficients for each of the $N_c$ solution variables and each of the $N_e$ mesh elements. These systems can be solved using standard time-marching methods from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) by computing the semi-discrete residual $\underline{R}(\underline{u}(t),t)$ using the in-place function [`semi_discrete_residual!`](@ref).

## Reference

```@autodocs
Modules = [Solvers]
Order   = [:function, :type]
```