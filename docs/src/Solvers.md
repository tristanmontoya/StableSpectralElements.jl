# Module `Solvers`

## Overview

StableSpectralElements.jl is based on a semi-discrete approach to the numerical solution of partial differential equations, in which the spatial discretization is performed first in order to obtain a system of ordinary differential equations of the form 
```math
\frac{\mathrm{d} }{\mathrm{d}t}\underline{u}(t) = \underline{R}(\underline{u}(t),t),
```
where $\underline{u}(t) \in \mathbb{R}^{N_p \cdot N_c \cdot N_e}$ is the global solution array containing the $N_p$ coefficients for each of the $N_c$ solution variables and each of the $N_e$ mesh elements. Such systems can be solved using standard time-marching methods from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) by computing the semi-discrete residual $\underline{R}(\underline{u}(t),t)$ using a Julia function with the following signature:
```julia
semi_discrete_residual(dudt::AbstractArray{Float64,3},
                       u::AbstractArray{Float64, 3},
                       solver::Solver,
                       t::Float64)
```
The first parameter contains the time derivative to be computed in place, the second parameter is the current solution state, and the fourth parameter is the time $t$. The third parameter, which is of type `Solver`, contains all the information defining the spatial discretization as well as preallocated arrays used for temporary storage. The particular algorithm used for computing the semi-discrete residual is then dispatched based on the particular parametric subtype of `Solver` which is passed into the `semi_discrete_residual!` function.

## Reference

