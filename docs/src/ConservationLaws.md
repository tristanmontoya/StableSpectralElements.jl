# Module `ConservationLaws`

The `ConservationLaws` module defines the systems of partial differential equations which are solved by StableSpectralElements.jl.

## Overview

The equations to be solved are defined by subtypes of [`AbstractConservationLaw`](@ref) on which functions such as `physical_flux` and `numerical_flux` are dispatched. Whereas first-order problems (i.e. subtypes of `AbstractConservationLaw{d, FirstOrder}`) remove the dependence of the flux tensor on the solution gradient in order to obtain systems of the form
```math
\partial_t \underline{U}(\bm{x},t) + \bm{\nabla}_{\bm{x}} \cdot \underline{\bm{F}}(\underline{U}(\bm{x},t)) = \underline{0},
```
second-order problems (i.e. subtypes of `AbstractConservationLaw{d, SecondOrder}`) are treated by StableSpectralElements.jl as first-order systems of the form 
```math
\begin{aligned}
\underline{\bm{Q}}(\bm{x},t) - \bm{\nabla}_{\bm{x}} \underline{U}(\bm{x},t) &= \underline{0},\\
\partial_t \underline{U}(\bm{x},t) + \bm{\nabla}_{\bm{x}} \cdot \underline{\bm{F}}(\underline{U}(\bm{x},t), \underline{\bm{Q}}(\bm{x},t)) &= \underline{0}.
\end{aligned}
```
Currently, the linear advection and advection-diffusion equations ([`LinearAdvectionEquation`](@ref) and [`LinearAdvectionDiffusionEquation`](@ref)), the inviscid and viscous Burgers' equations ([`InviscidBurgersEquation`](@ref) and [`ViscousBurgersEquation`](@ref)), and the compressible Euler equations ([`EulerEquations`](@ref)) are supported by StableSpectralElements.jl, but any system of the above form can in principle be implemented, provided that appropriate physical and numerical fluxes are defined.

## Reference

```@autodocs
Modules = [ConservationLaws]
Order   = [:function, :type]
```