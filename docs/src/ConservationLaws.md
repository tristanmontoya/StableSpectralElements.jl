# Module `ConservationLaws`

## Overview

The equations to be solved are defined by subtypes of `AbstractConservationLaw` on which functions such as `physical_flux` and `numerical_flux` are dispatched. Objects of type `AbstractConservationLaw` contain two type parameters, `d` and `PDEType`, the former denoting the spatial dimension of the problem, which is inherited by all subtypes, and the latter being a subtype of `AbstractPDEType` denoting the particular type of PDE being solved, which is either `FirstOrder` or `SecondOrder`. Whereas first-order problems remove the dependence of the flux tensor on the solution gradient in order to obtain systems of the form
```math
\partial_t \underline{U}(\bm{x},t) + \bm{\nabla}_{\bm{x}} \cdot \underline{\bm{F}}(\underline{U}(\bm{x},t)) = \underline{0},
```
second-order problems are treated by StableSpectralElements.jl as first-order systems of the form 
```math
\begin{aligned}
\underline{\bm{Q}}(\bm{x},t) - \bm{\nabla}_{\bm{x}} \underline{U}(\bm{x},t) &= \underline{0},\\
\partial_t \underline{U}(\bm{x},t) + \bm{\nabla}_{\bm{x}} \cdot \underline{\bm{F}}(\underline{U}(\bm{x},t), \underline{\bm{Q}}(\bm{x},t)) &= \underline{0}.
\end{aligned}
```

## Reference

```@meta
CurrentModule = ConservationLaws
```
```@docs
    LinearAdvectionEquation
    LinearAdvectionDiffusionEquation
    EulerEquations
```