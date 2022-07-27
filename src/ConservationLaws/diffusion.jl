struct ConstantDiffusionFlux{d} <: AbstractSecondOrderFlux{d,1}
    b::Float64 # diffusion coefficient
end

struct BR1{d} <: AbstractSecondOrderNumericalFlux{ConstantDiffusionFlux{d}} end

```
Linear advection-diffusion equation

∂U/∂t + ∇⋅(aU) + ∇⋅(b∇U)  = 0
```
function advection_diffusion_equation(a::NTuple{d,Float64}, 
    b::Float64; λ=1.0) where {d}
    return ConservationLaw{d,1}(ConstantLinearAdvectionFlux(a), 
        ConstantDiffusionFlux{d}(b),
        ConstantLinearAdvectionNumericalFlux(a, λ),
        BR1{d}(),nothing,nothing)
end