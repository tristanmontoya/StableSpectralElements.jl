"""
Euler equations
"""
struct EulerEquations{d} <: AbstractConservationLaw{d,Hyperbolic}
    γ::Float64
end

function num_equations(::EulerEquations{d}) where {d}
   return d+2 
end