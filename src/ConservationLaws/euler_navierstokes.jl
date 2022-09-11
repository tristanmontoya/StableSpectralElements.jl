"""
Euler equations
"""
struct EulerEquations{d} <: AbstractConservationLaw{d,Hyperbolic}
    γ::Float64
end

num_equations(::EulerEquations{d}) = d+2