function χ(::DuffyTri, ξ::NTuple{2,Float64})
    return (2.0 * (1 + ξ[1])/(1-ξ[2]) - 1, ξ[2])
end

function ReferenceApproximation(approx_type::DGSEM, 
    elem_type::DuffyTri;
    quadrature_rule_1::AbstractQuadratureRule=LGQuadrature(),
    quadrature_rule_2::AbstractQuadratureRule=LGQuadrature(),
    mapping_degree::Int=1, N_plot::Int=10)

end