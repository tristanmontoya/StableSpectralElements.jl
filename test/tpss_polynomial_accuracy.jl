using Test, LinearAlgebra, StableSpectralElements

function test_tpss_polynomials(reference_approximation)
    (; approx_type, reference_element, D, R) = reference_approximation
    (; rstq, rstf) = reference_element
    degree = approx_type.p
    dimension = length(rstq)

    @testset "$(typeof(approx_type)) $(typeof(reference_element.element_type)) degree $degree" begin
        for powers in Iterators.product(ntuple(_ -> 0:degree, dimension)...)
            sum(powers) <= degree || continue
            values =
                [prod(rstq[m][i]^powers[m] for m = 1:dimension) for i in eachindex(rstq[1])]
            facet_values =
                [prod(rstf[m][i]^powers[m] for m = 1:dimension) for i in eachindex(rstf[1])]
            extrapolated = similar(facet_values)
            mul!(extrapolated, R, values)
            @test extrapolated ≈ facet_values atol = 1.0e-10 rtol = 1.0e-10

            differentiated = similar(values)
            for direction = 1:dimension
                exact = if powers[direction] == 0
                    zeros(length(values))
                else
                    [
                        powers[direction] * prod(
                            rstq[m][i]^(powers[m] - (m == direction)) for m = 1:dimension
                        ) for i in eachindex(values)
                    ]
                end
                mul!(differentiated, D[direction], values)
                @test differentiated ≈ exact atol = 1.0e-10 rtol = 1.0e-10
            end
        end
    end
end

@testset "TPSS polynomial accuracy" begin
    for element in (Tri(), Tet()), degree = 1:2
        assembled = ReferenceApproximation(NodalTPSSLGL(degree), element)
        matrix_free = ReferenceApproximation(MatrixFreeTPSSLGL(degree), element)
        test_tpss_polynomials(assembled)
        test_tpss_polynomials(matrix_free)

        # Compare the forward and transpose actions on the same nodal values
        values = sin.(collect(1:length(assembled.reference_element.wq)))
        expected = similar(values)
        actual = similar(values)
        for direction in eachindex(assembled.D)
            mul!(expected, assembled.D[direction], values)
            mul!(actual, matrix_free.D[direction], values)
            @test actual ≈ expected atol = 1.0e-10 rtol = 1.0e-10
            mul!(expected, transpose(assembled.D[direction]), values)
            mul!(actual, transpose(matrix_free.D[direction]), values)
            @test actual ≈ expected atol = 1.0e-10 rtol = 1.0e-10
        end
    end

    for approximation in
        (NodalTPSS(1), NodalTPSS(2), NodalTPSSMinimal(1), NodalTPSSOptimal(1))
        test_tpss_polynomials(ReferenceApproximation(approximation, Tri()))
    end
end
