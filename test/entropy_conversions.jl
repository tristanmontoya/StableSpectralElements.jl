@testset "Euler entropy conversions" begin
    for d = 1:3, γ in (1.4, 5 / 3), ρ in (0.1, 1.0, 10.0), p in (0.2, 1.0, 5.0)
        conservation_law = EulerEquations{d}(γ)
        velocity = [(-1.0)^m * (0.2 + 0.1 * m) for m = 1:d]
        kinetic_energy = 0.5 * ρ * sum(abs2, velocity)
        state = [ρ; ρ .* velocity; p / (γ - 1) + kinetic_energy]
        entropy_variables = [
            (γ - log(p / ρ^γ)) / (γ - 1) - kinetic_energy / p
            ρ .* velocity ./ p
            -ρ / p
        ]

        for row_view in (false, true)
            u = row_view ? view(repeat(reshape(state, 1, :), 3, 1), 2, :) : copy(state)
            w = row_view ? view(zeros(3, d + 2), 2, :) : zeros(d + 2)
            recovered = row_view ? view(zeros(3, d + 2), 2, :) : similar(state)

            conservative_to_entropy!(w, conservation_law, u)
            @test w ≈ entropy_variables rtol = 1.0e-13 atol = 1.0e-13
            @test u == state

            original_w = copy(w)
            entropy_to_conservative!(recovered, conservation_law, w)
            @test recovered ≈ state rtol = 1.0e-13 atol = 1.0e-13
            @test w == original_w

            entropy_to_conservative!(recovered, conservation_law, entropy_variables)
            @test recovered ≈ state rtol = 1.0e-13 atol = 1.0e-13
        end
    end
end
