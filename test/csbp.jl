using Test, LinearAlgebra, StableSpectralElements

@testset "Classical SBP polynomial accuracy" begin
    for degree = 1:4
        D, Q, H, x = StableSpectralElements.SpatialDiscretizations.get_1d_csbp(degree)
        weights = diag(H)

        @test all(>(0.0), weights)
        @test sum(weights) ≈ 2.0 atol = 1.0e-10

        # Differentiate every monomial through the boundary degree
        for power = 0:degree
            derivative = power == 0 ? zeros(length(x)) : power .* x .^ (power - 1)
            @test D * (x .^ power) ≈ derivative atol = 1.0e-10
        end

        # The norm integrates monomials through degree 2p - 1
        for power = 0:(2*degree-1)
            integral = iseven(power) ? 2.0 / (power + 1) : 0.0
            @test dot(weights, x .^ power) ≈ integral atol = 1.0e-10
        end

        boundary = zeros(length(x), length(x))
        boundary[1, 1] = -1.0
        boundary[end, end] = 1.0
        @test Q ≈ H * D atol = 1.0e-10
        @test Q + Q' ≈ boundary atol = 1.0e-10
    end
end
