@testset "linear advection flux 1D" begin
    a = 1.0
    cl = ConservationLaws.ConstantLinearAdvectionEquation1D(a)
    f = ConservationLaws.first_order_flux(cl)
    @test f(1.0) ≈ a
end

@testset "linear advection flux 2D" begin
    a = [1.0,1.0]
    cl = ConservationLaws.ConstantLinearAdvectionEquation2D(a)
    f = ConservationLaws.first_order_flux(cl)
    @test f(1.0) ≈ a
end

