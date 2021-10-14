@testset "linear advection flux" begin
    a = [1.0,1.0]
    pde = ConservationLaws.ConstantCoefficientLinearAdvectionEquation(a)
    f = ConservationLaws.first_order_flux(pde)
    @test f(1.0) ≈ a
end


