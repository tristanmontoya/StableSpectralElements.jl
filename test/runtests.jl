push!(LOAD_PATH,"../")

using Test
using CLOUD

tol = 1.0e-8

include("test_advection_2d.jl")
@testset "Advection 2D" begin
    (l2, conservation, energy) = test_advection_2d(
        DGMulti(4),Tri(), WeakConservationForm(
            SkewSymmetricMapping(),
            LaxFriedrichsNumericalFlux(0.0)))
            
    @test l2 ≈ 0.07409452050788953 atol=tol
    @test conservation ≈ 0.0 atol=tol
    @test energy ≈ 0.0 atol=tol
    println("success!")
end