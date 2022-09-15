push!(LOAD_PATH,"../")

using Test
using CLOUD

tol = 1.0e-10

include("test_advection_2d.jl")



@testset "Advection 2D Energy-Conservative DGMulti" begin
    (l2, conservation, energy) = test_advection_2d(
        DGMulti(4),Tri(), WeakConservationForm(
            mapping_form=SkewSymmetricMapping(),
            inviscid_numerical_flux=LaxFriedrichsNumericalFlux(0.0)))
            
    @test l2 ≈ 0.07409452050788953 atol=tol
    @test conservation ≈ 0.0 atol=tol
    @test energy ≈ 0.0 atol=tol
end

@testset "Advection 2D Standard DGSEM" begin
    (l2, conservation, energy) = test_advection_2d(
        DGSEM(4),Quad(), WeakConservationForm(
            mapping_form=StandardMapping(),
            inviscid_numerical_flux=LaxFriedrichsNumericalFlux()))
    @test l2 ≈ 0.05018367633381625 atol=tol
    @test conservation ≈ 0.0 atol=tol
    @test energy ≈ -0.008377987714660848 atol=tol
end