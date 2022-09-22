push!(LOAD_PATH,"../")

using Test
using CLOUD

tol = 1.0e-10

include("test_2d.jl")

# unit tests


# integration tests

@testset "Advection 2D Energy-Conservative DGMulti" begin
    (l2, conservation, energy) = test_2d(
        DGMulti(4),Tri(), 
        LinearAdvectionEquation((1.0,1.0)),
        InitialDataSine(1.0,(2*π, 2*π)),
        WeakConservationForm(
            mapping_form=SkewSymmetricMapping(),
            inviscid_numerical_flux=LaxFriedrichsNumericalFlux(0.0)),
        Eager())
            
    @test l2 ≈ 0.07409452050788953 atol=tol
    @test conservation ≈ 0.0 atol=tol
    @test energy ≈ 0.0 atol=tol
end

@testset "Advection 2D Standard DGSEM" begin
    (l2, conservation, energy) = test_2d(
        DGSEM(4),Quad(),
        LinearAdvectionEquation((1.0,1.0)), 
        InitialDataSine(1.0,(2*π, 2*π)),
        StrongConservationForm(
            mapping_form=StandardMapping(),
            inviscid_numerical_flux=LaxFriedrichsNumericalFlux()),
            Lazy())
    @test l2 ≈ 0.05018367633381625 atol=tol
    @test conservation ≈ 0.0 atol=tol
    @test energy ≈ -0.008377987714660848 atol=tol
end