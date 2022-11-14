push!(LOAD_PATH,"../")

using Test
using CLOUD

tol = 1.0e-10

include("test_1d.jl")
include("test_2d.jl")

# integration tests
@testset "Advection-Diffusion 1D ModalMulti" begin
    (l2, conservation, energy) = test_1d(
        ModalMulti(4),Line(), 
        LinearAdvectionDiffusionEquation(1.0,5.0e-2),
        InitialDataSine(1.0,2π),
        WeakConservationForm(
            mapping_form=StandardMapping(),
            inviscid_numerical_flux=LaxFriedrichsNumericalFlux(),
            viscous_numerical_flux=BR1()),
        BLASAlgorithm(),
        PhysicalOperator(), 4, "test_advection_diffusion_1d_dgmulti")
    
    @test l2 ≈ 6.988470301621085e-6 atol=tol
    @test conservation ≈ 0.0 atol=tol
    @test energy ≈ -0.24517593114338798 atol=tol
end

@testset "Advection 2D Energy-Conservative ModalTensor" begin
    (l2, conservation, energy) = test_2d(
        ModalTensor(4),Tri(), 
        LinearAdvectionEquation((1.0,1.0)),
        InitialDataSine(1.0,(2*π, 2*π)),
        WeakConservationForm(
            mapping_form=SkewSymmetricMapping(),
            inviscid_numerical_flux=LaxFriedrichsNumericalFlux(0.0)),
        DefaultOperatorAlgorithm(),
        ReferenceOperator(), 2, "test_advection_2d_collapsed_econ")
        
    @test l2 ≈ 0.0793011361997203 atol=tol
    @test conservation ≈ 0.0 atol=tol
    @test energy ≈ 0.0 atol=tol
end

@testset "Advection 2D Standard NodalTensor" begin
    (l2, conservation, energy) = test_2d(
        NodalTensor(4),Quad(),
        LinearAdvectionEquation((1.0,1.0)), 
        InitialDataSine(1.0,(2*π, 2*π)),
        StrongConservationForm(
            mapping_form=StandardMapping(),
            inviscid_numerical_flux=LaxFriedrichsNumericalFlux()),
            GenericMatrixAlgorithm(),
            ReferenceOperator(), 2, "test_advection_2d_dgsem_standard")

    @test l2 ≈ 0.050183676333818025 atol=tol
    @test conservation ≈ 0.0 atol=tol
    @test energy ≈ -0.00837798771466075 atol=tol
end