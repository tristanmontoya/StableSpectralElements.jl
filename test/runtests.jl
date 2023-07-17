push!(LOAD_PATH,"../")
using Test, StableSpectralElements, OrdinaryDiffEq

include("test_driver.jl")
include("burgers_fluxdiff_1d.jl")
include("euler_vortex_2d.jl")

const tol = 1.0e-10
const p = 4

@testset "Advection-Diffusion 1D ModalMulti" begin
    (l2, conservation, _) = test_driver(
        ReferenceApproximation(ModalMulti(p), Line()), 
        LinearAdvectionDiffusionEquation(1.0,5.0e-2),
        InitialDataSine(1.0,2π),
        StandardForm(
            mapping_form=StandardMapping(),
            inviscid_numerical_flux=LaxFriedrichsNumericalFlux(),
            viscous_numerical_flux=BR1()),
        GenericMatrixAlgorithm(),
        PhysicalOperator(), 1.0, 4, 1.0, 1.0/100.0, 0.1,
        "test_advection_diffusion_1d_dgmulti")

    @test l2 ≈ 6.988216111585663e-6 atol=tol
    @test conservation ≈ 0.0 atol=tol
end

@testset "Advection 2D Energy-Conservative ModalTensor Tri" begin
    (l2, conservation, energy) = test_driver(
        ReferenceApproximation(ModalTensor(p), Tri(), volume_quadrature_rule=(LGQuadrature(p),LGRQuadrature(p)),mapping_degree=p), 
        LinearAdvectionEquation((1.0,1.0)),
        InitialDataSine(1.0,(2*π, 2*π)),
        StandardForm(
            mapping_form=SkewSymmetricMapping(),
            inviscid_numerical_flux=LaxFriedrichsNumericalFlux(0.0)),
        GenericTensorProductAlgorithm(),
        ReferenceOperator(), 1.0, 2, 1.0, 1.0/100.0, 0.1,
        "test_advection_2d_collapsed_econ")

    @test l2 ≈ 0.2669410042275925 atol=tol
    @test conservation ≈ 0.0 atol=tol
    @test energy ≈ 0.0 atol=tol
end

@testset "Advection 2D FluxDiff NodalTensor Quad" begin
    (l2, conservation, _) = test_driver(
        ReferenceApproximation(NodalTensor(p), Quad(), mapping_degree=p,
        volume_quadrature_rule=LGLQuadrature(p), facet_quadrature_rule=LGLQuadrature(p)), 
        LinearAdvectionEquation((1.0,1.0)),
        InitialDataSine(1.0,(2*π, 2*π)),
        FluxDifferencingForm(
            inviscid_numerical_flux=LaxFriedrichsNumericalFlux(1.0)),
        BLASAlgorithm(),
        ReferenceOperator(), 1.0, 2, 1.0, 1.0/100.0, 0.1,
        "test_advection_2d_dgsem_fluxdiff")
    
    @test l2 ≈ 0.04790536605026519 atol=tol
    @test conservation ≈ 0.0 atol=tol
end

@testset "Inviscid Burgers FluxDiff 1D" begin
    (conservation, energy) = burgers_fluxdiff_1d()

    @test conservation ≈ 0.0 atol=tol
    @test energy ≈ 0.0 atol=tol
end

@testset "Isentropic Euler vortex FluxDiff NodalTensor Quad 2D" begin
    (l2, conservation, entropy) = euler_vortex_2d()

    @test l2 ≈ [0.004741291979358405, 0.10346265859644528, 0.10168232125253537,
        0.4339693000910734] atol=tol
    @test conservation ≈ [0.0, 0.0, 0.0, 0.0] atol=tol
    @test entropy ≈ 0.0 atol=tol
end

@testset "Advection 3D Energy-Conservative ModalTensor Tet" begin
    (l2, conservation, energy) = test_driver(
        ReferenceApproximation(ModalTensor(p), Tet(), mapping_degree=3),
        LinearAdvectionEquation((1.0,1.0,1.0)),
        InitialDataSine(1.0,(2*π, 2*π, 2*π)),
        StandardForm(
            mapping_form=SkewSymmetricMapping(),
            inviscid_numerical_flux=LaxFriedrichsNumericalFlux(0.0)),
        DefaultOperatorAlgorithm(),
        ReferenceOperator(), 1.0, 2, 1.0, 1.0/50.0, 0.1,
        "test_advection_3d_collapsed_econ")
    
    @test l2 ≈ 0.18683083992263985 atol=tol
    @test conservation ≈ 0.0 atol=tol
    @test energy ≈ 0.0 atol=tol
end