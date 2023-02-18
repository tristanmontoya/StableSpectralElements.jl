push!(LOAD_PATH,"../")

using Test, CLOUD, OrdinaryDiffEq, UnPack

const tol = 1.0e-10
const p = 4

function test_driver(
    reference_approximation::ReferenceApproximation,
    conservation_law::AbstractConservationLaw,
    initial_data::AbstractGridFunction{d},
    form::AbstractResidualForm,
    operator_algorithm::AbstractOperatorAlgorithm,
    strategy::AbstractStrategy,
    L::Float64,
    M::Int,
    T::Float64,
    dt::Float64,
    test_name::String) where {d}

    @unpack reference_element = reference_approximation

    if d == 1 spatial_discretization = SpatialDiscretization(           
        uniform_periodic_mesh(reference_element, 
            (0.0,L),M), reference_approximation)
    else spatial_discretization = SpatialDiscretization(
        warp_mesh(uniform_periodic_mesh(reference_element,
            Tuple((0.0,L) for m in 1:d), Tuple(M for m in 1:d)),
            reference_element, 0.1), reference_approximation)
    end

    exact_solution = ExactSolution(conservation_law,initial_data)

    results_path = save_project(conservation_law,
        spatial_discretization, initial_data, form, (0.0, T),
        string("results/", test_name,"/"), overwrite=true, clear=true)

    ode_problem = semidiscretize(conservation_law,
        spatial_discretization,
        initial_data, 
        form,
        (0.0, T),
        strategy,
        operator_algorithm)

    sol = solve(ode_problem, CarpenterKennedy2N54(),
        adaptive=false, dt=dt, callback=save_callback(results_path, (0.0,T),  floor(Int, 1.0/(dt*50))))

    error = analyze(ErrorAnalysis(results_path, conservation_law, 
        spatial_discretization), last(sol.u), exact_solution, 1.0)
    conservation = analyze(PrimaryConservationAnalysis(results_path, 
        conservation_law, spatial_discretization), 
        load_time_steps(results_path))
    energy = analyze(EnergyConservationAnalysis(results_path,
        conservation_law, spatial_discretization),
        load_time_steps(results_path))

    return (error..., maximum(abs.(conservation.dEdt[:,1])), 
        maximum(energy.dEdt[:,1]))
end

@testset "Advection-Diffusion 1D ModalMulti" begin
    (l2, conservation, energy) = test_driver(
        ReferenceApproximation(ModalMulti(p), Line()), 
        LinearAdvectionDiffusionEquation(1.0,5.0e-2),
        InitialDataSine(1.0,2π),
        WeakConservationForm(
            mapping_form=StandardMapping(),
            inviscid_numerical_flux=LaxFriedrichsNumericalFlux(),
            viscous_numerical_flux=BR1()),
        GenericMatrixAlgorithm(),
        PhysicalOperator(), 1.0, 4, 1.0, 1.0/100.0, 
        "test_advection_diffusion_1d_dgmulti")

    @test l2 ≈ 6.988216111585663e-6 atol=tol
    @test conservation ≈ 0.0 atol=tol
    @test energy <= 0.0
end

@testset "Advection 2D Energy-Conservative ModalTensor Tri" begin
    (l2, conservation, energy) = test_driver(
        ReferenceApproximation(ModalTensor(p), Tri(), volume_quadrature_rule=(LGQuadrature(p),LGRQuadrature(p)),mapping_degree=p), 
        LinearAdvectionEquation((1.0,1.0)),
        InitialDataSine(1.0,(2*π, 2*π)),
        WeakConservationForm(
            mapping_form=SkewSymmetricMapping(),
            inviscid_numerical_flux=LaxFriedrichsNumericalFlux(0.0)),
        DefaultOperatorAlgorithm(),
        ReferenceOperator(), 1.0, 2, 1.0, 1.0/100.0,
        "test_advection_2d_collapsed_econ")
    println(l2,conservation, energy)
    
    @test l2 ≈ 0.2669410042275925 atol=tol
    @test conservation ≈ 0.0 atol=tol
    @test energy ≈ 0.0 atol=tol
end

@testset "Advection 2D Standard NodalTensor Quad" begin
    (l2, conservation, energy) = test_driver(
        ReferenceApproximation(NodalTensor(p), Quad(), mapping_degree=p,
        volume_quadrature_rule=LGLQuadrature(p), facet_quadrature_rule=LGLQuadrature(p)), 
        LinearAdvectionEquation((1.0,1.0)),
        InitialDataSine(1.0,(2*π, 2*π)),
        WeakConservationForm(
            mapping_form=StandardMapping(),
            inviscid_numerical_flux=LaxFriedrichsNumericalFlux(1.0)),
        BLASAlgorithm(),
        ReferenceOperator(), 1.0, 2, 1.0, 1.0/100.0,
        "test_advection_2d_dgsem_standard")
    
    @test l2 ≈ 0.04930951393074396 atol=tol
    @test conservation ≈ 0.0 atol=tol
    @test energy <= 0.0
end

@testset "Advection 3D Energy-Conservative NodalTensor Hex" begin
    (l2, conservation, energy) = test_driver(
        ReferenceApproximation(NodalTensor(p), Hex(), mapping_degree=p,
        volume_quadrature_rule=LGLQuadrature(p), facet_quadrature_rule=LGLQuadrature(p)), 
        LinearAdvectionEquation((1.0,1.0,1.0)),
        InitialDataSine(1.0,(2*π, 2*π, 2*π)),
        WeakConservationForm(
            mapping_form=SkewSymmetricMapping(),
            inviscid_numerical_flux=LaxFriedrichsNumericalFlux(0.0)),
        DefaultOperatorAlgorithm(),
        ReferenceOperator(), 1.0, 2, 1.0, 1.0/100.0,
        "test_advection_3d_dgsem_econ")
    
    @test l2 ≈ 0.07897065489635471 atol=tol
    @test conservation ≈ 0.0 atol=tol
    @test energy≈ 0.0 atol=tol
end