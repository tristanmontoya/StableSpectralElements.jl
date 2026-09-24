using StableSpectralElements, OrdinaryDiffEq, .Threads
using Profile, ProfileCanvas,StatProfilerHTML
using Plots, Plots.PlotMeasures, TimerOutputs
using DelimitedFiles,Dates

function advec2d(opertype::String, alg::String, p, M)
    a = (5/4,sqrt(7)/4)  # advection velocity
    #a = (1.0,1.0)  # advection velocity
    L = 1.0  # domain length
    T = 1e-2  # end time
    ω = 4.0*π
    conservation_law = LinearAdvectionEquation(a)
    exact_solution= DataSine(1.0, ω, a);

    # set mesh warping
    warp = 0.0

    # set upwind (1.0) or symmetric flux (0.0)
    flux = 1.0

    #define reference approximation
    if opertype == "NodalTPSS"
        reference_approximation = ReferenceApproximation(
            NodalTPSS(p), Tri(), mapping_degree=p)
    elseif opertype == "NodalTPSSOpt"
        reference_approximation = ReferenceApproximation(
            NodalTPSSOpt(p), Tri(), mapping_degree=p)
    elseif opertype == "NodalTPSSLGL"
        reference_approximation = ReferenceApproximation(
            NodalTPSSLGL(p), Tri(), mapping_degree=p)
    elseif opertype == "NodalTPSSMinimal"
        reference_approximation = ReferenceApproximation(
            NodalTPSSMinimal(p), Tri(), mapping_degree=p)
    elseif opertype == "NodalTPSSOptimal"
        reference_approximation = ReferenceApproximation(
            NodalTPSSOptimal(p), Tri(), mapping_degree=p)
    elseif opertype == "ModalTensor"
        reference_approximation = ReferenceApproximation(
            ModalTensor(p), Tri(), mapping_degree=p)
    elseif opertype == "ModalOmega"
        reference_approximation = ReferenceApproximation(
            ModalMulti(p), Tri(), mapping_degree=p)
    elseif opertype == "NodalTensor"
        reference_approximation = ReferenceApproximation(
            NodalTensor(p), Tri(), mapping_degree=p)
    elseif opertype == "NodalOmega"
        reference_approximation = ReferenceApproximation(
            NodalMulti(p), Tri(), mapping_degree=p)
    elseif opertype == "ModalDiagE"
        reference_approximation = ReferenceApproximation(
            ModalMultiDiagE(p), Tri(), mapping_degree=p)
    elseif opertype == "NodalDiagE"
        reference_approximation = ReferenceApproximation(
            NodalMultiDiagE(p), Tri(), mapping_degree=p)
    elseif opertype == "MFNodalTPSSLGL"
        reference_approximation = ReferenceApproximation(
            MatrixFreeTPSSLGL(p), Tri(), mapping_degree=p)
    else
        error("opertype not supported")
    end

    # define discretization form, upwind flux
    form = StandardForm(mapping_form=SkewSymmetricMapping(),
        inviscid_numerical_flux=LaxFriedrichsNumericalFlux(flux))

    # define mesh and warp
    uniform_mesh = uniform_periodic_mesh(reference_approximation,
        ((0.0,L),(0.0,L)), (M,M))
    mesh = warp_mesh(uniform_mesh, reference_approximation, warp, L)

    # define spatial discretization
    spatial_discretization = SpatialDiscretization(mesh,
        reference_approximation, project_jacobian=true)

    # results path
    results_path = save_project(conservation_law,
        spatial_discretization, exact_solution, form, (0.0, T),
        "results/advection_2d/", overwrite=true, clear=true);

    # discretize to system of ODEs
    if alg == "BLAS"
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T),ReferenceOperator(), BLASAlgorithm())
    elseif alg == "Generic"
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T),ReferenceOperator(), GenericMatrixAlgorithm())            
    elseif alg == "Kronecker"
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), GenericTensorProductAlgorithm())
    elseif alg == "Default" #(octavian)
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), DefaultOperatorAlgorithm())
    elseif alg == "Sparse" 
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), SparseMatrixAlgorithm())
    else
        error("algorithm not supported")
    end
    dt = 1e-5
    sol = solve(ode_problem, RK4(),
    adaptive=false, dt=dt, save_everystep=false,
    callback=save_callback(results_path, (0.0,T)))
    #=     error_analysis = ErrorAnalysis(results_path, conservation_law,
    spatial_discretization)
    err = Float64(analyze(error_analysis, last(sol.u), exact_solution, T)[1])
    println(err) =#
    #println(typeof(ode_problem.p.operators.R))
end

function advec3d(opertype::String, alg::String, p, M)
    a = (3/2,1/2,1/sqrt(2))  # advection velocity
    L = 1.0  # domain length
    T = 1e-2  # end time
    ω = 2.0*π
    conservation_law = LinearAdvectionEquation(a)
    exact_solution= DataSine(1.0, ω, a);

    # set mesh warping
    warp = 0.0

    # set upwind (1.0) or symmetric flux (0.0)
    flux = 1.0

    #define reference approximation
    if opertype == "NodalTPSS"
        reference_approximation = ReferenceApproximation(
            NodalTPSS(p), Tet(), mapping_degree=p)
    elseif opertype == "NodalTPSSOpt"
        reference_approximation = ReferenceApproximation(
            NodalTPSSOpt(p), Tet(), mapping_degree=p)
    elseif opertype == "NodalTPSSLGL"
        reference_approximation = ReferenceApproximation(
            NodalTPSSLGL(p), Tet(), mapping_degree=p)
    elseif opertype == "MFNodalTPSSLGL"
        reference_approximation = ReferenceApproximation(
            MatrixFreeTPSSLGL(p), Tet(), mapping_degree=p)
    elseif opertype == "NodalTPSSMinimal"
        reference_approximation = ReferenceApproximation(
            NodalTPSSMinimal(p), Tet(), mapping_degree=p)
    elseif opertype == "NodalTPSSOptimal"
        reference_approximation = ReferenceApproximation(
            NodalTPSSOptimal(p), Tet(), mapping_degree=p)
    elseif opertype == "ModalTensor"
        reference_approximation = ReferenceApproximation(
            ModalTensor(p), Tet(), mapping_degree=p)
    elseif opertype == "ModalOmega"
        reference_approximation = ReferenceApproximation(
            ModalMulti(p), Tet(), mapping_degree=p)
    elseif opertype == "ModalDiagE"
        reference_approximation = ReferenceApproximation(
            ModalMultiDiagE(p), Tet(), mapping_degree=p)
    elseif opertype == "NodalDiagE"
        reference_approximation = ReferenceApproximation(
            NodalMultiDiagE(p), Tet(), mapping_degree=p)
    elseif opertype == "NodalOmega"
        reference_approximation = ReferenceApproximation(
            NodalMulti(p), Tet(), mapping_degree=p)
    elseif opertype == "NodalTensor"
        reference_approximation = ReferenceApproximation(
            NodalTensor(p), Tet(), mapping_degree=p)
    else
        error("opertype not supported")
    end

    # define discretization form, upwind flux
    form = StandardForm(mapping_form=SkewSymmetricMapping(),
        inviscid_numerical_flux=LaxFriedrichsNumericalFlux(flux))

    # define mesh and warp
    uniform_mesh = uniform_periodic_mesh(reference_approximation,
    ((0.0,L),(0.0,L),(0.0,L)), (M,M,M))
    mesh = warp_mesh(uniform_mesh, reference_approximation, warp, L)

    spatial_discretization = SpatialDiscretization(mesh,
    reference_approximation, ExactMetrics())

    results_path = save_project(conservation_law,
    spatial_discretization, exact_solution, form, (0.0, T),
    "results/advection_3d/", overwrite=true, clear=true);

    dt = 1e-5

    if alg == "BLAS"
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), BLASAlgorithm())
    elseif alg == "Generic"
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), GenericMatrixAlgorithm())            
    elseif alg == "Kronecker"
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), GenericTensorProductAlgorithm())
    elseif alg == "Default" #(octavian)
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), DefaultOperatorAlgorithm())
    elseif alg == "Sparse" 
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), SparseMatrixAlgorithm())
    else
        error("algorithm not supported")
    end

    solve(ode_problem, RK4(), adaptive=false, dt=dt,
    save_everystep=false,
    callback=save_callback(results_path, (0.0,0.01)))

end

function euler2d(opertype::String, alg::String, p, M)

    mach_number = 0.8
    angle = 0.0
    L = 1.0
    γ=1.4
    rho_central = 0.75
    T = L/mach_number # end time
    strength = sqrt(2/(γ-1)*(1-rho_central^(γ-1)))

    conservation_law = EulerEquations{2}(γ)
    exact_solution = IsentropicVortex(conservation_law, θ=angle,
        Ma=mach_number, β=strength, R=1.0/10.0, x_0=(L/2,L/2));

    mesh_warp = 1.0/16

    flux = 1.0 #1.0 for upwind/dissipative


    #define reference approximation
    if opertype == "NodalTPSS"
        reference_approximation = ReferenceApproximation(
            NodalTPSS(p), Tri(), mapping_degree=p)
    elseif opertype == "NodalTPSSOpt"
        reference_approximation = ReferenceApproximation(
            NodalTPSSOpt(p), Tri(), mapping_degree=p)
    elseif opertype == "NodalTPSSLGL"
        reference_approximation = ReferenceApproximation(
            NodalTPSSLGL(p), Tri(), mapping_degree=p)
    elseif opertype == "NodalTPSSMinimal"
        reference_approximation = ReferenceApproximation(
            NodalTPSSMinimal(p), Tri(), mapping_degree=p)
    elseif opertype == "NodalTPSSOptimal"
        reference_approximation = ReferenceApproximation(
            NodalTPSSOptimal(p), Tri(), mapping_degree=p)
    elseif opertype == "ModalTensor"
        reference_approximation = ReferenceApproximation(
            ModalTensor(p), Tri(), mapping_degree=p)
    elseif opertype == "ModalOmega"
        reference_approximation = ReferenceApproximation(
            ModalMulti(p), Tri(), mapping_degree=p)
    elseif opertype == "NodalTensor"
        reference_approximation = ReferenceApproximation(
            NodalTensor(p), Tri(), mapping_degree=p)
    elseif opertype == "NodalOmega"
        reference_approximation = ReferenceApproximation(
            NodalMulti(p), Tri(), mapping_degree=p)
    elseif opertype == "ModalDiagE"
        reference_approximation = ReferenceApproximation(
            ModalMultiDiagE(p), Tri(), mapping_degree=p)
    elseif opertype == "NodalDiagE"
        reference_approximation = ReferenceApproximation(
            NodalMultiDiagE(p), Tri(), mapping_degree=p)
    elseif opertype == "MFNodalTPSSLGL"
        reference_approximation = ReferenceApproximation(
            MatrixFreeTPSSLGL(p), Tri(), mapping_degree=p)
    else
        error("opertype not supported")
    end

    form = FluxDifferencingForm(inviscid_numerical_flux=LaxFriedrichsNumericalFlux(flux))

    uniform_mesh = uniform_periodic_mesh(reference_approximation, ((0.0,L),(0.0,L)), (M,M))

    mesh = warp_mesh(uniform_mesh, reference_approximation, ChanWarping(mesh_warp, (L,L)))

    spatial_discretization = SpatialDiscretization(mesh, reference_approximation)

    results_path = save_project(conservation_law,
        spatial_discretization, exact_solution, form, (0.0, T),
        "results/euler_vortex_2d", overwrite=true, clear=true)

    if alg == "BLAS"
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), BLASAlgorithm())
    elseif alg == "Generic"
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), GenericMatrixAlgorithm())            
    elseif alg == "Kronecker"
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), GenericTensorProductAlgorithm())
    elseif alg == "Default"
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), DefaultOperatorAlgorithm())
    elseif alg == "Sparse"
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), SparseMatrixAlgorithm())
    else
        error("algorithm not supported")
    end

    dt = 1e-3
    sol = solve(ode_problem, RK4(), adaptive=false, dt=dt, save_everystep=false,
        callback=save_callback(results_path, (0.0,0.0), 5))

    println(size(reference_approximation.R))
    println(typeof(ode_problem.p.operators.R))
end

function euler3d(opertype::String, alg::String, p, M)
    
    L = 2.0
    T = 1e-4
    Ω = ((0.0,L),(0.0,L),(0.0,L))
    
    conservation_law = EulerEquations{3}(1.4)
    exact_solution = EulerPeriodicTest(conservation_law);

    form = FluxDifferencingForm(inviscid_numerical_flux=LaxFriedrichsNumericalFlux(1.0))

    #define reference approximation
    if opertype == "NodalTPSS"
        reference_approximation = ReferenceApproximation(
            NodalTPSS(p), Tet(), mapping_degree=p)
    elseif opertype == "NodalTPSSOpt"
        reference_approximation = ReferenceApproximation(
            NodalTPSSOpt(p), Tet(), mapping_degree=p)
    elseif opertype == "NodalTPSSLGL"
        reference_approximation = ReferenceApproximation(
            NodalTPSSLGL(p), Tet(), mapping_degree=p)
    elseif opertype == "MFNodalTPSSLGL"
        reference_approximation = ReferenceApproximation(
            MatrixFreeTPSSLGL(p), Tet(), mapping_degree=p)
    elseif opertype == "NodalTPSSMinimal"
        reference_approximation = ReferenceApproximation(
            NodalTPSSMinimal(p), Tet(), mapping_degree=p)
    elseif opertype == "NodalTPSSOptimal"
        reference_approximation = ReferenceApproximation(
            NodalTPSSOptimal(p), Tet(), mapping_degree=p)
    elseif opertype == "ModalTensor"
        reference_approximation = ReferenceApproximation(
            ModalTensor(p), Tet(), mapping_degree=p)
    elseif opertype == "ModalOmega"
        reference_approximation = ReferenceApproximation(
            ModalMulti(p), Tet(), mapping_degree=p)
    elseif opertype == "ModalDiagE"
        reference_approximation = ReferenceApproximation(
            ModalMultiDiagE(p), Tet(), mapping_degree=p)
    elseif opertype == "NodalDiagE"
        reference_approximation = ReferenceApproximation(
            NodalMultiDiagE(p), Tet(), mapping_degree=p)
    elseif opertype == "NodalOmega"
        reference_approximation = ReferenceApproximation(
            NodalMulti(p), Tet(), mapping_degree=p)
    elseif opertype == "NodalTensor"
        reference_approximation = ReferenceApproximation(
            NodalTensor(p), Tet(), mapping_degree=p)
    else
        error("opertype not supported")
    end

    uniform_mesh = uniform_periodic_mesh(reference_approximation, Ω, (M,M,M))

    mesh = warp_mesh(uniform_mesh, reference_approximation, ChanWarping(0.0/16, (L,L,L)))

    spatial_discretization = SpatialDiscretization(mesh, reference_approximation,
        ChanWilcoxMetrics())

    results_path = save_project(conservation_law,
        spatial_discretization, exact_solution, form, (0.0, T),
        "results/euler_TGV_3d/", overwrite=true, clear=true);
    
    dt = 1e-5

    if alg == "BLAS"
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), BLASAlgorithm())
    elseif alg == "Generic"
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), GenericMatrixAlgorithm())            
    elseif alg == "Kronecker"
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), GenericTensorProductAlgorithm())
    elseif alg == "Default" #(octavian)
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), DefaultOperatorAlgorithm())
    elseif alg == "Sparse"
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), SparseMatrixAlgorithm())
    else
        error("algorithm not supported")
    end

    solve(ode_problem, RK4(), adaptive=false, dt=dt,
    save_everystep=false,
    callback=save_callback(results_path, (0.0,0.0)))
    println(typeof(ode_problem.p.operators.R))
end

Profile.init(n=10^8, delay=0.0001)
#advec2d("ModalTensor","Default",[4],[4])
#ProfileCanvas.@profview advec2d("ModalTensor","Default",[4],[4])

#start_time = time()
##euler3d("ModalTensor","Sparse",3,1)
#end_time = time()
#println("Execution time: ", end_time - start_time, " seconds")

#ProfileCanvas.@profview euler3d("NodalTensor","Sparse",3,3)
#start_time = time()
#advec2d("ModalTensor","Sparse",3,12)
#end_time = time()
#println("Execution time: ", end_time - start_time, " seconds")
ProfileCanvas.@profview euler2d("NodalOmega","Default", 4,16)

#start_time = time()
#euler3d("ModalTensor","Sparse",3,8)
#end_time = time()
#println("Execution time: ", end_time - start_time, " seconds")