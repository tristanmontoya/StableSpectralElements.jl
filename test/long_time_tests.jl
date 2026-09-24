import LinearAlgebra
using StableSpectralElements, OrdinaryDiffEq, .Threads
using Plots, Plots.PlotMeasures, TimerOutputs
using Profile, LinearAlgebra
using SparseArrays
using DelimitedFiles, Dates

const OPTYPE_MAP = Dict(
    "NodalTPSS" => NodalTPSS,
    "NodalTPSSOpt" => NodalTPSSOpt,
    "NodalTPSSLGL" => NodalTPSSLGL,
    "NodalTPSSMinimal" => NodalTPSSMinimal,
    "NodalTPSSOptimal" => NodalTPSSOptimal,
    "ModalTensor" => ModalTensor,
    "ModalOmega" => ModalMulti,
    "NodalTensor" => NodalTensor,
    "NodalOmega" => NodalMulti,
    "ModalDiagE" => ModalMultiDiagE,
    "NodalDiagE" => NodalMultiDiagE,
    "MFNodalTPSSLGL" => MatrixFreeTPSSLGL
)

const ALGO_MAP = Dict(
    "BLAS"      => BLASAlgorithm(),
    "Generic"   => GenericMatrixAlgorithm(),
    "Kronecker" => GenericTensorProductAlgorithm(),
    "Sparse"    => SparseMatrixAlgorithm(),
    "Default"   => DefaultOperatorAlgorithm()
)

function advec2d(opertype::String, algtype::String, comment::String,p, M)

    alg = get(ALGO_MAP, algtype) do
        error("Algorithm '$algtype' not supported. Options are: $(keys(ALGO_MAP))")
    end

    a = (5/4,sqrt(7)/4)  # advection velocity
    L = 1.0  # domain length
    T = 20.0  # end time
    ω = 8.0*π
    conservation_law = LinearAdvectionEquation(a)
    exact_solution= DataSine(1.0, (8.0*π,8.0*π), a);
    #exact_solution = InitialDataGaussian(1.0,0.05,(L/2,L/2),a)
    #exact_solution = InitialDataSine(1.0,(2π/L, 2π/L));

    Dtype = ""
    DTtype = ""
    Vtype = ""
    VTtype = ""
    Rtype = ""
    RTtype = ""
    # set mesh warping
    warp = 0.0

    # set upwind (1.0) or symmetric flux (0.0)
    flux = 1.0

    if haskey(OPTYPE_MAP, opertype)
        operator = OPTYPE_MAP[opertype](p) 
    else
        error("Operator type '$opertype' not supported. Options are: $(keys(OPTYPE_MAP))")
    end    

    #define reference approximation
    reference_approximation = ReferenceApproximation(operator, Tri(), mapping_degree=p)

    # define discretization form, upwind flux
    form = StandardForm(mapping_form=SkewSymmetricMapping(),
        inviscid_numerical_flux=LaxFriedrichsNumericalFlux(flux))

    # define mesh and warp
    uniform_mesh = uniform_periodic_mesh(reference_approximation,
        ((0.0,L),(0.0,L)), (M,M), strategy=OppositeDiagonal())
    mesh = warp_mesh(uniform_mesh, reference_approximation, warp, L)

    # define spatial discretization
    spatial_discretization = SpatialDiscretization(mesh,
        reference_approximation, project_jacobian=true)

    # results path
    results_path = save_project(conservation_law,
        spatial_discretization, exact_solution, form, (0.0, T),
        "results/advection_2d/", overwrite=true, clear=true);

    # discretize to system of ODEs based on alg and operator application
    ode_problem = semidiscretize(conservation_law, spatial_discretization, exact_solution, form, (0.0, T),ReferenceOperator(), alg, parallelism=Serial())

    dt = 1e-3

    sol = solve(ode_problem, RK4(),adaptive=false, dt=dt, save_everystep=false,callback=save_callback(results_path, (0.0,T)))

    # calculate and record error
    err_vec = zeros(Float64, length(sol.u))
    error_analysis = ErrorAnalysis(results_path, conservation_law,
        spatial_discretization)
    for i = 1:length(sol.u)
        err_vec[i] = Float64(analyze(error_analysis, sol.u[i], exact_solution, (i-1)*dt)[1])
    end
    println("Completed run for: "*string(opertype)*"_M"*join(M, "")*"_p"*join(p, ""))
    # save matrix types
    Dtype = typeof(ode_problem.p.operators.D)
    DTtype = typeof(ode_problem.p.operators.Dᵀ)
    Vtype = typeof(ode_problem.p.operators.V)
    VTtype = typeof(ode_problem.p.operators.Vᵀ)
    Rtype = typeof(ode_problem.p.operators.R)
    RTtype = typeof(ode_problem.p.operators.Rᵀ)   

    open("ConvergenceStudies/DEC_1_Results/long_time_tests/2D_linadvec/"*string(opertype)*string(algtype)*"_M"*join(M, "")*"_p"*join(p, "")*string(comment)*".txt", "w") do io
        println(io, "Operator type: ", opertype)
        println(io, "Algorithm: ", algtype)
        println(io, "D type ", Dtype)
        println(io, "Dᵀ type ", DTtype)
        println(io, "V type ", Vtype)
        println(io, "Vᵀ type ", VTtype)
        println(io, "R type ", Rtype)
        println(io, "Rᵀ type ", RTtype)
        println(io, "p: ", p)
        println(io, "M: ", M)
        println(io, "advection velocity: ", a)
        println(io, "omega: ", ω)
        println(io, "Domain length L: ", L)
        println(io, "Upwind (1.0)/symmetric flux (0.0): ", flux)
        println(io, "Mesh warping parameter: ", warp)
        println(io, "End time T: ", T)
        println(io, "dt: ", dt)

        println(io, "Error vector (rows=M_vec, cols=p_vec):")
        writedlm(io, err_vec)
        println(io, "")

    end

    return 0
end

function advec3d(opertype::String, algtype::String, comment::String,p, M)

    alg = get(ALGO_MAP, algtype) do
        error("Algorithm '$algtype' not supported. Options are: $(keys(ALGO_MAP))")
    end

    a = (3/2,1/2,1/sqrt(2))  # advection velocity
    L = 1.0  # domain length
    T = 3.0  # end time
    ω = 4.0*π
    conservation_law = LinearAdvectionEquation(a)
    exact_solution= DataSine(1.0, ω, a);
    #exact_solution = InitialDataSine(1.0,(2π/L, 2π/L));

    Dtype = ""
    DTtype = ""
    Vtype = ""
    VTtype = ""
    Rtype = ""
    RTtype = ""
    # set mesh warping
    warp = 0.0

    # set upwind (1.0) or symmetric flux (0.0)
    flux = 0.0

    if haskey(OPTYPE_MAP, opertype)
        operator = OPTYPE_MAP[opertype](p) 
    else
        error("Operator type '$opertype' not supported. Options are: $(keys(OPTYPE_MAP))")
    end    

    #define reference approximation
    reference_approximation = ReferenceApproximation(operator, Tet(), mapping_degree=p)

    # define discretization form, upwind flux
    form = StandardForm(mapping_form=SkewSymmetricMapping(),
        inviscid_numerical_flux=LaxFriedrichsNumericalFlux(flux))

    # define mesh and warp
    uniform_mesh = uniform_periodic_mesh(reference_approximation,
    ((0.0,L),(0.0,L),(0.0,L)), (M,M,M))
    mesh = warp_mesh(uniform_mesh, reference_approximation, warp, L)

    # define spatial discretization
    spatial_discretization = SpatialDiscretization(mesh,
        reference_approximation, project_jacobian=true)

    # results path
    results_path = save_project(conservation_law,
    spatial_discretization, exact_solution, form, (0.0, T),
    "results/advection_3d/", overwrite=true, clear=true);

    # discretize to system of ODEs based on alg and operator application
    ode_problem = semidiscretize(conservation_law, spatial_discretization, exact_solution, form, (0.0, T),ReferenceOperator(), alg, parallelism=Serial())

    dt = 1e-4

    sol = solve(ode_problem, RK4(),adaptive=false, dt=dt, save_everystep=false,callback=save_callback(results_path, (0.0,T)))

    # calculate and record error
    err_vec = zeros(Float64, length(sol.u))
    error_analysis = ErrorAnalysis(results_path, conservation_law,
        spatial_discretization)
    for i = 1:length(sol.u)
        err_vec[i] = Float64(analyze(error_analysis, sol.u[i], exact_solution, (i-1)*dt)[1])
    end
    println("Completed run for: "*string(opertype)*"_M"*join(M, "")*"_p"*join(p, ""))
    # save matrix types
    Dtype = typeof(ode_problem.p.operators.D)
    DTtype = typeof(ode_problem.p.operators.Dᵀ)
    Vtype = typeof(ode_problem.p.operators.V)
    VTtype = typeof(ode_problem.p.operators.Vᵀ)
    Rtype = typeof(ode_problem.p.operators.R)
    RTtype = typeof(ode_problem.p.operators.Rᵀ)   

    open("ConvergenceStudies/DEC_1_Results/_time_tests/3D_linadvec/"*string(opertype)*string(algtype)*"_M"*join(M, "")*"_p"*join(p, "")*string(comment)*".txt", "w") do io
        println(io, "Operator type: ", opertype)
        println(io, "Algorithm: ", algtype)
        println(io, "D type ", Dtype)
        println(io, "Dᵀ type ", DTtype)
        println(io, "V type ", Vtype)
        println(io, "Vᵀ type ", VTtype)
        println(io, "R type ", Rtype)
        println(io, "Rᵀ type ", RTtype)
        println(io, "p: ", p)
        println(io, "M: ", M)
        println(io, "advection velocity: ", a)
        println(io, "omega: ", ω)
        println(io, "Domain length L: ", L)
        println(io, "Upwind (1.0)/symmetric flux (0.0): ", flux)
        println(io, "Mesh warping parameter: ", warp)
        println(io, "End time T: ", T)
        println(io, "dt: ", dt)

        println(io, "Error vector (rows=M_vec, cols=p_vec):")
        writedlm(io, err_vec)
        println(io, "")

    end

    return 0
end

function euler2d(opertype::String, algtype::String, p, M)
    time_start = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")

    alg = get(ALGO_MAP, algtype) do
        error("Algorithm '$algtype' not supported. Options are: $(keys(ALGO_MAP))")
    end

    mach_number = 0.2
    angle = 0.0
    L = 2.0
    γ=1.4
    T = 3.0 # end time
    strength = sqrt(2/(γ-1)*(1-0.75^(γ-1))) # for central value of ρ=0.75
    offset = L/2
    conservation_law = EulerEquations{2}(γ)
    exact_solution = IsentropicVortex(conservation_law, θ=angle,
        Ma=mach_number, β=strength, R=1.0/10.0, x_0=(offset,L/2));

    Stype = ""
    Vtype = ""
    VTtype = ""
    Rtype = ""
    RTtype = ""
    mesh_warp = 0.0/16

    flux = 0.0 #1.0 for upwind/dissipative

    if haskey(OPTYPE_MAP, opertype)
        operator = OPTYPE_MAP[opertype](p) 
    else
        error("Operator type '$opertype' not supported. Options are: $(keys(OPTYPE_MAP))")
    end    

    reference_approximation = ReferenceApproximation(operator, Tri(), mapping_degree=p)

    form = FluxDifferencingForm(inviscid_numerical_flux=LaxFriedrichsNumericalFlux(flux))

    uniform_mesh = uniform_periodic_mesh(reference_approximation, ((0.0,L),(0.0,L)), (M,M))

    mesh = warp_mesh(uniform_mesh, reference_approximation, ChanWarping(mesh_warp, (L,L)))

    spatial_discretization = SpatialDiscretization(mesh, reference_approximation)

    results_path = save_project(conservation_law,
        spatial_discretization, exact_solution, form, (0.0, T),
        "results/euler_vortex_2d", overwrite=true, clear=true)

    ode_problem = semidiscretize(conservation_law,
        spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), alg, parallelism=Serial())

    dt = 1e-4

    if opertype == "NodalTensor"
        dt = 1e-5
    end
    
    sol = solve(ode_problem, RK4(), adaptive=false, dt=dt, save_everystep=false,
        callback=save_callback(results_path, (0.0,T)))
    
    # calculate and record error
    err_vec = zeros(Float64, length(sol.u))
    error_analysis = ErrorAnalysis(results_path, conservation_law,
        spatial_discretization)
    for i = 1:length(sol.u)
        err_vec[i] = Float64(analyze(error_analysis, sol.u[i], exact_solution, (i-1)*dt)[1])
    end
    println("Completed run for: "*string(opertype)*"_M"*join(M, "")*"_p"*join(p, ""))

    # save matrix types
    Stype = typeof(ode_problem.p.operators.S)
    Vtype = typeof(ode_problem.p.operators.V)
    VTtype = typeof(ode_problem.p.operators.Vᵀ)
    Rtype = typeof(ode_problem.p.operators.R)
    RTtype = typeof(ode_problem.p.operators.Rᵀ)   


    time_end = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    open("ConvergenceStudies/DEC_1_Results/_time_tests/2D_euler/"*string(opertype)*string(algtype)*"_M"*join(M, "")*"_p"*join(p, "")*"euler2D_lambda0.txt", "w") do io
        println(io, "Run ended at: ", time_start)
        println(io, "Run finished at: ", time_end)
        println(io, "Operator type: ", opertype)
        println(io, "Algorithm: ", alg)
        println(io, "S type ", Stype)
        println(io, "V type ", Vtype)
        println(io, "Vᵀ type ", VTtype)
        println(io, "R type ", Rtype)
        println(io, "Rᵀ type ", RTtype)
        println(io, "p: ", p)
        println(io, "M: ", M)
        println(io, "Mach number: ", mach_number)
        println(io, "angle: ", angle)
        println(io, "Domain length L: ", L)
        println(io, "End time T: ", T)
        println(io, "dt: ", dt)
        println(io, "L: ", L)
        println(io, "Offset: ", offset)
        println(io, "Vortex strength: ", strength)
        println(io, "Upwind (1.0)/symmetric flux (0.0): ", flux)
        println(io, "Mesh warping parameter: ", mesh_warp)

        println(io, "Error vector (rows=M_vec, cols=p_vec):")
        writedlm(io, err_vec)


    end
    return 0

end

function euler3d(opertype::String, algtype::String, p, M)

    alg = get(ALGO_MAP, algtype) do
        error("Algorithm '$algtype' not supported. Options are: $(keys(ALGO_MAP))")
    end

    L = 2.0
    T = 0.2 # end time
    Ω = ((0.0,L),(0.0,L),(0.0,L))
    conservation_law = EulerEquations{3}(1.4)
    exact_solution = EulerCosineWave(conservation_law);

    Stype = ""
    Vtype = ""
    VTtype = ""
    Rtype = ""
    RTtype = ""
    mesh_warp = 1.0/32

    flux = 1.0 #1.0 for upwind/dissipative

    if haskey(OPTYPE_MAP, opertype)
        operator = OPTYPE_MAP[opertype](p) 
    else
        error("Operator type '$opertype' not supported. Options are: $(keys(OPTYPE_MAP))")
    end    

    form = FluxDifferencingForm(inviscid_numerical_flux=LaxFriedrichsNumericalFlux(1.0))

    reference_approximation = ReferenceApproximation(operator, Tet(), mapping_degree=p)
            
    uniform_mesh = uniform_periodic_mesh(reference_approximation, Ω, (M,M,M))
    
    mesh = warp_mesh(uniform_mesh, reference_approximation, ChanWarping(mesh_warp, (L,L,L)))
    
    spatial_discretization = SpatialDiscretization(mesh, reference_approximation,
        ChanWilcoxMetrics())

    results_path = save_project(conservation_law,
        spatial_discretization, exact_solution, form, (0.0, T),
        "results/euler_periodic_3d/", overwrite=true, clear=true);
    
    ode_problem = semidiscretize(conservation_law, spatial_discretization, 
        exact_solution, form, (0.0, T), ReferenceOperator(), alg,parallelism=Serial());

    dt = 1e-4

    if opertype == "NodalTensor"
        dt = 1e-5
    end
    
    sol = solve(ode_problem, RK4(), adaptive=false, dt=dt, save_everystep=false,
        callback=save_callback(results_path, (0.0,T)))
    
    # calculate and record error
    err_vec = zeros(Float64, length(sol.u))
    error_analysis = ErrorAnalysis(results_path, conservation_law,
        spatial_discretization)
    for i = 1:length(sol.u)
        err_vec[i] = Float64(analyze(error_analysis, sol.u[i], exact_solution, (i-1)*dt)[1])
    end
    println("Completed run for: "*string(opertype)*"_M"*join(M, "")*"_p"*join(p, ""))

    # save matrix types
    Stype = typeof(ode_problem.p.operators.S)
    Vtype = typeof(ode_problem.p.operators.V)
    VTtype = typeof(ode_problem.p.operators.Vᵀ)
    Rtype = typeof(ode_problem.p.operators.R)
    RTtype = typeof(ode_problem.p.operators.Rᵀ)   


    time_end = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    open("ConvergenceStudies/DEC_1_Results/long_time_tests/3D_euler/"*string(opertype)*string(algtype)*"_M"*join(M, "")*"_p"*join(p, "")*"euler3D.txt", "w") do io
        println(io, "Operator type: ", opertype)
        println(io, "Algorithm: ", alg)
        println(io, "S type ", Stype)
        println(io, "V type ", Vtype)
        println(io, "Vᵀ type ", VTtype)
        println(io, "R type ", Rtype)
        println(io, "Rᵀ type ", RTtype)
        println(io, "p: ", p)
        println(io, "M: ", M)
        println(io, "Domain length L: ", L)
        println(io, "End time T: ", T)
        println(io, "dt: ", dt)
        println(io, "L: ", L)
        println(io, "Upwind (1.0)/symmetric flux (0.0): ", flux)
        println(io, "Mesh warping parameter: ", mesh_warp)

        println(io, "Error vector (rows=M_vec, cols=p_vec):")
        writedlm(io, err_vec)


    end
    return 0

end
comment = "oppDiag_8pi_warp0p0_T20p0"
#advec2d("NodalTPSSLGL", "Sparse", comment,4, 6)
advec2d("ModalTensor", "Sparse", comment,4, 16)
advec2d("NodalTensor", "Sparse", comment,4, 16)
advec2d("ModalOmega", "Default", comment,4, 16)
advec2d("NodalOmega", "Default", comment,4, 16)
advec2d("ModalDiagE", "Default", comment,4, 16)
#advec2d("NodalDiagE", "Default", comment,4, 16)
#=
advec2d("NodalTPSSLGL", "Sparse", "lambda0_loweromega",8, 4)
advec2d("ModalTensor", "Sparse", "lambda0_loweromega",8, 8)
advec2d("NodalTensor", "Sparse", "lambda0_loweromega",8, 8)
advec2d("ModalOmega", "Default", "lambda0_loweromega",8, 8)
advec2d("NodalOmega", "Default", "lambda0_loweromega",8, 8)
advec2d("ModalDiagE", "Default", "lambda0_loweromega",8, 8)
advec2d("NodalDiagE", "Default", "lambda0_loweromega",8, 8) =#

#= euler2d("NodalTPSSLGL", "Sparse",4, 4)
euler2d("ModalTensor", "Sparse",4, 8)
euler2d("NodalTensor", "Sparse",4, 8)
euler2d("ModalOmega", "Default",4, 8)
euler2d("NodalOmega", "Default",4, 8)
euler2d("ModalDiagE", "Default",4, 8)
euler2d("NodalDiagE", "Default",4, 8) =#

#= euler2d("NodalTPSSLGL", "Sparse",8, 4)
euler2d("ModalTensor", "Sparse",8, 8)
euler2d("NodalTensor", "Sparse",8, 8)
euler2d("ModalOmega", "Default",8, 8)
euler2d("NodalOmega", "Default",8, 8)
euler2d("ModalDiagE", "Default",8, 8)
euler2d("NodalDiagE", "Default",8, 8) =#

#= advec3d("NodalTPSSLGL", "Sparse", "lambda0",2, 1)
advec3d("ModalTensor", "Sparse", "lambda0",2, 3)
advec3d("NodalTensor", "Sparse", "lambda0",2, 3)
advec3d("ModalOmega", "Default", "lambda0",2, 3)
advec3d("NodalOmega", "Default", "lambda0",2, 3)
advec3d("ModalDiagE", "Default", "lambda0",2, 3)
advec3d("NodalDiagE", "Default", "lambda0",2, 3)

advec3d("NodalTPSSLGL", "Sparse", "lambda0",5, 1)
advec3d("ModalTensor", "Sparse", "lambda0",5, 3)
advec3d("NodalTensor", "Sparse", "lambda0",5, 3)
advec3d("ModalOmega", "Default", "lambda0",5, 3)
advec3d("NodalOmega", "Default", "lambda0",5, 3)
advec3d("ModalDiagE", "Default", "lambda0",5, 3)
advec3d("NodalDiagE", "Default", "lambda0",5, 3) =#

#= euler3d("NodalTPSSLGL", "Sparse",2, 1)
euler3d("ModalTensor", "Sparse",2, 2)
euler3d("NodalTensor", "Sparse",2, 2)
euler3d("ModalOmega", "Default",2, 2)
euler3d("NodalOmega", "Default",2, 2)
euler3d("ModalDiagE", "Default",2, 2)
euler3d("NodalDiagE", "Default",2, 2) =#

#euler3d("NodalTPSSLGL", "Sparse",5, 1)
#euler3d("ModalTensor", "Sparse",5, 3)
#euler2d("NodalTensor", "Sparse",5, 3)
#euler3d("ModalOmega", "Default",5, 3)
#euler3d("NodalOmega", "Default",5, 3)
#euler3d("ModalDiagE", "Default",5, 3)
#euler3d("NodalDiagE", "Default",5, 3)