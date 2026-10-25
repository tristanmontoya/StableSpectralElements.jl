# create A_v*f + A_SAT * f_SAT where A_v applies the volume differentiation and A_SAT applies the SAT terms, both matrices are sparse. 
import LinearAlgebra
using StableSpectralElements, OrdinaryDiffEq, .Threads
using Plots, Plots.PlotMeasures, TimerOutputs
using Profile, LinearAlgebra
using SparseArrays
using DelimitedFiles, Dates

const OPTYPE_MAP = Dict(
    "NodalTPSS" => NodalTPSS,
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

const OPAPP_MAP = Dict(
    "Physical"  => PhysicalOperator(),
    "Reference" => ReferenceOperator()
)

function advec2d(opertype::String, algtype::String, app::String, comment::String,p_vec, M_vec)
    time_start = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    println("Starting run for 2D advection with $opertype at ", time_start)
    alg = get(ALGO_MAP, algtype) do
        error("Algorithm '$algtype' not supported. Options are: $(keys(ALGO_MAP))")
    end
    opapp = get(OPAPP_MAP, app) do
        error("Algorithm '$app' not supported. Options are: $(keys(OPAPP_MAP))")
    end
    a = (5/4,sqrt(7)/4)  # advection velocity
    #a = (1.0,1.0)  # advection velocity
    L = 1.0  # domain length
    T = 1.0 # end time
    ω = (8*pi,4*pi)  # frequency vector
    conservation_law = LinearAdvectionEquation(a)
    exact_solution = DataSine(1.0, ω, a);
    #exact_solution= DataManySine(1.0, ω, a);
    #exact_solution = InitialDataGaussian(1.0,0.05,(L/2,L/2),a)
    #exact_solution = InitialDataSine(1.0,(2π/L, 2π/L));

    err_vec = zeros(length(M_vec),length(p_vec))
    dof_vec = zeros(length(M_vec),length(p_vec))
    dt_vec = zeros(length(M_vec),length(p_vec))
    setuptime_vec = zeros(length(M_vec),length(p_vec))
    runtime_vec = zeros(length(M_vec),length(p_vec))

    Dtype = ""
    DTtype = ""
    Vtype = ""
    VTtype = ""
    Rtype = ""
    RTtype = ""
    # set mesh warping
    warp = 1.0/32.0

    # set upwind (1.0) or symmetric flux (0.0)
    flux = 1.0

    # accumulator for p
    pcnt = 1        
    # loop over p
    for p in p_vec

        if haskey(OPTYPE_MAP, opertype)
            operator = OPTYPE_MAP[opertype](p) 
        else
            error("Operator type '$opertype' not supported. Options are: $(keys(OPTYPE_MAP))")
        end    
        # accumulator for M
        Mcnt = 1

        # loop over meshes
        for M in M_vec

            ts_start = time()
            #define reference approximation
            reference_approximation = ReferenceApproximation(operator, Tri(), mapping_degree=p)

            # define discretization form, upwind flux
            form = StandardForm(mapping_form=SkewSymmetricMapping(),
                inviscid_numerical_flux=LaxFriedrichsNumericalFlux(flux))

            # define mesh and warp
            uniform_mesh = uniform_periodic_mesh(reference_approximation,
                ((0.0,L),(0.0,L)), (M,M), strategy=OppositeDiagonal())
            #mesh = warp_mesh(uniform_mesh, reference_approximation, warp, L)
            mesh = warp_mesh(uniform_mesh, reference_approximation, ChanWarping(warp, (L,L)))

            # define spatial discretization
            spatial_discretization = SpatialDiscretization(mesh,
                reference_approximation, project_jacobian=true)

            # results path
            results_path = save_project(conservation_law,
                spatial_discretization, exact_solution, form, (0.0, T),
                "results/advection_2d/", overwrite=true, clear=true);

            # discretize to system of ODEs based on alg and operator application
            ode_problem = semidiscretize(conservation_law, spatial_discretization, exact_solution, form, (0.0, T),opapp, alg,parallelism=Serial())
        
            dof = size(ode_problem.u0)[1]*size(ode_problem.u0)[3]
            dof_vec[Mcnt,pcnt]= dof

            dt = 1e-4
            dt_vec[Mcnt,pcnt]=dt

            ts_end = time()
            setup_time = ts_end - ts_start
            setuptime_vec[Mcnt,pcnt] = setup_time
            println("Setup for p=$p, M=$M finished in ", round(setup_time, digits=1), " seconds.")

            # Dummy warm-up problem with a short tspan
            ode_dummy = remake(ode_problem, tspan=(0.0, 2 * dt))

            # Warm-up solve: failures here are not caught
            sol = solve(ode_dummy, RK4(),adaptive=false, dt=dt, save_everystep=false)

            # Timed solve: catch failures
            tr_start = time_ns()
            solve_failed = false
            failure_message = ""
            try
            sol = solve(ode_problem, RK4(),adaptive=false, dt=dt, save_everystep=false)

            catch e
                solve_failed = true

                # Capture the exception and stack trace as a string
                failure_message = sprint(showerror, e, catch_backtrace())
                println("Timed solve FAILED for p=$p, M=$M")
                println(failure_message)
            end
            tr_end = time_ns()

            if solve_failed
                # Record failed case
                runtime_vec[Mcnt, pcnt] = NaN
                err_vec[Mcnt, pcnt] = NaN
                failure_messages[Mcnt, pcnt] = failure_message
                println("Skipping remaining M values for p=$p.")

                # Stop the current M loop and proceed to the next p
                break

            else
                # Record successful runtime
                run_time = (tr_end - tr_start) / 1e9
                runtime_vec[Mcnt, pcnt] = run_time
                println("Run for p=$p, M=$M finished in ",round(run_time, digits=1), " seconds.")

                # Calculate and record error
                error_analysis = ErrorAnalysis(results_path,conservation_law,spatial_discretization)
                err = Float64(analyze(error_analysis, last(sol.u), exact_solution, T)[1])
                err_vec[Mcnt, pcnt] = err
                println("Error is $err and DOF is $dof")

            end
            
            # save matrix types
            Dtype = typeof(ode_problem.p.operators.D)
            DTtype = typeof(ode_problem.p.operators.Dᵀ)
            Vtype = typeof(ode_problem.p.operators.V)
            VTtype = typeof(ode_problem.p.operators.Vᵀ)
            Rtype = typeof(ode_problem.p.operators.R)
            RTtype = typeof(ode_problem.p.operators.Rᵀ)   
            # update accumulator
            Mcnt +=1
        end # end loop for M 

        # update accumulator
        pcnt += 1

     
    end # end loop for p

    time_end = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    open("ConvergenceStudies/2D_advec/"*string(opertype)*string(algtype)*string(app)*"_M"*join(M_vec, "")*"_p"*join(p_vec, "")*string(comment)*".txt", "w") do io
        println(io, "Run ended at: ", time_start)
        println(io, "Run finished at: ", time_end)
        println(io, "Operator type: ", opertype)
        println(io, "Algorithm: ", algtype)
        println(io, "D type ", Dtype)
        println(io, "Dᵀ type ", DTtype)
        println(io, "V type ", Vtype)
        println(io, "Vᵀ type ", VTtype)
        println(io, "R type ", Rtype)
        println(io, "Rᵀ type ", RTtype)
        println(io, "Operator Application: ", opapp)
        println(io, "p_vec: ", p_vec)
        println(io, "M_vec: ", M_vec)
        println(io, "advection velocity: ", a)
        println(io, "omega: ", ω)
        println(io, "Domain length L: ", L)
        println(io, "End time T: ", T)
        println(io, "Upwind (1.0)/symmetric flux (0.0): ", flux)
        println(io, "Mesh warping parameter: ", warp)

        println(io, "Error matrix (rows=M_vec, cols=p_vec):")
        writedlm(io, err_vec)
        println(io, "")

        println(io, "DOF matrix (rows=M_vec, cols=p_vec):")
        writedlm(io, dof_vec)

        println(io, "dt matrix (rows=M_vec, cols=p_vec):")
        writedlm(io,dt_vec)

        println(io, "Setup times (seconds) for each (M,p) pair (rows=M_vec, cols=p_vec):")
        writedlm(io, setuptime_vec)

        println(io, "Run times (seconds) for each (M,p) pair (rows=M_vec, cols=p_vec):")
        writedlm(io, runtime_vec)

    end

    return err_vec, dof_vec

end

function advec3d(opertype::String, algtype::String, app::String, comment::String, p_vec, M_vec)
    time_start = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    println("Starting run for 3D advection $opertype at ", time_start)
    alg = get(ALGO_MAP, algtype) do
        error("Algorithm '$algtype' not supported. Options are: $(keys(ALGO_MAP))")
    end
    opapp = get(OPAPP_MAP, app) do
        error("Algorithm '$app' not supported. Options are: $(keys(OPAPP_MAP))")
    end
    a = (3/2,1/2,1/sqrt(2))  # advection velocity
    L = 1.0  # domain length
    T = 0.2  # end time
    conservation_law = LinearAdvectionEquation(a)
    exact_solution = InitialDataGaussian(1.0,0.05,(0.5,0.5,0.5),a)
    ω="Gaussian"
    #ω=(2.0*π,4.0*π,2.0*π)
    #exact_solution= DataSine(1.0, ω, a);

    err_vec = zeros(length(M_vec),length(p_vec))
    dof_vec = zeros(length(M_vec),length(p_vec))
    dt_vec = zeros(length(M_vec),length(p_vec))
    setuptime_vec = zeros(length(M_vec),length(p_vec))
    runtime_vec = zeros(length(M_vec),length(p_vec))
    residual_time_vec = zeros(length(M_vec),length(p_vec))
    Dtype = ""
    DTtype = ""
    Vtype = ""
    VTtype = ""
    Rtype = ""
    RTtype = ""
    # set mesh warping
    warp = 1.0/32.0

    # set upwind (1.0) or symmetric flux (0.0)
    flux = 1.0

    # accumulator for p
    pcnt = 1 
        # loop over p
        for p in p_vec
            if haskey(OPTYPE_MAP, opertype)
                operator = OPTYPE_MAP[opertype](p) 
            else
                error("Operator type '$opertype' not supported. Options are: $(keys(OPTYPE_MAP))")
            end    
            # accumulator for M
            Mcnt = 1

            # loop over meshes
            for M in M_vec
                ts_start = time()
                #define reference approximation
                reference_approximation = ReferenceApproximation(operator, Tet(), mapping_degree=p)

                # define discretization form, upwind flux
                form = StandardForm(mapping_form=SkewSymmetricMapping(),
                inviscid_numerical_flux=LaxFriedrichsNumericalFlux(flux))

                # define mesh and warp
                uniform_mesh = uniform_periodic_mesh(reference_approximation,
                ((0.0,L),(0.0,L),(0.0,L)), (M,M,M))
                #mesh = warp_mesh(uniform_mesh, reference_approximation, warp, L)
                mesh = warp_mesh(uniform_mesh, reference_approximation, ChanWarping(warp, (L,L,L)))

                spatial_discretization = SpatialDiscretization(mesh,
                reference_approximation, ChanWilcoxMetrics())

                results_path = save_project(conservation_law,
                spatial_discretization, exact_solution, form, (0.0, T),
                "results/advection_3d/", overwrite=true, clear=true);

                # discretize to system of ODEs based on alg and operator application
                ode_problem = semidiscretize(conservation_law, spatial_discretization, exact_solution, form, (0.0, T),opapp, alg,parallelism=Serial())
            
                dof = size(ode_problem.u0)[1]*size(ode_problem.u0)[3]
                dof_vec[Mcnt,pcnt]= dof

                dt = 5e-4
                dt_vec[Mcnt,pcnt]=dt

                ts_end = time()
                setup_time = ts_end - ts_start
                setuptime_vec[Mcnt,pcnt] = setup_time
                println("Setup for p=$p, M=$M finished in ", round(setup_time, digits=1), " seconds.")

                # dummy warm-up problem with a short tspan
                ode_dummy = remake(ode_problem, tspan=(0.0, 2 * dt))
                
                # solve twice to reduce julia compilation time effects, only time the second run
                sol = solve(ode_dummy, RK4(),adaptive=false, dt=dt, save_everystep=false)
                tr_start = time_ns()
                sol = solve(ode_problem, RK4(),adaptive=false, dt=dt, save_everystep=false)
                tr_end = time_ns()

                run_time = (tr_end - tr_start) / 1e9
                runtime_vec[Mcnt,pcnt] = run_time
                println("Run for p=$p, M=$M finished in ", round(run_time, digits=1), " seconds.")
        
                # calculate and record error
                error_analysis = ErrorAnalysis(results_path, conservation_law,
                    spatial_discretization)
                err = Float64(analyze(error_analysis, last(sol.u), exact_solution, T)[1])
                err_vec[Mcnt,pcnt] = err
                
                println("Error is $err and DOF is $dof")

                # save matrix types
                Dtype = typeof(ode_problem.p.operators.D)
                DTtype = typeof(ode_problem.p.operators.Dᵀ)
                Vtype = typeof(ode_problem.p.operators.V)
                VTtype = typeof(ode_problem.p.operators.Vᵀ)
                Rtype = typeof(ode_problem.p.operators.R)
                RTtype = typeof(ode_problem.p.operators.Rᵀ)   

                # update accumulator
                Mcnt +=1
            end # end loop for M 

            # update accumulator
            pcnt += 1
        end # end loop for p
  
    time_end = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    open("ConvergenceStudies/3D_advec/"*string(opertype)*string(algtype)*string(app)*"_M"*join(M_vec, "")*"_p"*join(p_vec, "")*string(comment)*".txt", "w") do io
        println(io, "Run ended at: ", time_start)
        println(io, "Run finished at: ", time_end)
        println(io, "Operator type: ", opertype)
        println(io, "Algorithm: ", algtype)
        println(io, "Operator Application: ", opapp)
        println(io, "D type ", Dtype)
        println(io, "Dᵀ type ", DTtype)
        println(io, "V type ", Vtype)
        println(io, "Vᵀ type ", VTtype)
        println(io, "R type ", Rtype)
        println(io, "Rᵀ type ", RTtype)
        println(io, "p_vec: ", p_vec)
        println(io, "M_vec: ", M_vec)
        println(io, "advection velocity: ", a)
        println(io, "omega: ", ω)
        println(io, "Domain length L: ", L)
        println(io, "End time T: ", T)
        println(io, "Upwind (1.0)/symmetric flux (0.0): ", flux)
        println(io, "Mesh warping parameter: ", warp)

        println(io, "Error matrix (rows=M_vec, cols=p_vec):")
        writedlm(io, err_vec)
        println(io, "")

        println(io, "DOF matrix (rows=M_vec, cols=p_vec):")
        writedlm(io, dof_vec)

        println(io, "dt matrix (rows=M_vec, cols=p_vec):")
        writedlm(io,dt_vec)

        println(io, "Setup times (seconds) for each (M,p) pair (rows=M_vec, cols=p_vec):")
        writedlm(io, setuptime_vec)

        println(io, "Run times (seconds) for each (M,p) pair (rows=M_vec, cols=p_vec):")
        writedlm(io, runtime_vec)

        println(io, "Residual times (ms) for each (M,p) pair (rows=M_vec, cols=p_vec):")
        writedlm(io, residual_time_vec)
    end

    return err_vec, dof_vec

end

function euler2d(opertype::String, algtype::String, comment::String, p_vec, M_vec)
    time_start = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    println("Starting run for 2D Euler with $opertype at ", time_start)
    alg = get(ALGO_MAP, algtype) do
        error("Algorithm '$algtype' not supported. Options are: $(keys(ALGO_MAP))")
    end

    #sigma = 0.05
    #amplitude = 0.5
    L = 1.0
    γ=1.4
    mach_number = 0.8
    angle = 0.0
    T = L/mach_number # end time
    strength = sqrt(2/(γ-1)*(1-0.75^(γ-1))) # for central value of ρ=0.75
    
    conservation_law = EulerEquations{2}(γ)
    exact_solution = IsentropicVortex(conservation_law, θ=angle,
        Ma=mach_number, β=strength, R=1.0/10.0, x_0=(L/2,L/2));
    #exact_solution = EulerGaussianPulse(conservation_law, amplitude, sigma)
    err_vec = zeros(length(M_vec),length(p_vec))
    dof_vec = zeros(length(M_vec),length(p_vec))
    dt_vec = zeros(length(M_vec),length(p_vec))
    runtime_vec = zeros(length(M_vec),length(p_vec))
    Stype = ""
    Vtype = ""
    VTtype = ""
    Rtype = ""
    RTtype = ""
    mesh_warp = 1.0/32

    flux = 1.0 #1.0 for upwind/dissipative
    # accumulator for p
    pcnt = 1

    # loop over p
    for p in p_vec
        if haskey(OPTYPE_MAP, opertype)
            operator = OPTYPE_MAP[opertype](p) 
        else
            error("Operator type '$opertype' not supported. Options are: $(keys(OPTYPE_MAP))")
        end    
        # accumulator for M
        Mcnt = 1

        # loop over meshes
        for M in M_vec
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

            dof = size(ode_problem.u0)[1]*size(ode_problem.u0)[3]
            dof_vec[Mcnt,pcnt]= dof

            #dt = C_t * (L/M) / (mach_number*p^2)
            dt = 1e-4
            dt_vec[Mcnt,pcnt]=dt
            
            # Dummy warm-up problem with a short tspan
            ode_dummy = remake(ode_problem, tspan=(0.0, 2 * dt))

            # Warm-up solve: failures here are not caught
            sol = solve(ode_dummy, RK4(),adaptive=false, dt=dt, save_everystep=false)

            # Timed solve: catch failures
            tr_start = time_ns()
            solve_failed = false
            failure_message = ""
            try
            sol = solve(ode_problem, RK4(),adaptive=false, dt=dt, save_everystep=false)

            catch e
                solve_failed = true

                # Capture the exception and stack trace as a string
                failure_message = sprint(showerror, e, catch_backtrace())
                println("Timed solve FAILED for p=$p, M=$M")
                println(failure_message)
            end
            tr_end = time_ns()

            if solve_failed
                # Record failed case
                runtime_vec[Mcnt, pcnt] = NaN
                err_vec[Mcnt, pcnt] = NaN
                #failure_messages[Mcnt, pcnt] = failure_message
                println("Skipping remaining M values for p=$p.")

                # Stop the current M loop and proceed to the next p
                break

            else
                # Record successful runtime
                run_time = (tr_end - tr_start) / 1e9
                runtime_vec[Mcnt, pcnt] = run_time
                println("Run for p=$p, M=$M finished in ",round(run_time, digits=1), " seconds.")

                # Calculate and record error
                error_analysis = ErrorAnalysis(results_path,conservation_law,spatial_discretization)
                err = Float64(analyze(error_analysis, last(sol.u), exact_solution, T)[1])
                err_vec[Mcnt, pcnt] = err
                println("Error is $err and DOF is $dof")

            end

            # save matrix types
            Stype = typeof(ode_problem.p.operators.S)
            Vtype = typeof(ode_problem.p.operators.V)
            VTtype = typeof(ode_problem.p.operators.Vᵀ)
            Rtype = typeof(ode_problem.p.operators.R)
            RTtype = typeof(ode_problem.p.operators.Rᵀ)   

            # update accumulator
            Mcnt +=1
        end

        # update accumulator
        pcnt += 1
    end

    time_end = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    open("ConvergenceStudies/2D_euler/"*string(opertype)*string(algtype)*"_M"*join(M_vec, "")*"_p"*join(p_vec, "")*string(comment)*".txt", "w") do io
        println(io, "Run ended at: ", time_start)
        println(io, "Run finished at: ", time_end)
        println(io, "Operator type: ", opertype)
        println(io, "Algorithm: ", alg)
        println(io, "S type ", Stype)
        println(io, "V type ", Vtype)
        println(io, "Vᵀ type ", VTtype)
        println(io, "R type ", Rtype)
        println(io, "Rᵀ type ", RTtype)
        println(io, "p_vec: ", p_vec)
        println(io, "M_vec: ", M_vec)
        println(io, "Mach number: ", mach_number)
        println(io, "Angle: ", angle)
        println(io, "Domain length L: ", L)
        println(io, "End time T: ", T)
        println(io, "Vortex strength: ", strength)
        println(io, "Upwind (1.0)/symmetric flux (0.0): ", flux)
        println(io, "Mesh warping parameter: ", mesh_warp)

        println(io, "Error matrix (rows=M_vec, cols=p_vec):")
        writedlm(io, err_vec)
        println(io, "")

        println(io, "DOF matrix (rows=M_vec, cols=p_vec):")
        writedlm(io, dof_vec)
        println(io, "")

        println(io, "dt matrix (rows=M_vec, cols=p_vec):")
        writedlm(io,dt_vec)
        println(io, "")

        println(io, "Run times (seconds) for each (M,p) pair (rows=M_vec, cols=p_vec):")
        writedlm(io, runtime_vec)

    end
    return err_vec, dof_vec

end

function euler3d(opertype::String, algtype::String, comment::String,p_vec, M_vec)
    time_start = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    println("Starting run for 3D Euler with $opertype at ", time_start)
    alg = get(ALGO_MAP, algtype) do
        error("Algorithm '$algtype' not supported. Options are: $(keys(ALGO_MAP))")
    end

    L = 1.0
    T = 0.2
    Ω = ((0.0,L),(0.0,L),(0.0,L))
    
    conservation_law = EulerEquations{3}(1.4)
    exact_solution = EulerCosineWave(conservation_law);

    err_vec = zeros(length(M_vec),length(p_vec))
    dof_vec = zeros(length(M_vec),length(p_vec))
    dt_vec = zeros(length(M_vec),length(p_vec))
    runtime_vec = zeros(length(M_vec),length(p_vec))
    Stype = ""
    Vtype = ""
    VTtype = ""
    Rtype = ""
    RTtype = ""
    mesh_warp = 1.0/32

    flux = 1.0 #1.0 for upwind/dissipative
    # accumulator for p
    pcnt = 1

    # loop over p
    for p in p_vec
        if haskey(OPTYPE_MAP, opertype)
            operator = OPTYPE_MAP[opertype](p) 
        else
            error("Operator type '$opertype' not supported. Options are: $(keys(OPTYPE_MAP))")
        end    
        # accumulator for M
        Mcnt = 1

        # loop over meshes
        for M in M_vec
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

            dof = size(ode_problem.u0)[1]*size(ode_problem.u0)[3]
            dof_vec[Mcnt,pcnt]= dof

            #dt = C_t * (L/M) / (mach_number*p^2)
            dt = 5e-4
            dt_vec[Mcnt,pcnt]=dt
            # dummy warm-up problem with a short tspan
            ode_dummy = remake(ode_problem, tspan=(0.0, 2 * dt)) 
            
            # double solve to reduce julia compilation time effects, only time the second run
            sol = solve(ode_dummy, RK4(), adaptive=false, dt=dt, save_everystep=false)

            # clear any garbage
            GC.gc()

            tr_start = time_ns()
            sol = solve(ode_problem, RK4(), adaptive=false, dt=dt, save_everystep=false)
            tr_end = time_ns()
            run_time = (tr_end - tr_start) / 1e9

            runtime_vec[Mcnt,pcnt] = run_time
            println("Run for p=$p, M=$M finished in ", round(run_time, digits=1), " seconds.")

            error_analysis = ErrorAnalysis(results_path, conservation_law,
                spatial_discretization)

            err=Float64(analyze(error_analysis, last(sol.u), exact_solution, T)[1])
            err_vec[Mcnt,pcnt] = err

            println("Error is $err and DOF is $dof")

            # save matrix types
            Stype = typeof(ode_problem.p.operators.S)
            Vtype = typeof(ode_problem.p.operators.V)
            VTtype = typeof(ode_problem.p.operators.Vᵀ)
            Rtype = typeof(ode_problem.p.operators.R)
            RTtype = typeof(ode_problem.p.operators.Rᵀ)   

            # update accumulator
            Mcnt +=1
        end

        # update accumulator
        pcnt += 1
    end

    time_end = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    open("ConvergenceStudies/3D_euler/"*string(opertype)*string(algtype)*"_M"*join(M_vec, "")*"_p"*join(p_vec, "")*string(comment)*".txt", "w") do io
        println(io, "Run ended at: ", time_start)
        println(io, "Run finished at: ", time_end)
        println(io, "Operator type: ", opertype)
        println(io, "Algorithm: ", alg)
        println(io, "S type ", Stype)
        println(io, "V type ", Vtype)
        println(io, "Vᵀ type ", VTtype)
        println(io, "R type ", Rtype)
        println(io, "Rᵀ type ", RTtype)
        println(io, "p_vec: ", p_vec)
        println(io, "M_vec: ", M_vec)
        println(io, "Domain length L: ", L)
        println(io, "End time T: ", T)
        println(io, "Upwind (1.0)/symmetric flux (0.0): ", flux)
        println(io, "Mesh warping parameter: ", mesh_warp)

        println(io, "Error matrix (rows=M_vec, cols=p_vec):")
        writedlm(io, err_vec)
        println(io, "")

        println(io, "DOF matrix (rows=M_vec, cols=p_vec):")
        writedlm(io, dof_vec)
        println(io, "")

        println(io, "dt matrix (rows=M_vec, cols=p_vec):")
        writedlm(io,dt_vec)
        println(io, "")

        println(io, "Run times (seconds) for each (M,p) pair (rows=M_vec, cols=p_vec):")
        writedlm(io, runtime_vec)

    end
    return err_vec, dof_vec

end

comment = ""
#advec2d("NodalTPSSLGL", "Sparse", "Reference", comment, [1,2,3,4,5,6,7,8], [4,8,12,16])
#advec2d("NodalTensor", "Sparse", "Reference", comment, [1,2,3,4,5,6,7,8], [8,12,16,20,24])
#advec2d("NodalDiagE", "Default", "Reference", comment, [1,2,3,4,5,6,7,8], [8,12,16,20,24])
#advec2d("NodalOmega", "Default", "Reference", comment, [1,2,3,4,5,6,7,8], [8,12,16,20,24])
#advec2d("ModalTensor", "Sparse", "Reference", comment, [1,2,3,4,5,6,7,8], [8,12,16,20,24])
#advec2d("NodalTPSSOptimal", "Sparse", "Reference", comment, [1,2,3,4,5], [2,4,6,8,10])
#advec2d("NodalTPSSMinimal", "Sparse", "Reference", comment, [1,2,3,4,5], [2,4,6,8,10])

#euler2d("NodalTPSSLGL", "Sparse", comment, [1,2,3,4,5], [4,8,12,16])
euler2d("NodalTensor", "Sparse", comment, [1,2,3,4,5], [12,16,20,24])
euler2d("NodalDiagE", "Default", comment, [1,2,3,4,5], [12,16,20,24])
euler2d("NodalOmega", "Default", comment, [1,2,3,4,5], [12,16,20,24])
euler2d("ModalTensor", "Sparse", comment, [1,2,3,4,5], [12,16,20,24])
euler2d("NodalTPSSOptimal", "Sparse", comment, [1,2,3,4,5], [3,5,7,9])
euler2d("NodalTPSSMinimal", "Sparse", comment, [1,2,3,4,5], [3,5,7,9])


euler2d("NodalTPSSLGL", "Sparse", comment, [6,7,8], [4,6,8,10])
euler2d("NodalTensor", "Sparse", comment, [6,7,8], [4,8,12,16])
euler2d("NodalDiagE", "Default", comment, [6,7,8], [4,8,12,16])
euler2d("NodalOmega", "Default", comment, [6,7,8], [4,8,12,16])
euler2d("ModalTensor", "Sparse", comment, [6,7,8], [4,8,12,16])

#=try
    advec2d("NodalTPSSOptimal", "Sparse", "Reference", comment,[2,3,4,5,6], [1,2,3,4])
catch e
    @warn "TPSS Opt failed" exception=(e, catch_backtrace())
end
try
    advec2d("NodalTPSSMinimal", "Sparse", "Reference", comment, [2,3,4,5,6], [1,2,3,4])
catch e
    @warn "TPSS Min failed" exception=(e, catch_backtrace())
end
try
    advec2d("NodalTensor", "Sparse", "Reference", comment, [1,2,3,4,5,6], [8,12,16,24])
catch e
    @warn "NodalTensor failed" exception=(e, catch_backtrace())
end =#

#advec3d("NodalTPSSLGL", "Sparse", "Reference", [1,2,3,4,5], [5])
#advec3d("NodalTPSSOptimal", "Sparse", "Reference", [3,4,5], [1,2])
#advec3d("NodalTPSSMinimal", "Sparse", "Reference", [3,4,5], [1,2])
#advec3d("NodalDiagE", "Default", "Reference", [1,2,3,4,5], [10]) 
#advec3d("NodalOmega", "Default", "Reference", [1,2,3,4,5], [10])
#try
#advec3d("NodalTensor", "Sparse", "Reference", [1,2,3], [10])
#catch e
#    @warn "NodalTensor failed" exception=(e, catch_backtrace())
#end
#advec3d("ModalDiagE", "Default", "Reference", [1,2,3,4,5], [10]) 
#advec3d("ModalOmega", "Default", "Reference", [1,2,3,4,5], [10])
#advec3d("ModalTensor", "Sparse", "Reference", [1,2,3,4,5], [10])
#advec3d("ModalTensor", "Sparse", "Reference", [6,7], [2,4,6,8])

#= euler2d("NodalTPSSLGL", "Sparse",  [1,2,3,4], [4,6,8,12])
euler2d("NodalDiagE", "Default",  [1,2,3,4], [8,12,16,24])
euler2d("NodalOmega", "Default",  [1,2,3,4], [8,12,16,24])
euler2d("ModalDiagE", "Default",  [1,2,3,4], [8,12,16,24])
euler2d("ModalOmega", "Default",  [1,2,3,4], [8,12,16,24])
euler2d("ModalTensor", "Sparse", [1,2,3,4], [8,12,16,24])
try
    euler2d("NodalTensor", "Sparse", [1,2,3,4], [8,12,16,24])
catch e
    @warn "NodalTensor failed" exception=(e, catch_backtrace())
end    

euler2d("NodalDiagE", "Default",  [5,6,7,8], [8,10,12,16])
euler2d("NodalOmega", "Default",  [5,6,7,8], [8,10,12,16])
euler2d("ModalDiagE", "Default",  [5,6,7,8], [8,10,12,16])
euler2d("ModalOmega", "Default",  [5,6,7,8], [8,10,12,16])
euler2d("ModalTensor", "Sparse", [5,6,7,8], [8,10,12,16])
#try
#    euler2d("NodalTensor", "Sparse", [5,6,7,8], [4,6,8])
#catch e
#    @warn "NodalTensor failed" exception=(e, catch_backtrace())
#end
euler2d("NodalTPSSLGL", "Sparse",  [5,6,7,8], [4,5,6,8])
try
euler2d("NodalTPSSOptimal", "Sparse", [2,3,4,5,6], [1,2,3,4])
catch e
    @warn "TPSS Opt failed" exception=(e, catch_backtrace())
end
try
euler2d("NodalTPSSMinimal", "Sparse", [2,3,4,5,6], [1,2,3,4])
catch e
    @warn "TPSS Min failed" exception=(e, catch_backtrace())
end =#
#euler3d("ModalTensor", "Sparse", "T0p2warp1016", [1,2,3,4,5], [2,4,6,8])
#euler3d("ModalOmega","Default", "T0p2warp1016", [1,2,3,4,5], [2,4,6,8])
#euler3d("NodalOmega", "Default", "T0p2warp1016", [1,2,3,4,5], [2,4,6,8])
#euler3d("ModalDiagE","Default", "T0p2warp1016", [1,2,3,4,5], [2,4,6,8])
#euler3d("NodalDiagE", "Default", "T0p2warp1016", [1,2,3,4,5], [2,4,6,8])
#euler3d("NodalTPSSLGL", "Sparse", "T0p2warp1016", [1,2,3,4,5], [1,2,3,4])
#euler3d("NodalTensor", "Sparse", "T1p0", [1,2,3,4,5], [2,4,6])

