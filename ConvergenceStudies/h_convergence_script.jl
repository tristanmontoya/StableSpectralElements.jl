using StableSpectralElements, OrdinaryDiffEq, .Threads
using Plots, Plots.PlotMeasures, TimerOutputs
using DelimitedFiles,Dates

function advec2d(opertype::String, p_vec, M_vec)
    time_start = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    a = (2.0,2.0)  # advection velocity
    L = 1.0  # domain length
    T = 0.5  # end time
    conservation_law = LinearAdvectionEquation(a)
    exact_solution = InitialDataSine(1.0,(2π/L, 2π/L));

    # set mesh refinement levels
    err_vec = zeros(length(M_vec),length(p_vec))
    dof_vec = zeros(length(M_vec),length(p_vec))
    dt_vec = zeros(length(M_vec),length(p_vec))
    setuptime_vec = zeros(length(M_vec),length(p_vec))
    runtime_vec = zeros(length(M_vec),length(p_vec))

    # set CFL number
    CFL = 0.1

    # set mesh warping
    warp = 0.1

    # set upwind (1.0) or symmetric flux (0.0)
    flux = 1.0

    # accumulator for p
    pcnt = 1

    # loop over p
    for p in p_vec

        # accumulator for M
        Mcnt = 1

        # loop over meshes
        for M in M_vec
            ts_start = time()
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
            elseif opertype == "ModalDiagE"
                reference_approximation = ReferenceApproximation(
                    ModalMultiDiagE(p), Tri(), mapping_degree=p)
            elseif opertype == "NodalDiagE"
                reference_approximation = ReferenceApproximation(
                    NodalMultiDiagE(p), Tri(), mapping_degree=p)
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
            ode_problem = semidiscretize(conservation_law,
                spatial_discretization, exact_solution, form, (0.0, T))
            dof = size(ode_problem.u0)[1]*size(ode_problem.u0)[3]
            dof_vec[Mcnt,pcnt]= dof
            # time step definition
            h = L/sqrt(reference_approximation.N_p * spatial_discretization.N_e)
            dt = CFL * h / sqrt(a[1]^2 + a[2]^2)
            dt_vec[Mcnt,pcnt]=dt

            ts_end = time()
            setup_time = ts_end - ts_start
            setuptime_vec[Mcnt,pcnt] = setup_time
            println("Setup for p=$p, M=$M finished in ", round(setup_time, digits=1), " seconds.")

            # solve
            tr_start = time()
            sol = solve(ode_problem, DP5(),
            adaptive=false, dt=dt, save_everystep=false,
            callback=save_callback(results_path, (0.0,T), floor(Int, T/(dt*50))))
            tr_end = time()
            run_time = tr_end - tr_start
            runtime_vec[Mcnt,pcnt] = run_time
            println("Run for p=$p, M=$M finished in ", round(run_time, digits=1), " seconds.")
    
            # calculate and record error
            error_analysis = ErrorAnalysis(results_path, conservation_law,
                spatial_discretization, DefaultQuadrature(50))
            err = Float64(analyze(error_analysis, last(sol.u), exact_solution, T)[1])
            err_vec[Mcnt,pcnt] = err
            
            println("Error is $err and DOF is $dof")
            # update accumulator
            Mcnt +=1
        end

        # update accumulator
        pcnt += 1
    end

    time_end = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    open("ConvergenceStudies/linadvec2d_"*string(opertype)*"_M"*join(M_vec, "")*"_p"*join(p_vec, "")*"_CFL"*join(replace.(string.(CFL), "." => "p"), "")*"_highQuadrature"*".txt", "w") do io
        println(io, "Run ended at: ", time_start)
        println(io, "Run finished at: ", time_end)
        println(io, "Operator type: ", opertype)
        println(io, "p_vec: ", p_vec)
        println(io, "M_vec: ", M_vec)
        println(io, "advection velocity: ", a)
        println(io, "Domain length L: ", L)
        println(io, "End time T: ", T)
        println(io, "CFL ", CFL)
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

function advec3d(opertype::String, p_vec, M_vec)
    time_start = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    a = (1.0,1.0,1.0)  # advection velocity
    L = 1.0  # domain length
    T = 1.0  # end time
    conservation_law = LinearAdvectionEquation(a)
    exact_solution = InitialDataCosine(1.0,(2π/L, 2π/L, 2π/L));

    err_vec = zeros(length(M_vec),length(p_vec))
    dof_vec = zeros(length(M_vec),length(p_vec))
    dt_vec = zeros(length(M_vec),length(p_vec))
    setuptime_vec = zeros(length(M_vec),length(p_vec))
    runtime_vec = zeros(length(M_vec),length(p_vec))
    # set CFL number
    CFL = 0.1

    # set mesh warping
    warp = 0.1

    # set upwind (1.0) or symmetric flux (0.0)
    flux = 1.0

    # accumulator for p
    pcnt = 1

    # loop over p
    for p in p_vec

        # accumulator for M
        Mcnt = 1

        # loop over meshes
        for M in M_vec
            ts_start = time()
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

            # discretize to system of ODEs
            h = L/(reference_approximation.N_p * spatial_discretization.N_e)^(1/3)
            dt = CFL * h / sqrt(a[1]^2 + a[2]^2 + a[3]^2)
            dt_vec[Mcnt,pcnt]=dt

            ode_problem = semidiscretize(conservation_law, spatial_discretization,
            exact_solution, form, (0.0, T), ReferenceOperator());
            dof = size(ode_problem.u0)[1]*size(ode_problem.u0)[3]
            dof_vec[Mcnt,pcnt]= dof

            ts_end = time()
            setup_time = ts_end - ts_start
            setuptime_vec[Mcnt,pcnt] = setup_time
            println("Setup for p=$p, M=$M finished in ", round(setup_time, digits=1), " seconds.")

            # solve
            tr_start = time()
            sol = solve(ode_problem, CarpenterKennedy2N54(), adaptive=false, dt=dt,
            save_everystep=false,
            callback=save_callback(results_path, (0.0,0.01), floor(Int, T/(dt*50))))
            tr_end = time()
            run_time = tr_end - tr_start
            runtime_vec[Mcnt,pcnt] = run_time
            println("Run for p=$p, M=$M finished in ", round(run_time, digits=1), " seconds.")

            # calculate and record error
            error_analysis = ErrorAnalysis(results_path, conservation_law,
            spatial_discretization, JaskowiecSukumarQuadrature(2p+3))
            err=Float64(analyze(error_analysis, last(sol.u), exact_solution, T)[1])
            err_vec[Mcnt,pcnt] = err

            println("Error is $err and DOF is $dof")
            # update accumulator
            Mcnt +=1
        end

        # update accumulator
        pcnt += 1
    end

    time_end = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    open("ConvergenceStudies/linadvec3d_"*string(opertype)*"_M"*join(M_vec, "")*"_p"*join(p_vec, "")*"_CFL"*join(replace.(string.(CFL), "." => "p"), "")*".txt", "w") do io
        println(io, "Run ended at: ", time_start)
        println(io, "Run finished at: ", time_end)
        println(io, "Operator type: ", opertype)
        println(io, "p_vec: ", p_vec)
        println(io, "M_vec: ", M_vec)
        println(io, "advection velocity: ", a)
        println(io, "Domain length L: ", L)
        println(io, "End time T: ", T)
        println(io, "CFL ", CFL)
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

function euler2d(opertype::String, p_vec, M_vec)
    time_start = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    mach_number = 0.4
    angle = 0.0
    L = 1.0
    γ=1.4
    rho_central = 0.9
    T = L/mach_number # end time
    strength = sqrt(2/(γ-1)*(1-rho_central^(γ-1)))
    C_t = 0.1
    conservation_law = EulerEquations{2}(γ)
    exact_solution = IsentropicVortex(conservation_law, θ=angle,
        Ma=mach_number, β=strength, R=1.0/10.0, x_0=(L/2,L/2));

    err_vec = zeros(length(M_vec),length(p_vec))
    dof_vec = zeros(length(M_vec),length(p_vec))
    dt_vec = zeros(length(M_vec),length(p_vec))
    setuptime_vec = zeros(length(M_vec),length(p_vec))
    runtime_vec = zeros(length(M_vec),length(p_vec))

    mesh_warp = 1.0/16

    flux = 1.0 #1.0 for upwind/dissipative
    # accumulator for p
    pcnt = 1

    # loop over p
    for p in p_vec

        # accumulator for M
        Mcnt = 1

        # loop over meshes
        for M in M_vec
            ts_start = time()
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
            else
                error("opertype should be 'NodalTPSS', 'NodalTPSSOpt', or 'NodalTPSSLGL'")
            end

            form = FluxDifferencingForm(inviscid_numerical_flux=LaxFriedrichsNumericalFlux(flux))

            uniform_mesh = uniform_periodic_mesh(reference_approximation, ((0.0,L),(0.0,L)), (M,M))

            mesh = warp_mesh(uniform_mesh, reference_approximation, ChanWarping(mesh_warp, (L,L)))

            spatial_discretization = SpatialDiscretization(mesh, reference_approximation)

            results_path = save_project(conservation_law,
                spatial_discretization, exact_solution, form, (0.0, T),
                "results/euler_vortex_2d_3", overwrite=true, clear=true)

            ode = semidiscretize(conservation_law, spatial_discretization,
                exact_solution, form, (0.0, T), parallelism=Serial());

            dof = size(ode.u0)[1]*size(ode.u0)[3]
            dof_vec[Mcnt,pcnt]= dof

            dt = C_t * (L/M) / (mach_number*p^2)
            dt_vec[Mcnt,pcnt]=dt

            ts_end = time()
            setup_time = ts_end - ts_start
            setuptime_vec[Mcnt,pcnt] = setup_time
            println("Setup for p=$p, M=$M finished in ", round(setup_time, digits=1), " seconds.")

            tr_start = time()
            sol = solve(ode, DP8(), adaptive=false, dt=dt, save_everystep=false,
                callback=save_callback(results_path, (0.0,0.0), 5))
            tr_end = time()
            run_time = tr_end - tr_start
            runtime_vec[Mcnt,pcnt] = run_time
            println("Run for p=$p, M=$M finished in ", round(run_time, digits=1), " seconds.")

            error_analysis = ErrorAnalysis(results_path, conservation_law,
                spatial_discretization, DefaultQuadrature(35))

            err=Float64(analyze(error_analysis, last(sol.u), exact_solution, T)[1])
            err_vec[Mcnt,pcnt] = err

            println("Error is $err and DOF is $dof")

            # update accumulator
            Mcnt +=1
        end

        # update accumulator
        pcnt += 1
    end

    time_end = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    open("ConvergenceStudies/euler2d_"*string(opertype)*"_M"*join(M_vec, "")*"_p"*join(p_vec, "")*"_C_t"*join(replace.(string.(C_t), "." => "p"), "")*"_rhomin"*join(replace.(string.(rho_central), "." => "p"), "")*".txt", "w") do io
        println(io, "Run ended at: ", time_start)
        println(io, "Run finished at: ", time_end)
        println(io, "Operator type: ", opertype)
        println(io, "p_vec: ", p_vec)
        println(io, "M_vec: ", M_vec)
        println(io, "Mac number: ", mach_number)
        println(io, "angle: ", angle)
        println(io, "Domain length L: ", L)
        println(io, "End time T: ", T)
        println(io, "Vortex strength: ", strength)
        println(io, "C_t ", C_t)
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

        println(io, "Setup times (seconds) for each (M,p) pair (rows=M_vec, cols=p_vec):")
        writedlm(io, setuptime_vec)
        println(io, "")

        println(io, "Run times (seconds) for each (M,p) pair (rows=M_vec, cols=p_vec):")
        writedlm(io, runtime_vec)

    end
    return err_vec, dof_vec

end

advec2d("ModalDiagE",[2,3,4,5],[4,8,12])
# advec2d("NodalTPSSLGL",[2,3,4,5,6,7],[4,8,12,16,20])
# advec2d("NodalTPSSMinimal",[2,3,4,5,6],[2,3,4,5,6])
#advec2d("ModalOmega",[2,3,4,5],[4,8,12])
#advec2d("NodalDiagE",[2,3,4,5,6,7],[4,8,12,16,20])

# advec3d("ModalTensor",[2,3,4,5,6],[2,4,6,8])
# advec3d("NodalTPSSLGL",[2,3,4,5,6],[2,4,6,8])
# advec3d("NodalTPSSMinimal",[2,3,4,5,6],[2,3,4,5,6]) 
#advec3d("ModalOmega",[2,3,4,5,6],[2,4,6,8])
#advec3d("NodalDiagE",[2,3,4,5,6],[2,4,6,8])

#euler2d("ModalTensor",[2,3,4,5],[24,28,32])

