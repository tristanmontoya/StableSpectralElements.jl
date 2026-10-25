using Statistics
using Printf
using StableSpectralElements, OrdinaryDiffEq, .Threads
using Plots, Plots.PlotMeasures, TimerOutputs
using DelimitedFiles, Dates

"""
Run the solver for ONLY 10 timesteps and return:

(success, err)

success = false if:
- solve throws an exception
- error becomes NaN/Inf
- error exceeds stability threshold
"""
function run_solver_stability(opertype::String, dt, p, M, parameters; nsteps, stab_tol)
    try
        (mach_number, angle, rho, flux, warp) = (parameters[:mach_number], parameters[:angle], parameters[:rho], parameters[:flux], parameters[:warp])

        L = 1.0
        γ=1.4

        #strength = sqrt(2/(γ-1)*(1-rho^(γ-1)))
        conservation_law = EulerEquations{2}(γ)
        #exact_solution = IsentropicVortex(conservation_law, θ=angle,
        #    Ma=mach_number, β=strength, R=1.0/10.0, x_0=(L/2,L/2));
        
        exact_solution = EulerGaussianPulse(conservation_law)

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
            error("Unsupported operator type")
        end

        form = FluxDifferencingForm(inviscid_numerical_flux=LaxFriedrichsNumericalFlux(flux))

        uniform_mesh = uniform_periodic_mesh(reference_approximation, ((0.0,L),(0.0,L)), (M,M))

        mesh = warp_mesh(uniform_mesh, reference_approximation, ChanWarping(warp, (L,L)))

        spatial_discretization = SpatialDiscretization(mesh, reference_approximation)

        T = nsteps * dt
        
        results_path = save_project(conservation_law,
                spatial_discretization, exact_solution, form, (0.0, T),
                "results/euler_vortex_2d", overwrite=true, clear=true)

        if opertype == "ModalTensor" || opertype == "NodalTensor" || opertype == "NodalTPSSLGL"
            ode_problem = semidiscretize(conservation_law,
                spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), SparseMatrixAlgorithm(), parallelism=Serial())
        else
            ode_problem = semidiscretize(conservation_law,
                spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), DefaultOperatorAlgorithm(), parallelism=Serial())
        end

        println("Running: p=$p, M=$M, dt=$dt")

        sol = solve(ode_problem, RK4(), adaptive = false, dt = dt, save_everystep = false)

        u0 = sol.u[1]
        uT = sol.u[end]

        energy_analysis = EnergyConservationAnalysis(conservation_law, spatial_discretization)

        E0 = evaluate_conservation(energy_analysis, u0)
        ET = evaluate_conservation(energy_analysis, uT)

        difference = ET - E0
        
        # convert diff to type Float64
        difference = Float64(difference[1])

        if difference > stab_tol || isnan(difference) || isinf(difference)
            println("Unstable: energy change = ", difference, ", decreasing upper bound")
            return (false, difference)
        end
        println("Stable: energy change = ", difference, ", increasing lower bound")
        return (true, difference)
    catch e
        if isa(e, DivideError) || isa(e, DomainError) 

            println("Unstable: energy change = ", Inf, ", decreasing upper bound")
            return (false, Inf)
        else
            rethrow(e) # Catch unexpected bugs (syntax, etc)
        end
    end        
end

function run_solver_stability_3d(opertype::String, dt, p, M, parameters; nsteps, stab_tol)
    try
        (mach_number, angle, rho, flux, warp) = (parameters[:mach_number], parameters[:angle], parameters[:rho], parameters[:flux], parameters[:warp])

        L = 2.0
        Ω = ((0.0,L),(0.0,L),(0.0,L))
        conservation_law = EulerEquations{3}(1.4)
        exact_solution = EulerPeriodicTest(conservation_law);

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

        elseif opertype == "NodalTensor"
            reference_approximation = ReferenceApproximation(
                NodalTensor(p), Tet(), mapping_degree=p)

        elseif opertype == "NodalOmega"
            reference_approximation = ReferenceApproximation(
                NodalMulti(p), Tet(), mapping_degree=p)

        elseif opertype == "ModalDiagE"
            reference_approximation = ReferenceApproximation(
                ModalMultiDiagE(p), Tet(), mapping_degree=p)

        elseif opertype == "NodalDiagE"
            reference_approximation = ReferenceApproximation(
                NodalMultiDiagE(p), Tet(), mapping_degree=p)

        elseif opertype == "MFNodalTPSSLGL"
            reference_approximation = ReferenceApproximation(
                MatrixFreeTPSSLGL(p), Tet(), mapping_degree=p)

        else
            error("Unsupported operator type")
        end

        form = FluxDifferencingForm(inviscid_numerical_flux=LaxFriedrichsNumericalFlux(flux))

        uniform_mesh = uniform_periodic_mesh(reference_approximation, Ω, (M,M,M))
            
        mesh = warp_mesh(uniform_mesh, reference_approximation, ChanWarping(warp, (L,L,L)))
        
        spatial_discretization = SpatialDiscretization(mesh, reference_approximation,
            ChanWilcoxMetrics())

        T = nsteps * dt
        
        results_path = save_project(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T),
            "results/euler_periodic_3d/", overwrite=true, clear=true);

        if opertype == "ModalTensor" || opertype == "NodalTensor" || opertype == "NodalTPSSLGL"
            ode_problem = semidiscretize(conservation_law,
                spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), SparseMatrixAlgorithm(), parallelism=Serial())
        else
            ode_problem = semidiscretize(conservation_law,
                spatial_discretization, exact_solution, form, (0.0, T), ReferenceOperator(), DefaultOperatorAlgorithm(), parallelism=Serial())
        end

        println("Running: p=$p, M=$M, dt=$dt")

        sol = solve(ode_problem, DP8(), adaptive = false, dt = dt, save_everystep = false)

        u0 = sol.u[1]
        uT = sol.u[end]

        energy_analysis = EnergyConservationAnalysis(conservation_law, spatial_discretization)

        E0 = evaluate_conservation(energy_analysis, u0)
        ET = evaluate_conservation(energy_analysis, uT)

        difference = ET - E0
        
        # convert diff to type Float64
        difference = Float64(difference[1])

        if difference > stab_tol || isnan(difference) || isinf(difference)
            println("Unstable: energy change = ", difference, ", decreasing upper bound")
            return (false, difference)
        end
        println("Stable: energy change = ", difference, ", increasing lower bound")
        return (true, difference)
    catch e
        if isa(e, DivideError) || isa(e, DomainError) 

            println("Unstable: energy change = ", Inf, ", decreasing upper bound")
            return (false, Inf)
        else
            rethrow(e) # Catch unexpected bugs (syntax, etc)
        end
    end        
end


"""
Find maximum stable dt for fixed (p,M).

Procedure:
1. Compute reference error using tiny dt
2. Increase dt using golden section search
3. Stable if:
      err <= growth_tol * reference_error
   and solve succeeds
"""
function find_max_stable_dt(opertype::String, p, M, d, parameters; nsteps, stab_tol, stop_tol)
    dt_min = 1e-6
    dt_max = 1e-1
    iterations = 30
    println("="^80)
    println("Finding maximum stable dt")
    println("p = $p | M = $M")
    println("="^80)

    phi = (sqrt(5) - 1) / 2
    a = dt_min
    b = dt_max

    best_dt = dt_min
    previous_dt = dt_min

    for iter in 1:iterations

        #dt = a + phi * (b - a)
        dt = 0.5*(a+b)
        println("\nIteration $iter")
        println("Testing dt = ", dt)

        # Exit if dt changes by less stab_tol
        if abs(dt - previous_dt) / previous_dt < stop_tol
            println("dt change below ", string(stop_tol*100), "% exiting search")
            break
        end

        previous_dt = dt

        if d == 2
            (stable, diff) = run_solver_stability(opertype, dt, p, M, parameters; nsteps, stab_tol)
        else 
            (stable, diff) = run_solver_stability_3d(opertype, dt, p, M, parameters; nsteps, stab_tol)
        end

        if stable
            best_dt = dt
            a = dt
        else
            b = dt
        end
    end

    println("\nMaximum stable dt ≈ ", best_dt)

    return best_dt
end


# --- TEST SUITE ---
function run_test_suite(opertype::String, p_vec, M_vec,d)
    mach_number = 0.4
    angle = 0.0
    rho = 0.75
    L = 1.0  # domain length
    nsteps =100
    stab_tol = 1e-3
    stop_tol = 0.01
    # set mesh warping
    warp = 1/32

    # set upwind (1.0) or symmetric flux (0.0)
    flux = 1.0

    parameters = Dict(
        :mach_number => mach_number,
        :angle => angle,
        :rho => rho,
        :flux => flux,
        :warp => warp
    )

    dt_matrix = zeros(length(M_vec), length(p_vec))
    
    pcnt = 1
    for p in p_vec
        Mcnt = 1
        for M in M_vec
            
            println("="^100, "\n Starting tests for p=$p, M=$M \n", "="^100)
            max_dt = find_max_stable_dt(opertype, p, M, d, parameters; nsteps, stab_tol, stop_tol)

            dt_matrix[Mcnt, pcnt] = max_dt
            println("Found max dt for p=$p, M=$M: dt=$max_dt")

            Mcnt += 1
        end
        pcnt += 1
    end

    # Write results to output.txt
    Date = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    file_name = "ConvergenceStudies/Summer2026/Rerun/Maxdt_euler"*string(d)*"d_"*string(opertype)*"_p"*join(p_vec, "")*"_M"*join(M_vec, "")*"warp1o32.txt"
    open(file_name, "w") do io
        println(io, "Run ended at: ", Date)
        println(io, "Operator type: ", opertype)
        println(io, "p_vec: ", p_vec)
        println(io, "M_vec: ", M_vec)
        #println(io, "Mach number: ", mach_number)
        #println(io, "angle: ", angle)
        println(io, "Euler Gaussian Pulse test case")
        println(io, "Domain length L: ", L)
        println(io, "Number of steps for stability test: ", nsteps)
        println(io, "Upwind (1.0)/symmetric flux (0.0): ", flux)
        println(io, "Mesh warping parameter: ", warp)
        println(io, "Number of timesteps: ", nsteps)
        println(io, "Stability tolerance: ", stab_tol)
        println(io, "Stop tolerance: ", stop_tol)
    
        # --- Table 1: Max Timesteps (dt) ---
        println(io, "MAXIMUM ALLOWABLE TIMESTEP (dt)")
        writedlm(io, dt_matrix)
        
    end

    println("Results successfully saved to ", file_name)
end


run_test_suite("NodalDiagE", [1,2,3,4], [8,12,16,24],2)
run_test_suite("NodalDiagE", [5,6,7,8], [8,10,12,16],2)
run_test_suite("ModalDiagE", [1,2,3,4], [8,12,16,24],2)
run_test_suite("ModalDiagE", [5,6,7,8], [8,10,12,16],2)
run_test_suite("NodalOmega", [1,2,3,4], [8,12,16,24],2)
run_test_suite("NodalOmega", [5,6,7,8], [8,10,12,16],2)
run_test_suite("ModalOmega", [1,2,3,4], [8,12,16,24],2)
run_test_suite("ModalOmega", [5,6,7,8], [8,10,12,16],2)
run_test_suite("NodalTensor", [1,2,3,4], [8,12,16,24],2)
run_test_suite("NodalTensor", [5,6,7,8], [8,10,12,16],2)
run_test_suite("ModalTensor", [1,2,3,4], [8,12,16,24],2)
run_test_suite("ModalTensor", [5,6,7,8], [8,10,12,16],2)
run_test_suite("NodalTPSSLGL", [1,2,3,4], [4,6,8,12],2)
run_test_suite("NodalTPSSLGL", [5,6,7,8], [4,5,6,8],2)
run_test_suite("NodalTPSSOptimal", [2,3,4,5,6], [1,2,3,4],2) 

run_test_suite("ModalTensor",[1,2,3,4,5],[2,4,6,8],3)
run_test_suite("NodalTensor",[1,2,3,4,5],[2,4,6,8],3)
run_test_suite("ModalOmega",[1,2,3,4,5],[2,4,6,8],3)
run_test_suite("NodalOmega",[1,2,3,4,5],[2,4,6,8],3)
run_test_suite("ModalDiagE",[1,2,3,4,5],[2,4,6,8],3)
run_test_suite("NodalDiagE",[1,2,3,4,5],[2,4,6,8],3)
run_test_suite("NodalTPSSLGL",[1,2,3,4,5],[1,2,3,4],3)
