push!(LOAD_PATH,"../")
ENV["MPLBACKEND"]="agg"

using OrdinaryDiffEq: solve, DP8, RK4, CarpenterKennedy2N54, OrdinaryDiffEqAlgorithm
using LinearAlgebra
using TimerOutputs
using UnPack
using ArgParse
using Dates
using Suppressor
using CLOUD

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "-p"
            help = "polynomial degree"
            arg_type = Int
            default = 4
        "-r"
            help = "degree of mapping"
            arg_type = Int
            default = 1
        "-C"
            help = "Courant number"
            arg_type = Float64
            default = 5.0e-4
        "-n"
            help = "number of writes to file"
            arg_type = Int
            default = 10
        "--scheme", "-s"
            help = "spatial_discretization scheme"
            arg_type = String
            default = "DGMulti"
        "--form", "-f"
            help = "residual form"
            arg_type = String
            default = "WeakConservationForm"
        "--integrator", "-i"
            help = "time integrator"
            arg_type = String
            default = "RK4"
        "--path"
            help = "results path"
            arg_type = String
            default = "../results/"
        "-M"
            help = "number of intervals"
            arg_type = Int
            default = 2
        "--lambda", "-l"
            help = "upwinding parameter (0 for central, 1 for upwind)"
            arg_type = Float64
            default = 1.0
        "-L"
            help = "domain length"
            arg_type = Float64
            default = 1.0
        "-a"
            help = "advection wave speed"
            arg_type = Float64
            default = sqrt(2)
        "--theta", "-t"
            help = "advection wave angle"
            arg_type = Float64
            default = π/4.0
        "--mesh_perturb", "-m"
            help = "mesh perturbation"
            arg_type = Float64
            default = 0.15
        "--n_grids", "-g"
            help = "number of grids"
            arg_type = Int
            default = 1
        "--n_periods"
            help = "number of periods"
            arg_type = Float64
            default = 1.0
        "--timer"
            help = "whether to run timer"
            arg_type = Bool
            default = false
    end

    return parse_args(s)
end

struct AdvectionDriver{d}
    p::Int
    r::Int
    C::Float64
    n_s::Int
    scheme::AbstractApproximationType
    elem_type::AbstractElemShape
    form::AbstractResidualForm
    integrator::OrdinaryDiffEqAlgorithm
    path::String

    M0::Int
    λ::Float64
    L::Float64
    a::NTuple{d,Float64}
    T::Float64
    mesh_perturb::Float64
    n_grids::Int
    timer::Bool
end

function advection_driver_2d(parsed_args::Dict)

    p = parsed_args["p"]
    r = parsed_args["r"]
    C = parsed_args["C"]
    n_s = parsed_args["n"]
    
    if parsed_args["scheme"] == "DGMulti"
        scheme = DGMulti(p)
        elem_type = Tri()
    elseif parsed_args["scheme"] == "DGSEM"
        scheme = DGSEM(p)
        elem_type = Quad()
    elseif parsed_args["scheme"] == "CollapsedSEM"
        scheme = CollapsedSEM(p)
        elem_type = Tri()
    elseif parsed_args["scheme"] == "CollapsedModal"
        scheme = CollapsedModal(p)
        elem_type = Tri()
    else
        error("Invalid discretization scheme")
    end

    if parsed_args["form"] == "WeakConservationForm"
        form = WeakConservationForm()
    elseif parsed_args["form"] == "StrongConservationForm"
        form = StrongConservationForm()
    elseif parsed_args["form"] == "SplitConservationForm"
        form = SplitConservationForm()
    else
        error("Invalid discretization form")
    end

    if parsed_args["integrator"] == "RK4"
        integrator = RK4()
    elseif parsed_args["integrator"] == "CarpenterKennedy2N54"
        integrator = CarpenterKennedy2N54()
    elseif parsed_args["integrator"] == "DP8"
        integrator = DP8()
    else
        error("Unsupported time integrator")
    end

    path = parsed_args["path"]
    M0 = parsed_args["M"]
    λ = parsed_args["lambda"]
    L = parsed_args["L"]
    a = parsed_args["a"]
    θ = parsed_args["theta"]
    T = parsed_args["n_periods"]/(a*max(abs(cos(θ)),abs(cos(θ))))
    mesh_perturb = parsed_args["mesh_perturb"]
    n_grids = parsed_args["n_grids"]
    timer = parsed_args["timer"]

    return AdvectionDriver(p,r,C,n_s,scheme,elem_type,
        form, integrator, path,M0,λ,L,(a*cos(θ), a*sin(θ)), T,
        mesh_perturb, n_grids, timer)
end

function main(args)
    parsed_args = parse_commandline()
    @unpack p,r,C,n_s,scheme,elem_type,form,integrator,path,M0,λ,L,a,T,mesh_perturb, n_grids, timer =  advection_driver_2d(parsed_args)

    date_time = Dates.format(now(), "yyyymmdd_HHMMSS")
    path = new_path(string(path, "advection_", parsed_args["scheme"], "_p", string(p), "M", string(Int(M0)), "l", string(Int(λ)), "_", date_time, "/"))

    initial_data = InitialDataSine(1.0,(2*π/L, 2*π/L))
    conservation_law = linear_advection_equation(a,λ=λ)

    for n in 1:n_grids

        M = M0^n

        reference_approximation =ReferenceApproximation(
            scheme, elem_type, mapping_degree=r, N_plot=ceil(Int,20/M));
        
        results_path = new_path(string(path, "grid_", n, "/"))
        mesh = warp_mesh(uniform_periodic_mesh(
            reference_approximation.reference_element, ((0.0,L),(0.0,L)), 
            (M,M)), reference_approximation.reference_element, mesh_perturb)

        spatial_discretization = SpatialDiscretization(mesh, 
            reference_approximation);
        solver = Solver(conservation_law, spatial_discretization, form, Lazy())

        save_project(conservation_law,
            spatial_discretization, initial_data, form, 
            (0.0, T), Lazy(), results_path, overwrite=true, clear=true)

        open(string(results_path,"screen.txt"), "a") do io
            println(io, "Number of Julia threads: ", Threads.nthreads())
            println(io, "Number of BLAS threads: ", BLAS.get_num_threads(),"\n")
            println(io, "Results Path: ", "\"", results_path, "\"\n")
            println(io, "Parameters: ", parsed_args, "\n")
        end

        dt = C*(L/M)/norm(a)
        ode_problem = semidiscretize(solver, initialize(initial_data, 
            conservation_law, spatial_discretization), (0.0, T))

        save_solution(ode_problem.u0, 0.0, results_path, 0)
        for t in 1:Threads.nthreads()
            reset_timer!(get_timer(string("thread_timer_",t)))
        end

        reset_timer!()
        sol = solve(ode_problem, integrator, adaptive=false,
            dt=dt, save_everystep=false, callback=save_callback(
                results_path, ceil(Int, T/(dt*n_s))))
        save_solution(last(sol.u), last(sol.t), results_path, "final")

        open(string(results_path,"screen.txt"), "a") do io
            println(io, "Solver finished!\n")
            
        if timer
            to = merge(Tuple(get_timer(string("thread_timer_",t)) 
                for t in 1:Threads.nthreads())...)
            println(io, @capture_out print_timer(to), "\n")
        end

        error_analysis = ErrorAnalysis(results_path, conservation_law, 
            spatial_discretization)
        conservation_analysis = PrimaryConservationAnalysis(results_path, 
            conservation_law, spatial_discretization)
        energy_analysis = EnergyConservationAnalysis(results_path, 
            conservation_law, spatial_discretization)
        println(io,"L2 error:\n", 
            analyze(error_analysis, last(sol.u), initial_data))
        println(io,"Conservation (initial/final/diff):\n", 
            analyze(conservation_analysis)...)
        println(io,"Energy (initial/final/diff):\n",
            analyze(energy_analysis)...)
        end
    end
end

main(ARGS)
