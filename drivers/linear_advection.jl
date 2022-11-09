# Driver script - grid refinement studies for the linear advection equation
push!(LOAD_PATH,"../")

using OrdinaryDiffEq
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
        "--beta", "-b"
            help = "Time step factor"
            arg_type = Float64
            default = 2.5e-3
        "-n"
            help = "number of writes to file"
            arg_type = Int
            default = 10
        "--element_type", "-e"
            help = "element type"
            arg_type = String
            default = "Tri"
        "--scheme", "-s"
            help = "spatial_discretization scheme"
            arg_type = String
            default = "ModalMulti"
        "--mapping_form", "-f"
            help = "mapping form"
            arg_type = String
            default = "StandardMapping"
        "--ode_algorithm", "-i"
            help = "ode algorithm"
            arg_type = String
            default = "CarpenterKennedy2N54"
        "--path"
            help = "results path"
            arg_type = String
            default = "../results/advection_test/"
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
            default = sqrt(2.0)
        "--theta", "-t"
            help = "advection azimuthal angle"
            arg_type = Float64
            default = π/4.0
        "--phi"
            help = "advection polar angle"
            arg_type = Float64
            default = π/2.0
        "--mesh_perturb", "-m"
            help = "mesh perturbation"
            arg_type = Float64
            default = 0.15
        "--n_grids", "-g"
            help = "number of grids"
            arg_type = Int
            default = 1
        "--final_time"
            help = "final time"
            arg_type = Float64
            default = 1.0
        "--load_from_file"
            help = "load_from_file"
            action = :store_true
    end

    return parse_args(s)
end

struct AdvectionDriver{d}
    p::Int
    r::Int
    β::Float64
    n_s::Int
    scheme::AbstractApproximationType
    element_type::AbstractElemShape
    mapping_form::AbstractMappingForm
    ode_algorithm::OrdinaryDiffEqAlgorithm
    path::String
    M0::Int
    λ::Float64
    L::Float64
    a::NTuple{d,Float64}
    T::Float64
    mesh_perturb::Float64
    n_grids::Int
    load_from_file::Bool
end

function AdvectionDriver(parsed_args::Dict)

    p = parsed_args["p"]
    r = parsed_args["r"]
    β = parsed_args["beta"]
    n_s = parsed_args["n"]
    element_type = eval(Symbol(parsed_args["element_type"]))()
    scheme = eval(Symbol(parsed_args["scheme"]))(p)
    mapping_form = eval(Symbol(parsed_args["mapping_form"]))()
    ode_algorithm = eval(Symbol(parsed_args["ode_algorithm"]))()
    path = parsed_args["path"]
    M0 = parsed_args["M"]
    λ = parsed_args["lambda"]
    L = parsed_args["L"]
    a = parsed_args["a"]
    θ = parsed_args["theta"]
    ϕ = parsed_args["phi"]
    T = parsed_args["final_time"]
    mesh_perturb = parsed_args["mesh_perturb"]
    n_grids = parsed_args["n_grids"]
    load_from_file = parsed_args["load_from_file"]

    a_vec = (a*cos(θ)*sin(ϕ), a*sin(θ)*sin(ϕ), a*cos(ϕ))

    return AdvectionDriver(p,r,β,n_s,scheme,element_type,
        mapping_form,ode_algorithm,path,M0,λ,L,
        Tuple(a_vec[m] for m in 1:dim(element_type)), T,
        mesh_perturb, n_grids, load_from_file)
end

function run_driver(driver::AdvectionDriver{d}) where {d}

    @unpack p,r,β,n_s,scheme,element_type,mapping_form,ode_algorithm,path,M0,λ,L,a,T,mesh_perturb, n_grids, load_from_file = driver

    if !load_from_file || !isdir(path)
        path = new_path(path)
    end
    if !isdir(string(path, "grid_1/"))
        n_start = 1
    else
        for i in 1:n_grids
            if !isdir(string(path, "grid_", i + 1, "/"))
                n_start = i
                break
            end
        end
    end
    open(string(path,"screen.txt"), "a") do io
        println(io, "Starting refinement from grid level ", n_start)
    end

    if d == 3
        initial_data = InitialDataCosine(1.0,Tuple(2*π/L for m in 1:d))
    else
        initial_data = InitialDataSine(1.0,Tuple(2*π/L for m in 1:d))
    end

    conservation_law = LinearAdvectionEquation(a)
    form = WeakConservationForm(mapping_form=mapping_form, 
            inviscid_numerical_flux=LaxFriedrichsNumericalFlux(λ))
    strategy = ReferenceOperator()
    eoc = -1.0

    for n in n_start:n_grids

        M = M0*2^(n-1)

        reference_approximation =ReferenceApproximation(
            scheme, element_type, mapping_degree=r)
        
        mesh = warp_mesh(uniform_periodic_mesh(
            reference_approximation.reference_element, 
                Tuple((0.0,L) for m in 1:d), Tuple(M for m in 1:d)), 
                reference_approximation.reference_element, mesh_perturb)

        spatial_discretization = SpatialDiscretization(mesh, 
            reference_approximation)

        solver = Solver(conservation_law, spatial_discretization,
            form, strategy)
        
        results_path = string(path, "grid_", n, "/")
        if !isdir(results_path)
            save_project(conservation_law,
                spatial_discretization, initial_data, form, 
                (0.0, T), results_path, overwrite=true, clear=true)
            open(string(results_path,"screen.txt"), "a") do io
                println(io, "Number of Julia threads: ", Threads.nthreads())
                println(io, "Number of BLAS threads: ", 
                    BLAS.get_num_threads(),"\n")
                println(io, "Results Path: ", "\"", results_path, "\"\n")
            end
        end

        time_steps = load_time_steps(results_path)
        if !isempty(time_steps)
            restart_step = last(time_steps)
            u0, t0 = load_solution(results_path, restart_step)
            open(string(results_path,"screen.txt"), "a") do io
                println(io, "\nRestarting from time step ", restart_step,
                     "  t = ", t0)
            end
        else
            restart_step = 0
            u0, t0 = initialize(initial_data, conservation_law,
                spatial_discretization), 0.0
        end
        ode_problem = semidiscretize(solver, u0, (t0, T))

        dt = β*(L/M)/(norm(a)*(2*p+1))   
        CLOUD_reset_timer!()
        sol = solve(ode_problem, ode_algorithm, adaptive=false,
            dt=dt, save_everystep=false, callback=save_callback(
                results_path, (t0,T), ceil(Int, T/(dt*n_s)), restart_step))

        open(string(results_path,"screen.txt"), "a") do io
            println(io, "Solver finished!\n")
            println(io, @capture_out CLOUD_print_timer(), "\n")
            error_analysis = ErrorAnalysis(results_path, conservation_law, 
                spatial_discretization)
            conservation_analysis = PrimaryConservationAnalysis(results_path, 
                conservation_law, spatial_discretization)
            energy_analysis = EnergyConservationAnalysis(results_path, 
                conservation_law, spatial_discretization)
            N_t = last(load_time_steps(results_path))
            println(io,"L2 error:\n", 
                analyze(error_analysis, last(sol.u), initial_data))
            println(io,"Conservation (initial/final/diff):\n", 
                analyze(conservation_analysis, 0, N_t)...)
            println(io,"Energy (initial/final/diff):\n",
                analyze(energy_analysis, 0, N_t)...)
        end
        if n > 1
            refinement_results = analyze(RefinementAnalysis(initial_data, path,
            "./", "advection test"), n, max_derivs=true)
            open(string(path,"screen.txt"), "a") do io
                println(io, tabulate_analysis(refinement_results, e=1,
                    print_latex=false))
            end
            eoc = refinement_results.eoc[end,1]
        end
    end
    return eoc
end

function main(args)
    parsed_args = parse_commandline()
    eoc = run_driver(AdvectionDriver(parsed_args))
    println("order on finest two grids: ", eoc)
end

main(ARGS)