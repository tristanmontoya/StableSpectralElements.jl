push!(LOAD_PATH,"../")

using OrdinaryDiffEq
using LinearAlgebra
using TimerOutputs
using UnPack
using ArgParse
using Dates
using Suppressor
using Plots; gr()
using CLOUD

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "-p", "--poly_degree"
            help = "polynomial degree"
            arg_type = Int
            default = 4
        "-l", "--mapping_degree"
            help = "degree of mapping"
            arg_type = Int
            default = 1
        "--CFL", "-C"
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
        "--mass_solver"
            help = "mass matrix solver"
            arg_type = String
            default = "CholeskySolver"
        "--path"
            help = "results path"
            arg_type = String
            default = "../results/advection_test/"
        "-M"
            help = "number of intervals"
            arg_type = Int
            default = 2
        "--lambda"
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
        "--overwrite", "-o"
            help = "overwrite"
            action = :store_true
    end

    return parse_args(s)
end

struct AdvectionDriver{d}
    p::Int
    l::Int
    CFL::Float64
    n_s::Int
    scheme::AbstractApproximationType
    element_type::AbstractElemShape
    mapping_form::AbstractMappingForm
    ode_algorithm::OrdinaryDiffEqAlgorithm
    mass_solver::AbstractMassMatrixSolver
    path::String
    M0::Int
    λ::Float64
    L::Float64
    a::NTuple{d,Float64}
    T::Float64
    mesh_perturb::Float64
    n_grids::Int
    load_from_file::Bool
    overwrite::Bool
end

function AdvectionDriver(parsed_args::Dict)

    p = parsed_args["poly_degree"]
    l = parsed_args["mapping_degree"]
    CFL = parsed_args["CFL"]
    n_s = parsed_args["n"]
    element_type = eval(Symbol(parsed_args["element_type"]))()
    scheme = eval(Symbol(parsed_args["scheme"]))(p)
    mapping_form = eval(Symbol(parsed_args["mapping_form"]))()
    ode_algorithm = eval(Symbol(parsed_args["ode_algorithm"]))()
    mass_solver = eval(Symbol(parsed_args["mass_solver"]))()

    if Int(round(parsed_args["lambda"])) == 0 path = string(parsed_args["path"],   
        parsed_args["scheme"], "_", parsed_args["element_type"], "_", parsed_args["mapping_form"], "_p", string(parsed_args["poly_degree"]),
         "/central/")
    elseif Int(round(parsed_args["lambda"])) == 1 path = string(parsed_args["path"],
        parsed_args["scheme"], "_", parsed_args["element_type"], "_", 
        parsed_args["mapping_form"], "_p", string(parsed_args["poly_degree"]),
        "/upwind/")
    else path = string(parsed_args["path"], parsed_args["scheme"], "_",
        parsed_args["element_type"], "_", parsed_args["mapping_form"], "_p",
        string(parsed_args["poly_degree"]), "/lambda", 
        replace(string(parsed_args["lambda"]), "." => "_"), "/")
    end

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
    overwrite = parsed_args["overwrite"]

    a_vec = (a*cos(θ)*sin(ϕ), a*sin(θ)*sin(ϕ), a*cos(ϕ))

    return AdvectionDriver(p,l,CFL,n_s,scheme,element_type,
        mapping_form,ode_algorithm,mass_solver,path,M0,λ,L,
        Tuple(a_vec[m] for m in 1:dim(element_type)), T,
        mesh_perturb, n_grids, load_from_file, overwrite)
end

function run_driver(driver::AdvectionDriver{d}) where {d}

    @unpack p,l,CFL,n_s,scheme,element_type,mapping_form,ode_algorithm,mass_solver,path,M0,λ,L,a,T,mesh_perturb, n_grids, load_from_file, overwrite = driver

    if (!load_from_file || !isdir(path)) path = new_path(
        path,overwrite,overwrite) end
    if !isdir(string(path, "grid_1/")) n_start = 1 else
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

    if d == 3 initial_data = InitialDataCosine(1.0,Tuple(2*π/L for m in 1:d))
    else initial_data = InitialDataSine(1.0,Tuple(2*π/L for m in 1:d)) end

    conservation_law = LinearAdvectionEquation(a)
    form = SplitConservationForm(mapping_form=mapping_form, 
            inviscid_numerical_flux=LaxFriedrichsNumericalFlux(λ))
    eoc = -1.0

    for n in n_start:n_grids

        M = M0*2^(n-1)

        reference_approximation =ReferenceApproximation(
            scheme, element_type, mapping_degree=l)
        
        mesh = warp_mesh(uniform_periodic_mesh(
            reference_approximation.reference_element,
                Tuple((0.0,L) for m in 1:d), Tuple(M for m in 1:d),
                collapsed_orientation=(isa(element_type,Tet) && isa(scheme,Union{ModalTensor,NodalTensor}))),
                reference_approximation.reference_element,
                ChanWarping(mesh_perturb,Tuple(L for m in 1:d)))

        spatial_discretization = SpatialDiscretization(mesh, 
            reference_approximation, project_jacobian=isa(mass_solver,
            WeightAdjustedSolver))

        solver = Solver(conservation_law, spatial_discretization, 
            form, PhysicalOperator(), DefaultOperatorAlgorithm(),
            mass_solver)
        
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

        h = L/(reference_approximation.N_p * spatial_discretization.N_e)^(1/d)
        dt = CFL*h/(norm(a))   
        reset_timer!()
        sol = solve(ode_problem, ode_algorithm, adaptive=false,
            dt=dt, save_everystep=false, callback=save_callback(
                results_path, (t0,T), ceil(Int, T/(dt*n_s)), restart_step))

        if sol.retcode != :Success 
            open(string(results_path,"screen.txt"), "a") do io
                println(io, "Solver failed! Retcode: ", string(sol.retcode))
            end
            continue
        end

        error_analysis = ErrorAnalysis(results_path, conservation_law, 
            spatial_discretization)

        open(string(results_path,"screen.txt"), "a") do io
            println(io, "Solver successfully finished!\n")
            println(io, @capture_out print_timer(), "\n")
            println(io,"L2 error:\n", analyze(error_analysis, 
                last(sol.u), initial_data))
        end

        if n > 1
            refinement_results = analyze(RefinementAnalysis(initial_data, path,
            "./", "advection test"), n, max_derivs=true,
            use_weight_adjusted_mass_matrix=isa(mass_solver,
                WeightAdjustedSolver))
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