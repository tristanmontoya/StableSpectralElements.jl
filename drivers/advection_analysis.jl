push!(LOAD_PATH,"../")

using OrdinaryDiffEq
using LinearAlgebra
using TimerOutputs
using LaTeXStrings
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
            nargs = '*'
            arg_type = Int
            default = collect(2:5)
        "-l", "--mapping_degree"
            help = "degree of mapping"
            nargs = '*'
            arg_type = Int
            default = collect(2:5)
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
        "--path"
            help = "results path"
            arg_type = String
            default = "../results/advection_test/"
        "-M"
            help = "number of intervals"
            arg_type = Int
            default = 2
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
        "--n_ev", "-n"
            help = "Number of eigenvalues"
            arg_type = Int
            default = 1
        "--load_from_file"
            help = "load_from_file"
            action = :store_true
        "--overwrite"
            help = "overwrite"
            action = :store_true
    end

    return parse_args(s)
end

struct AdvectionAnalysisDriver{d}
    p::Vector{Int}
    l::Vector{Int}
    scheme::Vector{<:AbstractApproximationType}
    element_type::AbstractElemShape
    mapping_form::AbstractMappingForm
    path::String
    M::Int
    L::Float64
    a::NTuple{d,Float64}
    mesh_perturb::Float64
    n_ev::Int
    load_from_file::Bool
    overwrite::Bool
end

function AdvectionAnalysisDriver(parsed_args::Dict)

    p = parsed_args["poly_degree"]
    l = parsed_args["mapping_degree"]
    element_type = eval(Symbol(parsed_args["element_type"]))()
    scheme = [eval(Symbol(parsed_args["scheme"]))(p[i]) for i in eachindex(p)] 
    mapping_form = eval(Symbol(parsed_args["mapping_form"]))()

    path = string(parsed_args["path"], parsed_args["scheme"], "_",
    parsed_args["element_type"], "_", parsed_args["mapping_form"], "/")

    M = parsed_args["M"]
    L = parsed_args["L"]
    a = parsed_args["a"]
    θ = parsed_args["theta"]
    ϕ = parsed_args["phi"]
    
    mesh_perturb = parsed_args["mesh_perturb"]
    n_ev = parsed_args["n_ev"]
    load_from_file = parsed_args["load_from_file"]
    overwrite = parsed_args["overwrite"]

    a_vec = (a*cos(θ)*sin(ϕ), a*sin(θ)*sin(ϕ), a*cos(ϕ))

    return AdvectionAnalysisDriver(p,l,scheme,element_type,
        mapping_form,path,M,L, Tuple(a_vec[m] for m in 1:dim(element_type)),
        mesh_perturb, n_ev, load_from_file, overwrite)
end

function run_driver(driver::AdvectionAnalysisDriver{d}) where {d}

    @unpack p,l,scheme,element_type,mapping_form,path,M,L,a,mesh_perturb, n_ev, load_from_file, overwrite = driver

    if (!load_from_file || !isdir(path))
        path = new_path(path,overwrite,overwrite)
    end

    if d == 3 initial_data = InitialDataCosine(1.0,Tuple(2*π/L for m in 1:d))
    else initial_data = InitialDataSine(1.0,Tuple(2*π/L for m in 1:d)) end

    conservation_law = LinearAdvectionEquation(a)
    spectral_radius_central = Vector{Float64}(undef,length(p))
    spectral_radius_upwind = Vector{Float64}(undef,length(p))

    for i in eachindex(p)

        reference_approximation =ReferenceApproximation(
            scheme[i], element_type, mapping_degree=l[i])
        
        mesh = warp_mesh(uniform_periodic_mesh(
            reference_approximation.reference_element, 
                Tuple((0.0,L) for m in 1:d), Tuple(M for m in 1:d)), 
                reference_approximation.reference_element, mesh_perturb)

        spatial_discretization = SpatialDiscretization(mesh, 
            reference_approximation)

        solver_central = Solver(conservation_law, spatial_discretization,
            WeakConservationForm(mapping_form=mapping_form,
                inviscid_numerical_flux=LaxFriedrichsNumericalFlux(0.0)))

        solver_upwind = Solver(conservation_law, spatial_discretization,
            WeakConservationForm(mapping_form=mapping_form,
                inviscid_numerical_flux=LaxFriedrichsNumericalFlux(1.0)))

        if n_ev < 1
            (N_p, N_c, N_e) = get_dof(spatial_discretization, conservation_law)
            rank = N_p*N_c*N_e - 2
        else rank = n_ev end
        
        linear_analysis_central = LinearAnalysis(path,
            conservation_law, spatial_discretization, 
            LinearResidual(solver_central), r=rank,
            use_data=false, name=string("p", p[i], "_central"))
        
        linear_results_central = analyze(linear_analysis_central)
        spectral_radius_central[i] =  maximum(abs.(linear_results_central.λ))
        open(string(path,"central.txt"), "a") do io
            println(io, "p = ", string(p[i]), ", specr = ",spectral_radius_central[i])
        end

        linear_analysis_upwind = LinearAnalysis(path,
            conservation_law, spatial_discretization, 
            LinearResidual(solver_upwind), r=rank,
            use_data=false, name=string("p", p[i], "_upwind"))

        linear_results_upwind = analyze(linear_analysis_upwind)
        spectral_radius_upwind[i] =  maximum(abs.(linear_results_upwind.λ))

        open(string(path,"upwind.txt"), "a") do io
            println(io, "p = ", string(p[i]), ", specr = ",
            spectral_radius_upwind[i])
        end

        plt = plot([linear_results_central.λ, linear_results_upwind.λ], 
            windowsize=(400,400))
        savefig(plt, string(path,"spectrum_p", p[i], ".pdf"))
    end

    spectral_radius_plot = plot(p,spectral_radius_central, windowsize=(400,400),
        fontfamily="Computer Modern", legend=:topleft, xlabel=latexstring("p"),
        ylabel=LaTeXString("Spectral Radius"), label=LaTeXString("Central"))
    plot!(spectral_radius_plot, p, spectral_radius_upwind, 
        label=LaTeXString("Upwind"))
    savefig(spectral_radius_plot, string(path, "spectral_radius.pdf"))
end

function main(args) run_driver(AdvectionAnalysisDriver(parse_commandline())) end

main(ARGS)