# Driver script - analysis of grid refinement studies for the linear advection equation
push!(LOAD_PATH,"../")

using UnPack
using ArgParse
using LaTeXStrings
using Suppressor
using CLOUD
using Plots; gr()

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "p"
            help = "polynomial degrees"
            nargs = '*'
            arg_type = Int
            default = [4]
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
    end

    return parse_args(s)
end

struct AdvectionAnalysisDriver
    p::Vector{Int}
    path_prefix::Vector{String}
end

function AdvectionAnalysisDriver(parsed_args::Dict)
    p = parsed_args["p"]
    path_prefix = String[]
    for i in eachindex(p)
        push!(path_prefix, 
            string(parsed_args["path"], parsed_args["scheme"], "_",
                parsed_args["element_type"], "_", parsed_args["mapping_form"], "_p", string(p[i])))
    end
    
    return AdvectionAnalysisDriver(p, path_prefix)
end

function refinement_analysis!(refinement_analysis::Vector{RefinementAnalysis},
    refinement_results::Vector{RefinementAnalysisResults},
    sequence_path::String, label::String)

    (conservation_law, spatial_discretization, 
        initial_data, form, tspan) = load_project(
        string(sequence_path,"grid_1/"))
    push!(refinement_analysis, RefinementAnalysis(initial_data, 
        sequence_path, sequence_path, label))
    push!(refinement_results, analyze(last(refinement_analysis),
        max_derivs=true))

end

function run_driver(driver::AdvectionAnalysisDriver)

    @unpack p, path_prefix = driver

    refinement_analysis = RefinementAnalysis[]
    refinement_results = RefinementAnalysisResults[]
    labels = String[]
    println("p = ", p)
    for i in eachindex(p)
        println("analysis for p = ", p[i])

        sequence_path = string(path_prefix[i], "_lambda0/")
        if isdir(sequence_path)
            push!(labels, string("\$p=\$", 
                latexstring(string(p[i])), ", central"))
            
            refinement_analysis!(refinement_analysis, refinement_results,       sequence_path, last(labels))
        end

        sequence_path = string(path_prefix[i], "_lambda1/")
        if isdir(sequence_path)
            push!(labels, string("\$p=\$", 
                latexstring(string(p[i])), ", upwind"))
            
            refinement_analysis!(refinement_analysis, refinement_results, sequence_path, last(labels))
        end

        open(string(path_prefix[i],"_refinement.txt"), "w") do io
            println(io, @capture_out tabulate_analysis_for_paper(
                (refinement_results[end-1],refinement_results[end])), "\n")
        end
    end

    plt = plot(refinement_analysis, refinement_results, d=2, 
        xlims=(10.0, 200.0),ylims=(1.0e-12,1.0))

    savefig(plt, string(path_prefix[1], "_lambda0/../",
        "refinement.pdf"))

end

function main(args)
    parsed_args = parse_commandline()
    run_driver(AdvectionAnalysisDriver(parsed_args))
end

main(ARGS)