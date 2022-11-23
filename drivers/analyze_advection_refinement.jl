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
        "--poly_degree", "-p"
            help = "polynomial degree"
            nargs = '*'
            arg_type = Int
            default = [4]
        "--element_type", "-e"
            help = "element type"
            nargs = '*'
            arg_type = String
            default = ["Tri"]
        "--scheme", "-s"
            help = "spatial_discretization scheme"
            nargs = '*'
            arg_type = String
            default = ["NodalTensor"]
        "--mapping_form", "-f"
            help = "mapping form"
            nargs = '*'
            arg_type = String
            default = ["SkewSymmetricMapping"] 
        "--name", "-n"
            help = "schme name/abbreviation"
            nargs = '*'
            arg_type = String
            default = ["Nodal"]
        "--path"
            help = "results path"
            arg_type = String
            default = "../results/advection_test/"
    end

    return parse_args(s)
end

function refinement_analysis!(refinement_analysis::Vector{RefinementAnalysis},
    refinement_results::Vector{RefinementAnalysisResults},
    conservation_results::Vector{ConservationAnalysisResultsWithDerivative},
    energy_results::Vector{ConservationAnalysisResultsWithDerivative},
    sequence_path::String, label::String)

    (conservation_law, spatial_discretization, 
        initial_data, form, tspan) = load_project(
        string(sequence_path,"grid_1/"))
    push!(refinement_analysis, RefinementAnalysis(initial_data, 
        sequence_path, sequence_path, label))
    push!(refinement_results, analyze(last(refinement_analysis),
        max_derivs=true))
    push!(conservation_results, 
        analyze(PrimaryConservationAnalysis(string(sequence_path,"grid_1/"), 
        conservation_law, spatial_discretization),
        load_time_steps(string(sequence_path,"grid_1/"))))
    push!(energy_results, 
        analyze(EnergyConservationAnalysis(string(sequence_path,"grid_1/"), 
        conservation_law, spatial_discretization),
        load_time_steps(string(sequence_path,"grid_1/"))))

end

function analyze_advection_refinement(parsed_args::Dict)

    p = parsed_args["poly_degree"]
    scheme = parsed_args["scheme"]
    element_type = parsed_args["element_type"]
    mapping_form = parsed_args["mapping_form"]
    name = parsed_args["name"]

    refinement_analysis = RefinementAnalysis[]
    refinement_results = RefinementAnalysisResults[]
    conservation_results = ConservationAnalysisResultsWithDerivative[]
    energy_results = ConservationAnalysisResultsWithDerivative[]
    labels = String[]
    
    for i in eachindex(p)
        for j in eachindex(scheme)
            path_prefix = string(parsed_args["path"], scheme[j], "_",
            element_type[j], "_", mapping_form[j], "_p", string(p[i]), "/")

            sequence_path = string(path_prefix, "central/")
            if isdir(sequence_path)
                push!(labels, string(name[j], ", \$p=\$", 
                    latexstring(string(p[i])), " (central)"))
                
                refinement_analysis!(refinement_analysis, refinement_results,
                    conservation_results, energy_results ,sequence_path, 
                    last(labels))
            end

            sequence_path = string(path_prefix, "upwind/")
            if isdir(sequence_path)
                push!(labels, string(name[j], ", \$p=\$", 
                    latexstring(string(p[i])), " (upwind)"))

                refinement_analysis!(refinement_analysis, refinement_results,
                    conservation_results, energy_results ,sequence_path, 
                    last(labels))
            end

            open(string(path_prefix,"refinement.txt"), "w") do io
                println(io, @capture_out tabulate_analysis_for_paper(
                    (refinement_results[end-1],refinement_results[end])), "\n")
            end

        cons_central = plot(conservation_results[end-1],
            ylabel="Central", link=:all, legend=:topright)
        cons_upwind = plot(conservation_results[end],
            ylabel="Upwind",legend=:none)

        plt = plot(cons_central,cons_upwind, layout=(2,1), windowsize=(400,400))
        savefig(plt, string(path_prefix, "conservation.pdf"))

        ener_central = plot(energy_results[end-1],
            ylabel="Central", link=:x, legend=:topright)
        ener_upwind = plot(energy_results[end],
            ylabel="Upwind",legend=:none)
            
        plt = plot(ener_central,ener_upwind, layout=(2,1), windowsize=(400,400))
        savefig(plt, string(path_prefix, "energy.pdf"))
        end
    end

    plt = plot(refinement_analysis, refinement_results, d=2, 
        xlims=(5.0, 200.0), xticks=[10,100], ylims=(1.0e-12,1.0), legendfontsize=10)

    savefig(plt, string(parsed_args["path"], "refinement.pdf"))
end

function main(args) analyze_advection_refinement(parse_commandline()) end

main(ARGS)