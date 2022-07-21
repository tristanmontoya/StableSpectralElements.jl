"""Analyze results from grid refinement studies"""
struct RefinementAnalysis{d} <: AbstractAnalysis
    exact_solution::AbstractParametrizedFunction{d}
    sequence_path::String
    analysis_path::String
    label::String
end

struct RefinementAnalysisResults <: AbstractAnalysisResults
    error::Matrix{Float64} # columns are solution variables
    eoc::Matrix{Union{Float64,Missing}}
    dof::Matrix{Int} # columns are N_p, N_el
    conservation::Matrix{Float64}
    energy::Matrix{Float64}
end

function analyze(analysis::RefinementAnalysis{d}, n_grids=100) where {d}

    @unpack sequence_path, exact_solution = analysis

    results_path = string(sequence_path, "grid_1/")
    conservation_law, spatial_discretization = load_project(results_path) 
    (N_p, N_eq, N_el) = get_dof(spatial_discretization, conservation_law)
    dof = [N_p N_el]
    u, _ = load_solution(results_path, "final")
    error = transpose(analyze(ErrorAnalysis(results_path, conservation_law,  
        spatial_discretization), u, exact_solution))
    conservation = transpose(analyze(PrimaryConservationAnalysis(results_path, 
        conservation_law, spatial_discretization))[3])
    energy = transpose(analyze(EnergyConservationAnalysis(results_path, 
        conservation_law, spatial_discretization))[3])

    eoc = fill!(Array{Union{Float64, Missing}}(undef,1,N_eq), missing)

    i = 2
    grid_exists = true
    while grid_exists && i <= n_grids
        results_path = string(sequence_path, "grid_", i, "/")
        conservation_law, spatial_discretization = load_project(results_path) 
        (N_p, N_eq, N_el) = get_dof(spatial_discretization, conservation_law)
        dof = [dof; [N_p N_el]]
        u, _ = load_solution(results_path, "final")
        error = [error; transpose(analyze(ErrorAnalysis(results_path,       
            conservation_law, spatial_discretization), u, exact_solution))]
        eoc = [eoc; transpose([ 
            (log(error[i,e]) - log(error[i-1,e])) /
                (log((dof[i,1]*dof[i,2])^(-1.0/d) ) - 
                log((dof[i-1,1]*dof[i-1,2])^(-1.0/d)))
            for e in 1:N_eq])]
        conservation = [conservation; 
            transpose(analyze(PrimaryConservationAnalysis(results_path, 
                conservation_law, spatial_discretization))[3])]
        energy = [energy;
            transpose(analyze(EnergyConservationAnalysis(results_path, 
            conservation_law, spatial_discretization))[3])]

        if !isdir(string(sequence_path, "grid_", i+1, "/"))
            grid_exists = false
        end
        i = i+1
    end
    return RefinementAnalysisResults(error, eoc, dof, conservation, energy)
end

function plot_analysis(analysis::RefinementAnalysis{d},
    results::RefinementAnalysisResults; e=1) where {d}

    @unpack analysis_path = analysis
    @unpack error, dof = results

    if d == 1
        xlabel = latexstring("\\mathrm{DOF}")
    elseif d == 2
        xlabel = latexstring("\\sqrt{\\mathrm{DOF}}")
    else
        xlabel = latexstring(string("\\sqrt"),"[", d, "]{\\mathrm{DOF}}")
    end

    p = plot((dof[:,1].*dof[:,2]).^(1.0/d), error[:,e], 
        xlabel=xlabel, ylabel=(LaTeXString("\$L^2\$ Error")), 
        xaxis=:log, yaxis=:log, legend=false, linecolor="black", markershape=:circle, markercolor="black")
    savefig(p, string(analysis_path, "refinement.pdf"))
    return p
end

function plot_analysis(analysis::Vector{RefinementAnalysis{d}},
    results::Vector{RefinementAnalysisResults}; ylabel=LaTeXString("\$L^2\$ Error"), ylims=nothing, xlims=nothing, pairs=true, 
    filename="refinement.pdf",
    reference_line=nothing, e=1, plots_path=nothing) where {d}

    if isnothing(plots_path)
        plots_path = analysis[1].analysis_path
    end

    if d == 1
        xlabel = latexstring("\\mathrm{DOF}")
    elseif d == 2
        xlabel = latexstring("\\sqrt{\\mathrm{DOF}}")
    else
        xlabel = latexstring(string("\\sqrt"),"[", d, "]{\\mathrm{DOF}}")
    end

    n = length(analysis)
    colors = [(i-1) ÷ 2 + 1 for i in 1:n]
    p = plot()
    for i in 1:n
        if pairs && iseven(i)
            style = :dash
            shape = :circle
        else
            style = :solid
            shape = :square
        end
        plot!((results[i].dof[:,1].*results[i].dof[:,2]).^(1.0/d), 
            results[i].error[:,e], label=analysis[i].label,
            xaxis=:log10, yaxis=:log10, linestyle=style,
            markerstrokewidth=0.0, markershape=shape,
            linecolor=colors[i], markercolor=colors[i],  markersize=5)
    end
    if !isnothing(reference_line)
        for i in 1:length(reference_line)
            plot!(p, 
                (results[1].dof[end-1:end,1].*results[1].dof[end-1:end,2]).^(1.0/d),
                reference_line[i][2]./((results[1].dof[end-1:end,1].*results[1].dof[end-1:end,2]).^(1.0/d)).^reference_line[i][1], linecolor=:black, linestyle=:solid, label="")
        end
    end

    plot!(p, windowsize=(400,400), legend=:bottomleft, legendfontsize=10,
    xlabel=xlabel, ylabel=ylabel)
    if !isnothing(xlims)
        plot!(p, xlims=xlims, xticks=get_tickslogscale(xlims))
    end
    if !isnothing(ylims)
        plot!(p, ylims=ylims)
    end

    savefig(p, string(plots_path, filename))
    return p
end


function tabulate_analysis(results::RefinementAnalysisResults; e=1, 
    print_latex=true)
    tab = hcat(results.dof[:,2], results.conservation[:,e], 
        results.energy[:,e], results.error[:,e], results.eoc[:,e])

    if print_latex
        latex_header = ["Elements", "Conservation Metric", "Energy Metric",
            "\$L^2\$ Error", "Order"]
        pretty_table(tab, header=latex_header, backend = Val(:latex),
            formatters = (ft_nomissing, ft_printf("%d", [1,]), ft_printf("%.5e", [2,3,4]), ft_printf("%1.5f", [5,])), tf = tf_latex_booktabs)
    end

    return pretty_table(String, tab, header=["Elements", "Conservation Metric",
        "Energy Metric", "L² Error", "Order"],
        formatters = (ft_nomissing, ft_printf("%d", [1,]), 
            ft_printf("%.5e", [2,3,4,]),
        ft_printf("%1.5f", [5,])),
        tf = tf_markdown)
end
    
function tabulate_analysis_for_paper(results::NTuple{2,RefinementAnalysisResults}; e=1)
    cons = Tuple(results[i].conservation[:,e] for i in 1:2)
    ener = Tuple(results[i].energy[:,e] for i in 1:2)
    err =Tuple(results[i].error[:,e] for i in 1:2)
    eoc =Tuple(results[i].eoc[:,e] for i in 1:2)

    tab = hcat(results[1].dof[:,2], cons..., ener..., err..., eoc...)

    latex_header = vcat(["\$N_e\$", "Conservation Metric", "", "Energy Metric", "", "\$L^2\$ Error","", "Order",""])
    pretty_table(tab, header=latex_header, backend = Val(:latex),
        formatters = (ft_nomissing, ft_printf("& %d", [1,]), ft_printf("%.3e", [2,3,4,5,6,7]), ft_printf("%1.2f", [8,9])), tf = tf_latex_booktabs)
end
    
    