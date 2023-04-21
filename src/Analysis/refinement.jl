"""Analyze results from grid refinement studies"""
struct RefinementAnalysis{d} <: AbstractAnalysis
    exact_solution::AbstractGridFunction{d}
    sequence_path::String
    analysis_path::String
    label::String
end

struct RefinementAnalysisResults <: AbstractAnalysisResults
    error::Matrix{Float64} # columns are solution variables
    eoc::Matrix{Union{Float64,Missing}}
    dof::Matrix{Int} # columns are N_p, N_e
    conservation::Matrix{Float64}
    energy::Matrix{Float64}
end

function analyze(analysis::RefinementAnalysis{d}, n_grids=100;
    max_derivs::Bool=false, 
    use_weight_adjusted_mass_matrix::Bool=true) where {d}

    @unpack sequence_path, exact_solution = analysis

    results_path = string(sequence_path, "grid_1/")
    if !isfile(string(results_path,"error.jld2")) error("File not found!") end

    conservation_law, spatial_discretization = load_project(results_path) 
    (N_p, N_c, N_e) = get_dof(spatial_discretization, conservation_law)
    dof = [N_p N_e]
    time_steps = load_time_steps(results_path)
    N_t = last(time_steps)
    u, _ = load_solution(results_path, N_t)
    error = transpose(analyze(ErrorAnalysis(results_path, conservation_law,  
        spatial_discretization), u, exact_solution))

    if use_weight_adjusted_mass_matrix
        mass_solver = WeightAdjustedSolver(spatial_discretization)
    else
        mass_solver = CholeskySolver(spatial_discretization)
    end

    if max_derivs
        conservation_results = analyze(
            PrimaryConservationAnalysis(results_path, 
            conservation_law, spatial_discretization), time_steps)
        energy_results = analyze(
                EnergyConservationAnalysis(results_path, 
                conservation_law, spatial_discretization,
                mass_solver), time_steps)
        conservation = [maximum(abs.(conservation_results.dEdt[:,e])) 
            for e in 1:N_c]'
        energy = [maximum((energy_results.dEdt[:,e])) for e in 1:N_c]'
    else
        conservation = transpose(
            analyze(PrimaryConservationAnalysis(results_path, 
            conservation_law, spatial_discretization), 0, N_t)[3])
        energy = transpose(analyze(EnergyConservationAnalysis(results_path, 
            conservation_law, spatial_discretization, mass_solver), 0, N_t)[3])
    end

    eoc = fill!(Array{Union{Float64, Missing}}(undef,1,N_c), missing)

    i = 2
    grid_exists = true
    
    while isdir(string(sequence_path, "grid_", i, "/")) && i <= n_grids
        results_path = string(sequence_path, "grid_", i, "/")
        
        conservation_law, spatial_discretization = load_project(results_path) 
        (N_p, N_c, N_e) = get_dof(spatial_discretization, conservation_law)
        dof = [dof; [N_p N_e]]

        if use_weight_adjusted_mass_matrix
            mass_solver = WeightAdjustedSolver(spatial_discretization)
        else
            mass_solver = CholeskySolver(spatial_discretization)
        end    

        if !isfile(string(results_path), "error.jld2")  
            error = [error; fill(NaN, 1, N_c)]
            eoc = [eoc; fill(NaN, 1, N_c)]
            conservation = [conservation; fill(NaN, 1, N_c)]
            energy = [energy; fill(NaN, 1, N_c)]
        else
            time_steps = load_time_steps(results_path)
            N_t = last(time_steps)
            u, _ = load_solution(results_path, N_t)
            error = [error; transpose(analyze(ErrorAnalysis(results_path,       
                conservation_law, spatial_discretization), u, exact_solution))]
            eoc = [eoc; transpose([ 
                (log(error[i,e]) - log(error[i-1,e])) /
                    (log((dof[i,1]*dof[i,2])^(-1.0/d) ) - 
                    log((dof[i-1,1]*dof[i-1,2])^(-1.0/d)))
                for e in 1:N_c])]

            if max_derivs
                conservation_results = analyze(
                    PrimaryConservationAnalysis(results_path, 
                    conservation_law, spatial_discretization), time_steps)
                energy_results = analyze(
                        EnergyConservationAnalysis(results_path, 
                        conservation_law, spatial_discretization,
                        mass_solver), time_steps)
                conservation = [conservation; 
                    [maximum(abs.(conservation_results.dEdt[:,e])) for e in 1:N_c]']
                energy = [energy;
                    [maximum((energy_results.dEdt[:,e])) for e in 1:N_c]']
            else
                conservation = [conservation; transpose(
                    analyze(PrimaryConservationAnalysis(results_path, 
                    conservation_law, spatial_discretization), 0, N_t)[3])]
                energy = [energy; 
                    transpose(analyze(EnergyConservationAnalysis(results_path, 
                    conservation_law, spatial_discretization,
                    mass_solver), 0, N_t)[3])]
            end
        end

        if !isfile(string(sequence_path, "grid_", i+1, "/error.jld2"))
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

@recipe function plot(analysis::Vector{RefinementAnalysis{d}},
    results::Vector{RefinementAnalysisResults}; n_grids=nothing, pairs=true, xlims=nothing, reference_line=nothing, e=1) where {d}

    if d == 1 xlabel --> latexstring("\\mathrm{DOF}")
    elseif d == 2 xlabel --> latexstring("\\sqrt{\\mathrm{DOF}}")
    else xlabel --> latexstring(string("\\sqrt"),"[", d, "]{\\mathrm{DOF}}") end
    if !isnothing(xlims) xticks --> get_tickslogscale(xlims) end

    ylabel --> LaTeXString("Error Metric")
    xaxis --> :log10
    yaxis --> :log10
    markersize --> 5
    windowsize --> (400,400)
    legend --> :bottomleft
    legendfontsize --> 10
    xlims --> xlims
    fontfamily --> "Computer Modern"
    for i in eachindex(analysis)
        @series begin
            if pairs && iseven(i)
                markershape --> :circle
                markerstrokewidth --> 2.0
            else
                markershape --> :square
                markerstrokewidth --> 2.0
            end
            if pairs && (i % 4 == 1 || i % 4 == 2)
                linestyle --> :solid
            else
                linestyle --> :dash
            end
            label --> analysis[i].label
            linecolor --> (i-1) ÷ 2 + 1
            markercolor --> (i-1) ÷ 2 + 1
            if isnothing(n_grids)
                (results[i].dof[:,1].*results[i].dof[:,2]).^(1.0/d), results[i].error[:,e]
            else
                (results[i].dof[1:n_grids[i],1].*results[i].dof[1:n_grids[i],2]).^(1.0/d), results[i].error[1:n_grids[i],e]
            end
        end
    end

    if !isnothing(reference_line)
        for i in eachindex(reference_line)
            @series begin
                linecolor --> :black
                linestyle --> :solid
                label --> ""
                x = [reference_line[i][3], reference_line[i][4]]
                x, reference_line[i][2]./(x.^reference_line[i][1])
            end
        end
    end
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
        tf = tf_unicode_rounded)
end
    
function tabulate_analysis_for_paper(results::NTuple{2,RefinementAnalysisResults}; e=1)
    cons = Tuple(results[i].conservation[:,e] for i in 1:2)
    ener = Tuple(results[i].energy[:,e] for i in 1:2)
    err =Tuple(results[i].error[:,e] for i in 1:2)
    eoc =Tuple(results[i].eoc[:,e] for i in 1:2)

    tab = hcat(results[1].dof[:,2], cons..., ener..., err..., eoc...)

    latex_header = vcat(["\$N_e\$", "Conservation Metric", "", "Energy Metric", "", "Error Metric","", "Order",""])
    pretty_table(tab, header=latex_header, backend = Val(:latex),
        formatters = (ft_nomissing, ft_printf("& %d", [1,]), ft_printf("%.3e", [2,3,4,5,6,7]), ft_printf("%1.2f", [8,9]),
        (v, i, j) -> (v == "NaN") ? "---" : v),
        
        tf = tf_latex_booktabs)
end

"""
This function is provided by getzze on GitHub,
    see https://github.com/JuliaPlots/Plots.jl/issues/3318

get_tickslogscale(lims; skiplog=false)
Return a tuple (ticks, ticklabels) for the axis limit `lims`
where multiples of 10 are major ticks with label and minor ticks have no label
skiplog argument should be set to true if `lims` is already in log scale.
"""
function get_tickslogscale(lims::Tuple{T, T}; skiplog::Bool=false) where {T<:AbstractFloat}
mags = if skiplog
    # if the limits are already in log scale
    floor.(lims)
else
    floor.(log10.(lims))
end
rlims = if skiplog; 10 .^(lims) else lims end

total_tickvalues = []
total_ticknames = []

rgs = range(mags..., step=1)
for (i, m) in enumerate(rgs)
    if m >= 0
        tickvalues = range(Int(10^m), Int(10^(m+1)); step=Int(10^m))
        ticknames  = vcat([string(round(Int, 10^(m)))],
                        ["" for i in 2:9],
                        [string(round(Int, 10^(m+1)))])
    else
        tickvalues = range(10^m, 10^(m+1); step=10^m)
        ticknames  = vcat([string(10^(m))], ["" for i in 2:9], [string(10^(m+1))])
    end

    if i==1
        # lower bound
        indexlb = findlast(x->x<rlims[1], tickvalues)
        if isnothing(indexlb); indexlb=1 end
    else
        indexlb = 1
    end
    if i==length(rgs)
        # higher bound
        indexhb = findfirst(x->x>rlims[2], tickvalues)
        if isnothing(indexhb); indexhb=10 end
    else
        # do not take the last index if not the last magnitude
        indexhb = 9
    end

    total_tickvalues = vcat(total_tickvalues, tickvalues[indexlb:indexhb])
    total_ticknames = vcat(total_ticknames, ticknames[indexlb:indexhb])
end
return (total_tickvalues, total_ticknames)
end