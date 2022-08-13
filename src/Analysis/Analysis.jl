module Analysis

    using LinearMaps: LinearMap
    using LinearAlgebra: Diagonal, dot, eigen, inv, svd, pinv, eigsortby
    using JLD2: save, load, save_object, load_object
    using Plots: plot, savefig, plot!, scatter, text, annotate!, grid, theme_palette, @layout
    using LaTeXStrings: LaTeXString, latexstring
    using UnPack
    using StartUpDG: MeshData
    using SparseArrays: sparse, blockdiag, kron
    using Arpack: eigs
    using OrdinaryDiffEq: OrdinaryDiffEqAlgorithm, ODESolution, solve, RK4
    using PrettyTables
    using Printf
    using Markdown

    using ..ConservationLaws: AbstractConservationLaw
    using ..Mesh: uniform_periodic_mesh
    using ..SpatialDiscretizations: SpatialDiscretization, ReferenceApproximation
    using ..ParametrizedFunctions: AbstractParametrizedFunction, evaluate
    using ..Solvers: AbstractResidualForm, AbstractStrategy, Solver, semidiscretize, LinearResidual, get_dof
    using ..IO: new_path, load_project, load_solution, load_time_steps, load_snapshots, save_callback, save_solution, save_project, Plotter

    export AbstractAnalysis, AbstractAnalysisResults, analyze, save_analysis, plot_analysis, plot_spectrum, plot_modes, tabulate_analysis, tabulate_analysis_for_paper

    abstract type AbstractAnalysis end
    abstract type AbstractAnalysisResults end

    function save_analysis(analysis::AbstractAnalysis,
        results::AbstractAnalysisResults)
        save(string(analysis.analysis_path, "analysis.jld2"), 
            Dict("analysis" => analysis,
            "results" => results))
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
         
    export ErrorAnalysis, AbstractNorm, QuadratureL2
    include("error.jl")

    export LinearAnalysis, DynamicalAnalysisResults, DMDAnalysis, forecast, project_onto_modes
    include("dynamics.jl")

    export ConservationAnalysis, PrimaryConservationAnalysis, EnergyConservationAnalysis, plot_evolution
    include("conservation.jl")

    export RefinementAnalysis, RefinementAnalysisResults, run_refinement
    include("refinement.jl")
end