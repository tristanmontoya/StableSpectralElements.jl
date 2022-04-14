module Analysis

    using LinearMaps: LinearMap
    using LinearAlgebra: Diagonal, dot, eigen, inv, svd, pinv, eigsortby
    using JLD2: save, load, save_object, load_object
    using Plots: plot, savefig, plot!, scatter, text, annotate!, grid, @layout
    using LaTeXStrings: LaTeXString, latexstring
    using UnPack
    using StartUpDG: MeshData
    using SparseArrays: sparse, blockdiag, kron
    using Arpack: eigs
    using OrdinaryDiffEq: OrdinaryDiffEqAlgorithm, ODESolution, solve, RK4
    using PrettyTables
    using Markdown

    using ..ConservationLaws: ConservationLaw
    using ..Mesh: uniform_periodic_mesh
    using ..SpatialDiscretizations: SpatialDiscretization, ReferenceApproximation
    using ..ParametrizedFunctions: AbstractParametrizedFunction, evaluate
    using ..Solvers: AbstractResidualForm, AbstractStrategy, Solver, semidiscretize, LinearResidual, get_dof
    using ..IO: new_path, load_project, load_solution, load_time_steps, load_snapshots, save_callback, save_solution, save_project, Plotter

    import PyPlot; const plt = PyPlot

    export AbstractAnalysis, AbstractAnalysisResults, analyze, save_analysis, plot_analysis, tabulate_analysis

    abstract type AbstractAnalysis end
    abstract type AbstractAnalysisResults end

    function save_analysis(analysis::AbstractAnalysis,
        results::AbstractAnalysisResults)
        save(string(analysis.analysis_path, "analysis.jld2"), 
            Dict("analysis" => analysis,
            "results" => results))
    end
    
    export ErrorAnalysis, AbstractNorm, QuadratureL2
    include("error.jl")

    export LinearAnalysis, DynamicalAnalysisResults, DMDAnalysis
    include("dynamics.jl")

    export ConservationAnalysis, PrimaryConservationAnalysis, EnergyConservationAnalysis
    include("conservation.jl")

    export RefinementAnalysis, RefinementAnalysisResults, run_refinement
    include("refinement.jl")
end