module Analysis

using LinearMaps: LinearMap, UniformScalingMap
using LinearAlgebra:
                     Diagonal,
                     diag,
                     diagm,
                     mul!,
                     lmul!,
                     dot,
                     norm,
                     eigen,
                     inv,
                     svd,
                     qr,
                     pinv,
                     eigsortby,
                     I,
                     rank,
                     cond
using JLD2: save, load, save_object, load_object
using Plots:
             plot,
             savefig,
             plot!,
             scatter,
             text,
             annotate!,
             vline!,
             grid,
             theme_palette,
             twinx,
             @layout
using LaTeXStrings: LaTeXString, latexstring
using SparseArrays: sparse, blockdiag, kron!
using OrdinaryDiffEq:
                      OrdinaryDiffEqAlgorithm,
                      ODESolution, ODEIntegrator, solve, RK4,
                      step!, reinit!
using StartUpDG: MeshData, vandermonde
using RecipesBase
using Printf
using PrettyTables
using Markdown
using TimerOutputs
using BenchmarkTools

using ..ConservationLaws
using ..SpatialDiscretizations
using ..GridFunctions
using ..Solvers
using ..MatrixFreeOperators
using ..File
using ..Visualize

export AbstractAnalysis,
       AbstractAnalysisResults,
       analyze,
       analyze_new,
       save_analysis,
       tabulate_analysis,
       tabulate_analysis_for_paper

abstract type AbstractAnalysis end
abstract type AbstractAnalysisResults end

function save_analysis(analysis::AbstractAnalysis, results::AbstractAnalysisResults)
    save(string(analysis.analysis_path, "analysis.jld2"),
        Dict("analysis" => analysis, "results" => results))
end

export ErrorAnalysis, AbstractNorm, QuadratureL2, QuadratureL2Normalized
include("error.jl")

export ConservationAnalysis,
       PrimaryConservationAnalysis,
       EnergyConservationAnalysis,
       EntropyConservationAnalysis,
       ConservationAnalysisResults,
       ConservationAnalysisResultsWithDerivative,
       evaluate_conservation,
       evaluate_conservation_residual
include("conservation.jl")

export RefinementAnalysis,
       RefinementErrorAnalysis,
       RefinementAnalysisResults,
       RefinementErrorAnalysisResults,
       run_refinement,
       get_tickslogscale
include("refinement.jl")

export scaling_test_euler_2d
include("benchmark.jl")
end
