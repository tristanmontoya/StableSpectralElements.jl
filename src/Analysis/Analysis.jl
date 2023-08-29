module Analysis

    using LinearMaps: LinearMap, UniformScalingMap
    using LinearAlgebra: Diagonal, mul!, lmul!, dot, norm, eigen, inv, svd, qr, pinv, eigsortby, I, rank, cond
    using JLD2: save, load, save_object, load_object
    using Plots: plot, savefig, plot!, scatter, text, annotate!, vline!, grid, theme_palette, twinx, @layout
    using RecipesBase
    using LaTeXStrings: LaTeXString, latexstring
    using StartUpDG: MeshData, vandermonde
    using SparseArrays: sparse, blockdiag, kron
    using Arpack: eigs
    using OrdinaryDiffEq: OrdinaryDiffEqAlgorithm, ODESolution, ODEIntegrator, solve, RK4, step!, reinit!
    using PrettyTables
    using Printf
    using Markdown
    using TimerOutputs

    using ..ConservationLaws: AbstractConservationLaw, entropy, conservative_to_entropy
    using ..SpatialDiscretizations: SpatialDiscretization, ReferenceApproximation, uniform_periodic_mesh, quadrature
    using ..GridFunctions: AbstractGridFunction, evaluate
    using ..Solvers: AbstractResidualForm, AbstractStrategy, Solver, AbstractMassMatrixSolver, WeightAdjustedSolver, mass_matrix,  mass_matrix_inverse, mass_matrix_solve!, semidiscretize, LinearResidual, get_dof, semi_discrete_residual!
    using ..File: new_path, load_project, load_solution, load_time_steps, load_snapshots, load_snapshots_with_derivatives, load_solver, save_callback, save_solution, save_project
    using ..Visualize: Plotter

    export AbstractAnalysis, AbstractAnalysisResults, analyze, save_analysis, plot_analysis, plot_spectrum, plot_modes, tabulate_analysis, tabulate_analysis_for_paper

    abstract type AbstractAnalysis end
    abstract type AbstractAnalysisResults end

    function save_analysis(analysis::AbstractAnalysis,
        results::AbstractAnalysisResults)
        save(string(analysis.analysis_path, "analysis.jld2"), 
            Dict("analysis" => analysis,
            "results" => results))
    end
         
    export ErrorAnalysis, AbstractNorm, QuadratureL2, QuadratureL2Normalized
    include("error.jl")

    export LinearAnalysis, DynamicalAnalysisResults, KoopmanAnalysis, AbstractKoopmanAlgorithm, StandardDMD, ExtendedDMD, KernelDMD, KernelResDMD, ExtendedResDMD, GeneratorDMD, AbstractSamplingAlgorithmx, GaussianSampling, analyze_running, forecast, monomial_basis, monomial_derivatives, make_dmd_matrices, dmd, generate_samples
    include("dynamics.jl")

    export ConservationAnalysis, PrimaryConservationAnalysis, EnergyConservationAnalysis, EntropyConservationAnalysis, ConservationAnalysisResults, ConservationAnalysisResultsWithDerivative, plot_evolution, evaluate_conservation, evaluate_conservation_residual
    include("conservation.jl")

    export RefinementAnalysis, RefinementErrorAnalysis, RefinementAnalysisResults, RefinementErrorAnalysisResults, run_refinement, get_tickslogscale
    include("refinement.jl")
end