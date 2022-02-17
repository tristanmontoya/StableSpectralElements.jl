module Analysis

    using LinearMaps: LinearMap
    using LinearAlgebra: Diagonal, dot, eigen, inv, svd, pinv, eigsortby
    using JLD2: save, load, save_object, load_object
    using Plots: plot, savefig, plot!, scatter, text, annotate!, grid, @layout,PyPlotBackend; PyPlotBackend()
    using LaTeXStrings: LaTeXString, latexstring
    using UnPack
    using StartUpDG: MeshData
    using SparseArrays: sparse, blockdiag, kron
    using Arpack: eigs
    using OrdinaryDiffEq: OrdinaryDiffEqAlgorithm, ODESolution, solve, RK4

    using ..ConservationLaws: ConservationLaw
    using ..Mesh: uniform_periodic_mesh
    using ..SpatialDiscretizations: SpatialDiscretization, ReferenceApproximation
    using ..InitialConditions: AbstractInitialData, evaluate
    using ..Solvers: AbstractResidualForm, AbstractStrategy, Solver, semidiscretize, LinearResidual, get_dof
    using ..IO: new_path, load_project, load_solution, load_time_steps, load_snapshots, save_callback, save_solution, save_project, Plotter

    import PyPlot; const plt = PyPlot

    export AbstractAnalysis, analyze, save_analysis, plot_analysis

    abstract type AbstractAnalysis end
    
    export ErrorAnalysis, AbstractNorm, QuadratureL2, error_analysis
    include("error.jl")

    export LinearAnalysis, DMDAnalysis, plot_spectrum, plot_modes 
    include("dynamics.jl")

    export ConservationAnalysis, PrimaryConservationAnalysis, EnergyConservationAnalysis
    include("conservation.jl")

    export run_refinement
    include("refinement.jl")
end