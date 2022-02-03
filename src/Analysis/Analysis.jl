module Analysis

    using LinearMaps: LinearMap
    using LinearAlgebra: Diagonal, dot, eigen, inv, svd, pinv, eigsortby
    using JLD2: save, load, save_object, load_object
    using Plots: plot, savefig, plot!, scatter, text
    using LaTeXStrings: LaTeXString, latexstring
    using UnPack
    using StartUpDG: MeshData

    using ..ConservationLaws: ConservationLaw
    using ..Mesh: uniform_periodic_mesh
    using ..SpatialDiscretizations: SpatialDiscretization, ReferenceApproximation
    using ..InitialConditions: AbstractInitialData, evaluate
    using ..Solvers: AbstractResidualForm, AbstractStrategy, semidiscretize
    using ..IO: new_path, load_solution, load_time_steps, Plotter

    export AbstractAnalysis, analyze, save_analysis

    abstract type AbstractAnalysis end
    
    export ErrorAnalysis, AbstractNorm, QuadratureL2, error_analysis
    include("error.jl")

    export DMDAnalysis, plot_spectrum, plot_modes 
    include("dynamics.jl")

    export ConservationAnalysis, PrimaryConservationAnalysis, EnergyConservationAnalysis
    include("conservation.jl")
end