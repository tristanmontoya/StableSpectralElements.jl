module IO

    using LinearMaps: LinearMap
    using Plots: plot, plot!, scatter, savefig
    using LaTeXStrings: latexstring
    using JLD2: save, load
    using OrdinaryDiffEq: ODEIntegrator, ODEProblem, ODESolution, DiscreteCallback
    using ..SpatialDiscretizations: SpatialDiscretization
    using ..Analysis: AbstractAnalysis

    export Plotter, visualize
    include("visualize.jl")

    export save_solution, load_solution
    include("file.jl")

end