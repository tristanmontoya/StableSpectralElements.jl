module IO

    using LinearMaps: LinearMap
    using Plots: plot, plot!, scatter, savefig
    using LaTeXStrings
    using OrdinaryDiffEq: ODEProblem, ODESolution, DiscreteCallback
    using ..SpatialDiscretizations: SpatialDiscretization
    using ..Analysis: AbstractAnalysis

    export Plotter, visualize
    include("visualize.jl")

    export write_solution
    include("write.jl")

    include("read.jl")
end