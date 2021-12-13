module IO

    using LinearMaps: LinearMap
    using Plots: plot, plot!, scatter, savefig
    using LaTeXStrings: latexstring
    using JLD2: save, load, save_object
    using OrdinaryDiffEq: ODEIntegrator, ODEProblem, ODESolution, DiscreteCallback

    using ..ConservationLaws: ConservationLaw
    using ..SpatialDiscretizations: SpatialDiscretization
    using ..InitialConditions: AbstractInitialData
    using ..Solvers: AbstractResidualForm, AbstractStrategy, initialize

    export Plotter, visualize
    include("visualize.jl")

    export save_callback, save_project, save_solution, load_solution, load_project
    include("file.jl")

end