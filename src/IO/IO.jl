module IO

    using LinearMaps: LinearMap
    using Combinatorics: combinations
    using Plots: plot, plot!, scatter, savefig
    using PyCall
    import PyPlot; const plt = PyPlot
    using LaTeXStrings: latexstring
    using StartUpDG: MeshPlotter, map_face_nodes, vandermonde, find_face_nodes, Tri, Quad
    using JLD2: save, load, save_object, load_object
    using OrdinaryDiffEq: ODEIntegrator, ODEProblem, ODESolution, DiscreteCallback, get_du
    using UnPack

    using ..ConservationLaws: AbstractConservationLaw
    using ..SpatialDiscretizations: SpatialDiscretization, ReferenceApproximation, centroids
    using ..ParametrizedFunctions: AbstractParametrizedFunction, evaluate
    using ..Solvers: AbstractResidualForm, AbstractStrategy, Solver, initialize, get_dof

    export new_path, save_callback, save_project, save_solution, load_solution, load_project, load_time_steps, load_snapshots, load_snapshots_with_derivatives, load_solver
    include("file.jl")

    export Plotter, visualize
    include("visualize.jl")
end
