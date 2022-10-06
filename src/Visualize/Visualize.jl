
module Visualize

    using LaTeXStrings: latexstring
    using Plots: plot, plot!, savefig
    using StartUpDG: AbstractElemShape, Line, Tri, Quad, map_face_nodes, find_face_nodes
    using WriteVTK
    using LinearMaps: LinearMap
    using UnPack
    using RecipesBase
    using Triangulate

    using ..SpatialDiscretizations: SpatialDiscretization, ReferenceApproximation, AbstractApproximationType, DGSEM, DGMulti, CollapsedSEM, CollapsedModal
    using ..GridFunctions: AbstractGridFunction, evaluate
    export visualize, plotter
    include("plot_1d.jl")

    export low_order_subdivision, postprocess_vtk, outline_element
    include("plot_2d.jl")
end