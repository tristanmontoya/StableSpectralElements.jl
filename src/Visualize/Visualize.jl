
module Visualize

    using LaTeXStrings: latexstring
    using Plots: plot, plot!, savefig
    using WriteVTK
    using LinearMaps: LinearMap
    using UnPack
    using RecipesBase
    using Triangulate

    using ..SpatialDiscretizations: SpatialDiscretization
    using ..GridFunctions: AbstractGridFunction, evaluate
    export visualize, plotter
    include("plot_1d.jl")

    export low_order_subdivision, postprocess_vtk
    include("plot_2d.jl")
end