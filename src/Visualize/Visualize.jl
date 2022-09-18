
module Visualize

    using LaTeXStrings: latexstring
    using Plots: plot, plot!, savefig
    using WriteVTK
    using LinearMaps: LinearMap
    using UnPack
    using RecipesBase

    using ..SpatialDiscretizations: SpatialDiscretization
    using ..ParametrizedFunctions: AbstractParametrizedFunction, evaluate
    export visualize, plotter
    include("plot_1d.jl")
end