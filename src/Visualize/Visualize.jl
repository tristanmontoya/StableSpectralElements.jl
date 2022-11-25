
module Visualize

    using LaTeXStrings: latexstring
    using Plots: plot, plot!, savefig
    using StartUpDG: RefElemData, MeshData, AbstractElemShape, Line, Tri, Quad, map_face_nodes, find_face_nodes, nodes, vandermonde
    using WriteVTK
    using LinearMaps: LinearMap
    using LinearAlgebra: diagm
    using UnPack
    using RecipesBase
    using Triangulate
    using TetGen

    using ..SpatialDiscretizations: SpatialDiscretization, ReferenceApproximation, AbstractApproximationType, NodalTensor, ModalMulti, NodalTensor, ModalTensor
    using ..GridFunctions: AbstractGridFunction, evaluate
    using ..File: new_path
    
    export visualize, plotter
    include("plot_1d.jl")

    export low_order_subdivision, postprocess_vtk, outline_element
    include("plot_2d.jl")
    include("plot_3d.jl")
end