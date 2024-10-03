module StableSpectralElements

if VERSION < v"1.10"
    error("StableSpectralElements.jl requires Julia v1.10 or newer.")
end

using Reexport

include("GridFunctions/GridFunctions.jl")
@reexport using .GridFunctions

include("ConservationLaws/ConservationLaws.jl")
@reexport using .ConservationLaws

include("MatrixFreeOperators/MatrixFreeOperators.jl")
@reexport using .MatrixFreeOperators

include("SpatialDiscretizations/SpatialDiscretizations.jl")
@reexport using .SpatialDiscretizations

include("Solvers/Solvers.jl")
@reexport using .Solvers

include("File/File.jl")
@reexport using .File

include("Visualize/Visualize.jl")
@reexport using .Visualize

include("Analysis/Analysis.jl")
@reexport using .Analysis
end
