module CLOUD

    if VERSION < v"1.6"
        error("CLOUD.jl requires Julia v1.6 or newer.")
    end

    using Reexport

    include("ParametrizedFunctions/ParametrizedFunctions.jl")
    @reexport using .ParametrizedFunctions

    include("ConservationLaws/ConservationLaws.jl")
    @reexport using .ConservationLaws

    include("Mesh/Mesh.jl")
    @reexport using .Mesh
    
    include("Operators/Operators.jl")
    @reexport using .Operators

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
