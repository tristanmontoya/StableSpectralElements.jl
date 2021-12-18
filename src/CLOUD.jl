module CLOUD

    if VERSION < v"1.6"
        error("CLOUD.jl requires Julia v1.6 or newer.")
    end

    using Reexport

    include("ConservationLaws/ConservationLaws.jl")
    @reexport using .ConservationLaws

    include("Mesh/Mesh.jl")
    @reexport using .Mesh

    include("SpatialDiscretizations/SpatialDiscretizations.jl")
    @reexport using .SpatialDiscretizations

    include("InitialConditions/InitialConditions.jl")
    @reexport using .InitialConditions
    
    include("Solvers/Solvers.jl")
    @reexport using .Solvers

    include("IO/IO.jl")
    @reexport using .IO

    include("Analysis/Analysis.jl")
    @reexport using .Analysis

    include("RunUtils/RunUtils.jl")
    @reexport using .RunUtils

end