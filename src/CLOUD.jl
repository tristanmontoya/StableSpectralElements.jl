module CLOUD

    if VERSION < v"1.6"
        error("CLOUD.jl requires Julia v1.6 or newer.")
    end

    include("ConservationLaws/ConservationLaws.jl")
    include("Mesh/Mesh.jl")
    include("SpatialDiscretizations/SpatialDiscretizations.jl")
    include("InitialConditions/InitialConditions.jl")
    include("Solvers/Solvers.jl")
    include("Analysis/Analysis.jl")
    include("IO/IO.jl")

end