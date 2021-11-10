module CLOUD

    if VERSION < v"1.6"
        error("CLOUD.jl requires Julia v1.6 or newer.")
    end

    include("ConservationLaws/ConservationLaws.jl")
    include("SpatialDiscretizations/SpatialDiscretizations.jl")

end