module Analysis

    using ..ConservationLaws
    using ..Mesh
    using ..SpatialDiscretizations
    using ..Solvers
    using ..IO

    export grid_refine

    include("grid_refine.jl")
    include("koopman.jl")

end