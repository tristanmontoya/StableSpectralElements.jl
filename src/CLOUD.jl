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

    include("IO/IO.jl")
    @reexport using .IO

    include("Analysis/Analysis.jl")
    @reexport using .Analysis

    include("PyPlotTools/PyPlotTools.jl")

    using TimerOutputs

    function CLOUD_reset_timer()
        for t in 1:Threads.nthreads()
            thread_timer = get_timer(string("thread_timer_",t))
            reset_timer!(thread_timer)
        end
        to = merge(Tuple(get_timer(string("thread_timer_",t)) 
        for t in 1:Threads.nthreads())...)
        reset_timer!(to)
    end

    function CLOUD_print_timer()
        to = merge(Tuple(get_timer(string("thread_timer_",t)) 
        for t in 1:Threads.nthreads())...)
        print_timer(to)
    end
    
    export CLOUD_reset_timer, CLOUD_print_timer
end
