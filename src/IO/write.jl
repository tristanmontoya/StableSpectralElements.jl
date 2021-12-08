
function write_solution(results_path::String, interval::Int=0) 
    condition(u,t,integrator) = integrator.iter % interval == 0
    affect!(integrator) = println("writing time step ", integrator.iter)
    return DiscreteCallback(condition, affect!, save_positions=(true,false))
end