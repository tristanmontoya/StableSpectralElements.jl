function write_callback(integrator::ODEIntegrator, results_path::String)
    println("writing time step ", integrator.iter, "  t = ", integrator.t)
    save(string(results_path, "res_", integrator.iter, ".jld2"),
        Dict("u" => integrator.u, "t" => integrator.t, "p" => integrator.p))
end

function save_solution(results_path::String, interval::Int=0) 
    condition(u,t,integrator) = integrator.iter % interval == 0
    affect!(integrator) = write_callback(integrator, results_path)
    return DiscreteCallback(condition, affect!, 
        save_positions=(true,false))
end

function load_solution(results_path::String, time_step::Int)
    dict = load(string(results_path, "res_", time_step, ".jld2"))
    return dict["u"], dict["t"], dict["p"]
end