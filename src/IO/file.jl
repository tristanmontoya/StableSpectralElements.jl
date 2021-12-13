function save_project(
    conservation_law::ConservationLaw,spatial_discretization::SpatialDiscretization,
    initial_data::AbstractInitialData, 
    form::AbstractResidualForm,
    tspan::NTuple{2,Float64}, 
    strategy::AbstractStrategy, 
    results_path::String; 
    overwrite::Bool=false)

    if !isdir(results_path) || overwrite
        new_path = results_path
    else
        dir_exists = true
        suffix = 1
        while dir_exists
            new_path = string(rstrip(results_path, '/'), "_", suffix, "/")
            if !isdir(new_path)
                dir_exists = false
            end
            suffix = suffix + 1
        end
    end

    mkpath(new_path)

    save(string(new_path, "project.jld2"), 
        Dict("conservation_law" => conservation_law,
            "spatial_discretization" => spatial_discretization,
            "initial_data" => initial_data,
            "form" => form,
            "tspan" => tspan,
            "strategy" => strategy))

    return new_path

end

function save_solution(integrator::ODEIntegrator, results_path::String)
    open(string(results_path,"screen.txt"), "a") do io
        println(io, "writing time step ", integrator.iter,
             "  t = ", integrator.t)
    end
    save(string(results_path, "res_", integrator.iter, ".jld2"),
        Dict("u" => integrator.u, "t" => integrator.t))
end

function save_solution(u::Array{Float64,3}, t::Float64, results_path::String, time_step::Union{Int,String}=0)
    save(string(results_path, "res_", time_step, ".jld2"), 
        Dict("u" => u, "t" => t))
end

function save_callback(results_path::String, interval::Int=0) 
    condition(u,t,integrator) = integrator.iter % interval == 0
    affect!(integrator) = save_solution(integrator, results_path)
    return DiscreteCallback(condition, affect!, save_positions=(true,false))
end

function load_solution(results_path::String, time_step::Union{Int,String}=0)
    if time_step == 0
        project = load(string(results_path, "project.jld2"))
        return initialize(project["initial_data"], project["conservation_law"],
            project["spatial_discretization"]), project["tspan"][1]
    else
        dict = load(string(results_path, "res_", time_step, ".jld2"))
        return dict["u"], dict["t"]
    end
end

function load_project(results_path::String)
    return load(string(results_path,"project.jld2"))
end