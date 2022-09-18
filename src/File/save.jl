function new_path(results_path::String,
    overwrite::Bool=false, clear::Bool=false)

    if !isdir(results_path)
        path = results_path
    elseif overwrite
        if clear  
            rm(results_path, force=true, recursive=true)
        end 
        path = results_path
    else
        dir_exists = true
        suffix = 1
        while dir_exists
            path = string(rstrip(results_path, '/'), "_", suffix, "/")
            if !isdir(path)
                dir_exists = false
            end
            suffix = suffix + 1
        end
    end 
    mkpath(path)
    return path
end

function save_project(
    conservation_law::AbstractConservationLaw,spatial_discretization::SpatialDiscretization,
    initial_data::AbstractParametrizedFunction, 
    form::AbstractResidualForm,
    tspan::NTuple{2,Float64}, 
    strategy::AbstractStrategy, 
    results_path::String; 
    overwrite::Bool=false,
    clear::Bool=false)

    results_path = new_path(results_path, overwrite, clear)

    save(string(results_path, "project.jld2"), 
        Dict("conservation_law" => conservation_law,
            "spatial_discretization" => spatial_discretization,
            "initial_data" => initial_data,
            "form" => form,
            "tspan" => tspan,
            "strategy" => strategy))
    
    save_object(string(results_path, "time_steps.jld2"), Int64[])

    return results_path
end

function save_solution(integrator::ODEIntegrator, results_path::String)
    open(string(results_path,"screen.txt"), "a") do io
        println(io, "writing time step ", integrator.iter,
             "  t = ", integrator.t)
    end
    save(string(results_path, "res_", integrator.iter, ".jld2"),
        Dict("u" => integrator.u, "t" => integrator.t, 
            "du" => get_du(integrator)))
    time_steps=load_object(string(results_path, "time_steps.jld2"))
    save_object(string(results_path, "time_steps.jld2"),
        push!(time_steps, integrator.iter))
end

function save_solution(u::Array{Float64,3}, t::Float64, results_path::String, time_step::Union{Int,String}=0)
    save(string(results_path, "res_", time_step, ".jld2"), 
        Dict("u" => u, "t" => t))
    if time_step isa Int
        time_steps=load_object(string(results_path, "time_steps.jld2"))
        save_object(string(results_path, "time_steps.jld2"), 
            push!(time_steps, time_step))
    end
end

function save_callback(results_path::String, interval::Int=0) 
    condition(u,t,integrator) = integrator.iter % interval == 0
    affect!(integrator) = save_solution(integrator, results_path)
    return DiscreteCallback(condition, affect!, save_positions=(true,false))
end