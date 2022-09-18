function load_solution(results_path::String, time_step::Union{Int,String}=0;
    load_du=false)

    #= TODO: check if this is necessary to keep
    if time_step == 0
        dict = load(string(results_path, "project.jld2"))
        return initialize(dict["initial_data"], dict["conservation_law"],
            dict["spatial_discretization"]), dict["tspan"][1]
    else
    
        dict = load(string(results_path, "res_", time_step, ".jld2"))
        return dict["u"], dict["t"]
    end
    =#
    dict = load(string(results_path, "res_", time_step, ".jld2"))
    if load_du
        return dict["u"], dict["du"], dict["t"]
    else
       return dict["u"], dict["t"]
    end

end

function load_project(results_path::String)
    dict = load(string(results_path,"project.jld2"))
    return (dict["conservation_law"], 
        dict["spatial_discretization"], 
        dict["initial_data"],
        dict["form"],
        dict["tspan"],
        dict["strategy"])
end

function load_solver(results_path::String)
    dict = load(string(results_path,"project.jld2"))
    return Solver(dict["conservation_law"],dict["spatial_discretization"], 
        dict["form"],dict["strategy"])
end

function load_time_steps(results_path::String)
    return load_object(string(results_path, "time_steps.jld2"))
end

function load_snapshots(results_path::String, time_steps::Vector{Int}, du=false)

    dict = load(string(results_path,"project.jld2"))
    N_p, N_eq, N_el = get_dof(dict["spatial_discretization"], 
        dict["conservation_law"])
    N_dof = N_p*N_eq*N_el
    N_t = length(time_steps)

    X = Matrix{Float64}(undef, N_dof, N_t)
    times = Vector{Float64}(undef,N_t)
    for i in 1:N_t
        u, times[i] = load_solution(results_path, time_steps[i])
        X[:,i] = vec(u)
    end

    t_s = (times[N_t] - times[1])/(N_t - 1.0)
    
    return X, t_s
end

function load_snapshots_with_derivatives(results_path::String,
    time_steps::Vector{Int})

    dict = load(string(results_path,"project.jld2"))
    N_p, N_eq, N_el = get_dof(dict["spatial_discretization"], 
        dict["conservation_law"])
    N_dof = N_p*N_eq*N_el
    N_t = length(time_steps)

    U = Matrix{Float64}(undef, N_dof, N_t)
    dU = Matrix{Float64}(undef, N_dof, N_t)
    times = Vector{Float64}(undef,N_t)
    for i in 1:N_t
        u, du, times[i] = load_solution(results_path, time_steps[i],
            load_du=true)
        U[:,i] = vec(u)
        dU[:,i] = vec(du)
    end

    t_s = (times[N_t] - times[1])/(N_t - 1.0)
    
    return U,dU,t_s
end