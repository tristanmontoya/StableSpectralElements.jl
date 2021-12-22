struct Plotter{d}
    x_plot::NTuple{d,Matrix{Float64}}
    V_plot::LinearMap
    N_el::Int
    directory_name::String
end

function Plotter(spatial_discretization::SpatialDiscretization{d},directory_name::String) where {d}
    path = new_path(directory_name, true, false)

    return Plotter{d}(spatial_discretization.x_plot, 
        spatial_discretization.reference_approximation.V_plot,
        spatial_discretization.N_el, path)
end

function visualize(sol::Array{Float64,3}, 
    plotter::Plotter{1}, file_name::String; e::Int=1,
    exact_solution::Union{AbstractInitialData{1},Nothing}=nothing,
    label::String="U^h(x,t)", label_exact::String="U(x,t)")

    @unpack x_plot, V_plot, N_el, directory_name = plotter

    u = convert(Matrix, V_plot * sol[:,e,:])
    if !isnothing(exact_solution)
        p = plot(vec(x_plot[1]), vec(evaluate(exact_solution,x_plot)[:,e,:]), 
            label=latexstring(label_exact))
        plot!(p, vec(vcat(x_plot[1],fill(NaN,1,N_el))), 
            vec(vcat(u,fill(NaN,1,N_el))), 
            label=latexstring(label), xlabel=latexstring("x"))
    else 
        p = plot(vec(vcat(x_plot[1],fill(NaN,1,N_el))), 
        vec(vcat(u,fill(NaN,1,N_el))), 
        label=latexstring(label), xlabel=latexstring("x"))
    end

    savefig(p, string(directory_name, file_name))   
    return p
end

function visualize(sol::Union{Array{Float64,3},AbstractInitialData},
    plotter::Plotter{2}, file_name::String; 
    label::String="U(\\mathbf{x},t)", 
    contours::Int=10, u_range=nothing, e::Int=1)

    @unpack x_plot, V_plot, N_el, directory_name = plotter

    if sol isa Array{Float64}
        u = vec(convert(Matrix, V_plot * sol[:,e,:]))
    else
        u = vec(evaluate(sol, x_plot)[:,e,:])
    end
    if isnothing(u_range)
        u_range = (minimum(u), maximum(u))
    end

    p = plt.figure(1)
    ax = plt.axes()
    ax.set_aspect("equal")
    contour = ax.tricontourf(vec(x_plot[1]),vec(x_plot[2]), u,
        cmap="viridis", levels=LinRange(
            u_range[1]-0.1*(u_range[2] - u_range[1]),
            u_range[2]+0.1*(u_range[2] - u_range[1]), contours))
    plt.xlabel(latexstring("x_1"))
    plt.ylabel(latexstring("x_2"))
    cbar = p.colorbar(contour)
    cbar.ax.set_ylabel(latexstring(label))
    p.savefig(string(directory_name, file_name))
end