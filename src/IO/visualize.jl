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

    p = plt.figure()
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

function visualize(spatial_discretization::SpatialDiscretization{2},
    directory_name::String, file_name::String; geometry_resolution=5, 
    markersize=4, plot_volume_nodes=true, plot_facet_nodes=true,label_elements=false, grid_lines=false, stride=nothing)

    path = new_path(directory_name, true, false)

    @unpack N_el, mesh, reference_approximation = spatial_discretization
    @unpack elementType, N, VDM = reference_approximation.reference_element
    
    if grid_lines == true && stride isa Nothing
        stride = convert(Int,sqrt(reference_approximation.N_q))
    end

    p = plt.figure()
    ax = plt.axes()
    ax.set_aspect("equal")
    plt.xlabel(latexstring("x_1"))
    plt.ylabel(latexstring("x_2"))
    
    ref_edge_nodes = map_face_nodes(elementType,
        collect(LinRange(-1.0,1.0, geometry_resolution)))

    edges = find_face_nodes(elementType, ref_edge_nodes...)

    edge_maps = Tuple(vandermonde(elementType, N, 
        ref_edge_nodes[1][edge], ref_edge_nodes[2][edge]) / VDM 
        for edge ∈ edges)

    if label_elements
        x_c = centroids(spatial_discretization)
    end

    for k in 1:N_el

        if grid_lines
            N1 = stride
            N2 = reference_approximation.N_q ÷ stride
            for i in 1:N1
                ax.plot(mesh.xq[i:N2:(N2*(N1-1) + i),k], 
                mesh.yq[i:N2:(N2*(N1-1) + i),k],
                        "-",
                        linewidth=markersize*0.2,
                        color="grey")
            end
            for i in 1:N2
                ax.plot(mesh.xq[(i-1)*N1+1:i*N1,k], 
                mesh.yq[(i-1)*N1+1:i*N1,k],
                        "-",
                        linewidth=markersize*0.2,
                        color="grey")
            end
        end
            
        # plot facet edge curves
        for V_edge ∈ edge_maps
            edge_points = (V_edge * mesh.x[:,k], 
                V_edge * mesh.y[:,k])
            ax.plot(edge_points[1], 
                    edge_points[2], 
                    "-", 
                    linewidth=markersize*0.25,
                    color="black")
        end

        if plot_volume_nodes
            ax.plot(mesh.xq[:,k], mesh.yq[:,k], "o",
                  markersize=markersize)
        end
        if plot_facet_nodes
            ax.plot(mesh.xf[:,k], mesh.yf[:,k], "s", 
                markersize=markersize, 
                markeredgewidth=markersize*0.25,
                color="black",
                fillstyle="none")
        end

        if label_elements
            ax.text(x_c[k][1], x_c[k][2], string(k))
        end
    end
    p.savefig(string(path, file_name))
end