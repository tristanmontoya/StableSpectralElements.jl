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
    exact_solution::Union{AbstractParametrizedFunction{1},Nothing}=nothing,
    label::String="U^h(x,t)", label_exact::String="U(x,t)", t=0.0)

    @unpack x_plot, V_plot, N_el, directory_name = plotter

    u = convert(Matrix, V_plot * sol[:,e,:])
    if !isnothing(exact_solution)
        p = plot(vec(x_plot[1]), vec(evaluate(exact_solution,x_plot,t)[:,e,:]), 
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

function visualize(sol::Vector{Array{Float64,3}}, labels::Vector{String}, file_name::String, plotter::Plotter{1}, ylabel="U^h(x,t)"; e::Int=1)
    @unpack x_plot, V_plot, N_el, directory_name = plotter

    p = plot()
    for i in 1:length(sol)
        u = convert(Matrix, V_plot * sol[i][:,e,:])
        plot!(p, vec(vcat(x_plot[1],fill(NaN,1,N_el))), 
            vec(vcat(u,fill(NaN,1,N_el))), 
            label=latexstring(labels[i]), xlabel=latexstring("x"),
            ylabel=latexstring(ylabel))
    end

    savefig(p, string(directory_name, file_name))   
    return p
end

function visualize(sol::Union{Array{Float64,3},AbstractParametrizedFunction{2}},
    plotter::Plotter{2}, file_name::String; 
    label::String="U(\\mathbf{x},t)", 
    contours::Int=10, u_range=nothing, e::Int=1, t=0.0)

    @unpack x_plot, V_plot, N_el, directory_name = plotter

    if sol isa Array{Float64}
        u = vec(convert(Matrix, V_plot * sol[:,e,:]))
    else
        u = vec(evaluate(sol, x_plot, t)[:,e,:])
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
    markersize=4, plot_volume_nodes=true, plot_facet_nodes=true,
    label_elements=false, grid_lines=false, stride=nothing)

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
        collect(LinRange(-0.99999,0.99999, geometry_resolution)))

    
    edges = find_face_nodes(elementType, ref_edge_nodes...)

    edge_maps = Tuple(vandermonde(elementType, N, 
        ref_edge_nodes[1][edge], 
        ref_edge_nodes[2][edge]) / VDM 
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
                  markersize=markersize, color="grey")
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

function visualize(reference_approximation::ReferenceApproximation{2},
    directory_name::String, file_name::String;  markersize=4,
    plot_volume_nodes=true, plot_facet_nodes=true,grid_lines=false,
    full_connect=false, labels::NTuple{2,String}=("ξ₁","ξ₂"), 
    stride=nothing, axes=false)

    path = new_path(directory_name, true, false)
    
    @unpack rq, sq, rf, sf, elementType = reference_approximation.reference_element
    ref_edge_nodes = map_face_nodes(elementType,
        collect(LinRange(-1.0,1.0, 10)))


    if grid_lines == true && stride isa Nothing
        stride = convert(Int,sqrt(reference_approximation.N_q))
    end

    p = plt.figure()
    ax = plt.axes()
    ax.set_aspect("equal")

    if axes
        plt.xlabel(latexstring(labels[1]), fontsize=markersize*3)
        plt.ylabel(latexstring(labels[2]), fontsize=markersize*3)
        ax.spines["top"].set_visible(false)
        ax.spines["right"].set_visible(false)
        plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize=markersize*2)
        plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize=markersize*2)
    else
        ax.set_axis_off()
    end
    
    if grid_lines
        N1 = stride
        N2 = reference_approximation.N_q ÷ stride

        if elementType isa Tri
            
            for i in 1:N1
                ax.plot(vcat([rf[stride*0+i]], rq[i:N2:(N2*(N1-1) + i)],[-1.0]),
                    vcat([sf[stride*0+i]], sq[i:N2:(N2*(N1-1) + i)],[1.0]),
                        "-",
                        linewidth=markersize*0.2,
                        color="grey")
            end

            for i in 1:N2
                ax.plot(vcat([rf[3*stride-i+1]], rq[(i-1)*N1+1:i*N1],
                        [rf[1*stride+i]]), 
                    vcat([sf[3*stride-i+1]], sq[(i-1)*N1+1:i*N1],
                        [sf[1*stride+i]]),
                    "-",
                    linewidth=markersize*0.2,
                    color="grey")
            end

        elseif elementType isa Quad

            for i in 1:N1
                ax.plot(vcat([rf[stride*2+i]], rq[i:N2:(N2*(N1-1) + i)], 
                        [rf[stride*3+i]]),
                    vcat([sf[stride*2+i]], sq[i:N2:(N2*(N1-1) + i)],
                        [sf[stride*3+i]]),
                    "-",
                    linewidth=markersize*0.2,
                    color="grey")
            end

            for i in 1:N2
                ax.plot(vcat([rf[0*stride+i]], rq[(i-1)*N1+1:i*N1],
                        [rf[1*stride+i]]), 
                    vcat([sf[0*stride+i]], sq[(i-1)*N1+1:i*N1],[
                        sf[1*stride+i]]),
                    "-",
                    linewidth=markersize*0.2,
                    color="grey")
            end
        end

    end

    if full_connect
        combs = collect(combinations(1:reference_approximation.N_q,2))
        for comb ∈ combs
            ax.plot(rq[comb], sq[comb], "-", markersize=markersize, color="lightgrey")
        end
    end
    
    # plot facet edge curves
    edges = find_face_nodes(elementType, ref_edge_nodes...)
    for edge ∈ edges
        ax.plot(ref_edge_nodes[1][edge], 
                ref_edge_nodes[2][edge], 
                "-", 
                linewidth=markersize*0.3,
                color="black")
    end
    if plot_volume_nodes
        ax.plot(rq, sq, "o", markersize=markersize, color="black")
    end

    if plot_facet_nodes
        ax.plot(rf, sf, "s", 
            markersize=markersize, 
            markeredgewidth=markersize*0.25,
            color="black",
            fillstyle="none")
    end

    p.savefig(string(path, file_name),bbox_inches="tight")
end