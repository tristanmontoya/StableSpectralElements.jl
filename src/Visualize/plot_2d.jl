"""
Subdivide to low-order mesh for plotting 
(thanks to Yimin Lin for sharing this trick)
https://github.com/yiminllin/ESDG-PosLimit
"""
function low_order_subdivision(reference_nodes::NTuple{2, Vector{Float64}},
        physical_nodes::NTuple{2, Matrix{Float64}})
    tri_in = Triangulate.TriangulateIO()
    tri_in.pointlist = permutedims(hcat(reference_nodes...))
    tri_out, _ = Triangulate.triangulate("Q", tri_in)
    connectivity = permutedims(tri_out.trianglelist)

    points = permutedims(hcat(vec(physical_nodes[1]), vec(physical_nodes[2])))
    N_sub = size(connectivity, 1)
    (N_p, N_e) = size(physical_nodes[1])

    cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE,
                 connectivity[mod1(i, N_sub), :] .+ N_p * div(i - 1, N_sub))
             for i in 1:(N_sub * N_e)]

    return points, cells
end

function postprocess_vtk(spatial_discretization::SpatialDiscretization{2},
        filename::String,
        u::Array{Float64, 3};
        e = 1,
        variable_name = "u")
    (; V_plot, reference_element) = spatial_discretization.reference_approximation
    (; x_plot) = spatial_discretization
    (; rstp) = reference_element

    points, cells = low_order_subdivision(rstp, x_plot)
    u_nodal = vec(Matrix(V_plot * u[:, e, :]))

    vtk_grid(filename, points, cells) do vtk
        vtk[variable_name] = u_nodal
    end
end

function postprocess_vtk_high_order(spatial_discretization::SpatialDiscretization{2},
        filename::String,
        u::Array{Float64, 3};
        e = 1,
        variable_name = "u")
    (; V_plot, reference_element) = spatial_discretization.reference_approximation
    (; x_plot, N_e) = spatial_discretization
    (; rstp) = reference_element

    points = permutedims(hcat(vec(x_plot[1]), vec(x_plot[2])))
    N_plot = size(V_plot, 1)

    cells = [MeshCell(VTKCellTypes.VTK_LAGRANGE_TRIANGLE,
                 collect(((k - 1) * N_plot + 1):(k * N_plot)))
             for k in 1:N_e]

    u_nodal = vec(Matrix(V_plot * u[:, e, :]))

    vtk_grid(filename, points, cells) do vtk
        vtk[variable_name] = u_nodal
    end
end

@recipe function plot(
        obj::Union{SpatialDiscretization{2},
            ReferenceApproximation{<:RefElemData{2}}};
        volume_quadrature = true,
        facet_quadrature = true,
        mapping_nodes = false,
        grid_connect = false,
        volume_quadrature_connect = false,
        mapping_nodes_connect = nothing,
        sketch = false,
        stride = nothing,
        elems = nothing,
        node_color = 1,
        facet_node_color = 2,
        mapping_node_color = 3,
        grid_line_width = 2.0,
        edge_line_width = 3.0)
    aspect_ratio --> 1.0
    legend --> false
    grid --> false
    xlabelfontsize --> 15
    ylabelfontsize --> 15
    windowsize --> (400, 400)

    if obj isa SpatialDiscretization
        (; N_e) = obj
        xlabel --> "\$x_1\$"
        ylabel --> "\$x_2\$"
        (; reference_approximation, mesh) = obj
        (; reference_element) = reference_approximation
    else
        N_e = 1
        xlims --> [-1.1, 1.1]
        ylims --> [-1.1, 1.1]
        xlabel --> "\$\\xi_1\$"
        ylabel --> "\$\\xi_2\$"
        reference_approximation = obj
    end

    if sketch
        xlabel --> ""
        ylabel --> ""
        ticks --> false
        showaxis --> false
    end

    if isnothing(elems)
        elems = 1:N_e
    end

    for k in elems
        if obj isa SpatialDiscretization
            X = function (ξ1, ξ2)
                V = vandermonde(reference_element.element_type,
                    reference_element.N,
                    ξ1,
                    ξ2) / reference_element.VDM
                return (sum(mesh.x[j, k] * V[:, j] for j in axes(mesh.x, 1)),
                    sum(mesh.y[j, k] * V[:, j] for j in axes(mesh.y, 1)))
            end
        else
            X = (x, y) -> (x, y)
        end

        (; element_type, r, s, rq, sq, rf, sf) = reference_approximation.reference_element

        if element_type isa Tri
            ref_edge_nodes = map_face_nodes(element_type, collect(LinRange(-1.0, 1.0, 40)))
            edges = find_face_nodes(element_type, ref_edge_nodes...)

            for edge in edges
                @series begin
                    linewidth --> edge_line_width
                    linecolor --> :black
                    X(ref_edge_nodes[1][edge][1:(end - 1)],
                        ref_edge_nodes[2][edge][1:(end - 1)])
                end
            end

        elseif element_type isa Quad
            N = 40
            range = collect(LinRange(-1.0, 1.0, N))
            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X(fill(-1.0, N), range)
            end
            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X(fill(1.0, N), range)
            end
            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X(range, fill(-1.0, N))
            end
            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X(range, fill(1.0, N))
            end
        end

        if volume_quadrature
            if grid_connect &&
               (reference_approximation.approx_type isa Union{NodalTensor, ModalTensor})
                if isnothing(stride)
                    stride = Int(sqrt(reference_approximation.N_q))
                end

                N1 = stride
                N2 = reference_approximation.N_q ÷ stride

                if element_type isa Tri
                    for i in 1:N1
                        @series begin
                            color --> node_color
                            linewidth --> grid_line_width
                            X(rq[i:N2:(N2 * (N1 - 1) + i)], sq[i:N2:(N2 * (N1 - 1) + i)])
                        end
                    end

                    for i in 1:N2
                        @series begin
                            color --> node_color
                            linewidth --> grid_line_width
                            X(rq[((i - 1) * N1 + 1):(i * N1)],
                                sq[((i - 1) * N1 + 1):(i * N1)])
                        end
                    end

                elseif element_type isa Quad
                    for i in 1:N1
                        @series begin
                            color --> node_color
                            linewidth --> grid_line_width
                            X(rq[i:N2:(N2 * (N1 - 1) + i)], sq[i:N2:(N2 * (N1 - 1) + i)])
                        end
                    end

                    for i in 1:N2
                        @series begin
                            color --> node_color
                            linewidth --> grid_line_width
                            X(rq[((i - 1) * N1 + 1):(i * N1)],
                                sq[((i - 1) * N1 + 1):(i * N1)])
                        end
                    end
                end
            else
                @series begin
                    seriestype --> :scatter
                    markerstrokewidth --> 0.0
                    markersize --> 5
                    color --> node_color
                    X(rq, sq)
                end
                if volume_quadrature_connect
                    j = argmin([sqrt((rq[i] + 1.0 / 3.0)^2 + (sq[i] + 1.0 / 3.0)^2)
                                for
                                i in 1:(reference_approximation.N_q)])
                    for i in 1:(reference_approximation.N_q)
                        @series begin
                            linewidth --> grid_line_width
                            linecolor --> node_color
                            x, y = [rq[j], rq[i]], [sq[j], sq[i]]
                            X(x, y)
                        end
                    end
                end
            end
        end

        if facet_quadrature
            @series begin
                seriestype --> :scatter
                markershape --> :circle
                markercolor --> facet_node_color
                markerstrokewidth --> 0.0
                markersize --> 4
                X(rf, sf)
            end
        end
        if mapping_nodes
            @series begin
                seriestype --> :scatter
                markerstrokewidth --> 0.0
                markersize --> 4
                color --> mapping_node_color
                X(r, s)
            end

            if !isnothing(mapping_nodes_connect)
                N_p = length(r)
                for i in 1:N_p
                    @series begin
                        linewidth --> grid_line_width
                        linecolor --> node_color
                        x, y = [r[mapping_nodes_connect], r[i]],
                        [s[mapping_nodes_connect], s[i]]
                        X(x, y)
                    end
                end
            end
        end
    end
end
