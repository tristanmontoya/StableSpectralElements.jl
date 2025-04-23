function low_order_subdivision(reference_nodes::NTuple{3, Vector{Float64}},
        physical_nodes::NTuple{3, Matrix{Float64}})
    tet_in = TetGen.RawTetGenIO{Cdouble}(pointlist = permutedims(hcat(reference_nodes[1],
        reference_nodes[2],
        reference_nodes[3])))
    tet_out = tetrahedralize(tet_in, "Q")
    connectivity = permutedims(tet_out.tetrahedronlist)
    points = permutedims(hcat(vec(physical_nodes[1]), vec(physical_nodes[2]),
        vec(physical_nodes[3])))
    N_sub = size(connectivity, 1)
    (N_p, N_e) = size(physical_nodes[1])

    cells = [MeshCell(VTKCellTypes.VTK_TETRA,
                 connectivity[mod1(i, N_sub), :] .+ N_p * div(i - 1, N_sub))
             for i in 1:(N_sub * N_e)]

    return points, cells
end

function low_order_subdivision(p_vis::Int,
        p_map::Int,
        reference_element::RefElemData{3},
        mesh::MeshData{3},
        res = 0.01)
    (; rq, sq, tq, wq, VDM, element_type) = reference_element
    (; x, y, z) = mesh

    r1, s1, t1 = nodes(element_type, 1)
    facet_list = hcat(find_face_nodes(element_type, r1, s1, t1)...)

    tet_in = TetGen.RawTetGenIO{Cdouble}(pointlist = permutedims(hcat(r1, s1, t1)))
    TetGen.facetlist!(tet_in, facet_list)
    params = string("Qpq1.1a", res)
    tet_out = tetrahedralize(tet_in, params)
    connectivity = permutedims(tet_out.tetrahedronlist)
    N_sub = size(connectivity, 1)
    rp = tet_out.pointlist[1, :]
    sp = tet_out.pointlist[2, :]
    tp = tet_out.pointlist[3, :]

    V_map_to_plot = vandermonde(element_type, p_map, rp, sp, tp) / VDM
    Vp = vandermonde(element_type, p_vis, rp, sp, tp)
    Vq = vandermonde(element_type, p_vis, rq, sq, tq)
    V_quad_to_plot = Vp * inv(Vq' * diagm(wq) * Vq) * Vq' * diagm(wq)

    xp, yp, zp = (x -> V_map_to_plot * x).((x, y, z))

    points = permutedims(hcat(vec(xp), vec(yp), vec(zp)))
    (N_p, N_e) = size(xp)

    cells = [MeshCell(VTKCellTypes.VTK_TETRA,
                 connectivity[mod1(i, N_sub), :] .+ N_p * div(i - 1, N_sub))
             for i in 1:(N_sub * N_e)]

    return points, cells, V_quad_to_plot
end

function postprocess_vtk(spatial_discretization::SpatialDiscretization{3},
        filename::String,
        u::Array{Float64, 3};
        e = 1,
        p_vis = nothing,
        p_map = nothing,
        variable_name = "u")
    (; reference_element, V, V_plot) = spatial_discretization.reference_approximation
    (; mesh, x_plot) = spatial_discretization
    (; rstp) = reference_element

    if isnothing(p_vis) || isnothing(p_map)
        points, cells = low_order_subdivision(rstp, x_plot)
        u_nodal = vec(Matrix(V_plot * u[:, e, :]))
    else
        points, cells, V_quad_to_plot = low_order_subdivision(p_vis, p_map,
            reference_element, mesh)
        u_nodal = vec(Matrix(V_quad_to_plot * V * u[:, e, :]))
    end

    vtk_grid(filename, points, cells) do vtk
        vtk[variable_name] = u_nodal
    end
end

@recipe function plot(
        obj::Union{SpatialDiscretization{3},
            ReferenceApproximation{<:RefElemData{3}}};
        volume_quadrature = true,
        facet_quadrature = true,
        mapping_nodes = false,
        edges = true,
        redraw_edge = true,
        sketch = false,
        volume_connect = false,
        facet_connect = false,
        node_color = 1,
        facet_node_color = 2,
        mapping_node_color = 3,
        edge_line_width = 3.0,
        grid_line_width = 2.0,
        qf = nothing,
        q = nothing,
        facet_color_inds = nothing,
        facet_inds = nothing,
        element_inds = nothing,
        mark_vertices = false,
        outline_facets = false)
    aspect_ratio --> :equal
    legend --> false
    grid --> false
    xlabelfontsize --> 15
    ylabelfontsize --> 15
    zlabelfontsize --> 15

    left_margin --> -10mm
    top_margin --> -20mm, bottom_margin --> -10mm
    windowsize --> (400, 400)

    if sketch
        xlabel --> ""
        ylabel --> ""
        zlabel --> ""
        ticks --> false
        showaxis --> false
    end

    if obj isa SpatialDiscretization
        (; N_e) = obj
        xlabel --> "\$x_1\$"
        ylabel --> "\$x_2\$"
        zlabel --> "\$x_3\$"
        (; reference_approximation, mesh) = obj
        (; reference_element) = reference_approximation
    else
        N_e = 1
        xlims --> [-1.1, 1.1]
        ylims --> [-1.1, 1.1]
        xlabel --> "\$\\xi_1\$"
        ylabel --> "\$\\xi_2\$"
        zlabel --> "\$\\xi_3\$"
        reference_approximation = obj
    end

    for k in 1:N_e
        if element_inds isa Vector{Int}
            if !(k in element_inds)
                continue
            end
        end

        if obj isa SpatialDiscretization
            X = function (ξ1, ξ2, ξ3)
                V = vandermonde(reference_element.element_type,
                    reference_element.N,
                    ξ1,
                    ξ2,
                    ξ3) / reference_element.VDM
                return (sum(mesh.x[j, k] * V[:, j] for j in axes(mesh.x, 1)),
                    sum(mesh.y[j, k] * V[:, j] for j in axes(mesh.y, 1)),
                    sum(mesh.z[j, k] * V[:, j] for j in axes(mesh.z, 1)))
            end
        else
            X = (x, y, z) -> (x, y, z)
        end

        (; element_type, r, s, t, rq, sq, tq, rf, sf, tf) = reference_approximation.reference_element

        if edges && (element_type isa Tet)
            up = collect(LinRange(-1.0, 1.0, 40))
            down = up[end:-1:1]
            e = ones(40)

            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X([up; down; -e], [-e; -e; -e], [-e; up; down])
            end
            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X([up; down; -e], [-e; up; down], [down; -e; up])
            end
            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X([-e; -e; -e], [up; down; -e], [-e; up; down])
            end
            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X([up; down; -e], [-e; up; down], [-e; -e; -e])
            end

            if outline_facets
                @series begin
                    linewidth --> edge_line_width * 1.25
                    linecolor --> 1
                    X(0.97 * [up; down; -e], [-e; -e; -e], 0.97 * [-e; up; down])
                end
                @series begin
                    linewidth --> edge_line_width * 1.25
                    linecolor --> 3
                    X([-e; -e; -e], 0.97 * [up; down; -e], 0.97 * [-e; up; down])
                end
                @series begin
                    linewidth --> edge_line_width * 1.25
                    linecolor --> 4
                    X(0.97 * [up; down; -e], 0.97 * [-e; up; down], [-e; -e; -e])
                end
                @series begin
                    linewidth --> edge_line_width * 1.25
                    linecolor --> 2
                    X(0.97 * [up; down; -e], 0.97 * [-e; up; down], 0.97 * [down; -e; up])
                end
            end

            if mark_vertices
                @series begin
                    seriestype --> :scatter
                    markersize --> 5
                    markerstrokewidth --> 0.0
                    color --> :red
                    markershape --> :utriangle
                    X([-1.0], [-1.0], [1.0])
                end

                @series begin
                    seriestype --> :scatter
                    markersize --> 5
                    markerstrokewidth --> 0.0
                    color --> :green
                    markershape --> :utriangle
                    X([-1.0], [1.0], [-1.0])
                end
            end

        elseif edges && (element_type isa Hex)
            up = collect(LinRange(-1.0, 1.0, 40))
            down = up[end:-1:1]
            e = ones(40)

            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X([-e; -e; -e; -e], [up; e; down; -e], [-e; up; e; down])
            end
            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X([e; e; e; e], [up; e; down; -e], [-e; up; e; down])
            end
            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X([up; e; down; -e], [-e; -e; -e; -e], [-e; up; e; down])
            end
            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X([up; e; down; -e], [e; e; e; e], [-e; up; e; down])
            end
            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X([up; e; down; -e], [-e; up; e; down], [-e; -e; -e; -e])
            end
            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X([up; e; down; -e], [-e; up; e; down], [e; e; e; e])
            end
        end

        if facet_quadrature
            if facet_connect
                if element_type isa Tet
                    nodes_per_facet = reference_approximation.N_f ÷ 4
                elseif element_type isa Hex
                    nodes_per_facet = reference_approximation.N_f ÷ 6
                end

                if isnothing(qf)
                    (N1, N2) = (round(Int, sqrt(nodes_per_facet)),
                        round(Int, sqrt(nodes_per_facet)))
                else
                    (N1, N2) = qf
                end

                if isnothing(facet_color_inds)
                    facet_color_inds = facet_inds
                end
                for z in eachindex(facet_inds)
                    for i in 1:N1
                        @series begin
                            color --> facet_color_inds[z]
                            linewidth --> grid_line_width
                            start = i + nodes_per_facet * (facet_inds[z] - 1)
                            X(rf[start:N2:(N2 * (N1 - 1) + start)],
                                sf[start:N2:(N2 * (N1 - 1) + start)],
                                tf[start:N2:(N2 * (N1 - 1) + start)])
                        end
                    end

                    for i in 1:N2
                        @series begin
                            color --> facet_color_inds[z]
                            linewidth --> grid_line_width
                            X(
                                rf[((i - 1) * N1 + 1 + nodes_per_facet * (facet_inds[z] - 1)):(i * N1 + nodes_per_facet * (facet_inds[z] - 1))],
                                sf[((i - 1) * N1 + 1 + nodes_per_facet * (facet_inds[z] - 1)):(i * N1 + nodes_per_facet * (facet_inds[z] - 1))],
                                tf[((i - 1) * N1 + 1 + nodes_per_facet * (facet_inds[z] - 1)):(i * N1 + nodes_per_facet * (facet_inds[z] - 1))])
                        end
                    end
                end
            else
                @series begin
                    seriestype --> :scatter
                    markershape --> :square
                    markercolor --> facet_node_color
                    markerstrokewidth --> 0.0
                    markersize --> 4
                    X(rf, sf, tf)
                end
            end
        end

        if volume_quadrature
            if volume_connect
                if isnothing(q)
                    q = (round(Int, reference_approximation.N_q^(1 / 3)),
                        round(Int, reference_approximation.N_q^(1 / 3)),
                        round(Int, reference_approximation.N_q^(1 / 3)))
                end
                (N1, N2, N3) = q

                for l in 1:N3
                    for j in 1:N2
                        @series begin
                            color --> l + 2
                            linewidth --> grid_line_width
                            line = [(i - 1) * N2 * N3 + (j - 1) * N3 + l for i in 1:N1]
                            X(rq[line], sq[line], tq[line])
                        end
                    end
                    for i in 1:N1
                        @series begin
                            color --> l + 2
                            linewidth --> grid_line_width
                            line = [(i - 1) * N2 * N3 + (j - 1) * N3 + l for j in 1:N1]
                            X(rq[line], sq[line], tq[line])
                        end
                    end
                end
            else
                @series begin
                    seriestype --> :scatter
                    markerstrokewidth --> 0.0
                    markersize --> 5
                    color --> node_color
                    X(rq, sq, tq)
                end
            end
        end

        if mapping_nodes
            @series begin
                seriestype --> :scatter
                markerstrokewidth --> 0.0
                markersize --> 4
                color --> mapping_node_color
                X(r, s, t)
            end
        end

        if redraw_edge
            up = collect(LinRange(-1.0, 1.0, 40))
            down = up[end:-1:1]
            e = ones(40)

            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X([up; down; -e], [-e; -e; -e], [-e; up; down])
            end
            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X([up; down; -e], [-e; up; down], [down; -e; up])
            end
        end
    end
end

function plot_ref_elem(
        reference_approximation::ReferenceApproximation{
            <:RefElemData{3,
                Tet},
            <:Union{NodalTensor,
                ModalTensor}},
        title::String)
    (; p) = reference_approximation.approx_type
    vol_nodes = plot(reference_approximation,
        volume_connect = true,
        facet_connect = true,
        facet_quadrature = false,
        volume_quadrature = true,
        mapping_nodes = false,
        markersize = 4,
        camera = (115, 30),
        sketch = true,
        facet_inds = [1, 3, 4, 2],
        q = (p + 1, p + 1, p + 1),
        linewidth = 3,
        mapping_node_color = :red)
    fac_nodes = plot(reference_approximation,
        volume_connect = true,
        facet_connect = true,
        facet_quadrature = true,
        volume_quadrature = false,
        mapping_nodes = false,
        markersize = 4,
        camera = (115, 30),
        sketch = true,
        facet_inds = [1, 3, 4, 2],
        q = (p + 1, p + 1, p + 1),
        linewidth = 3)
    plt = plot(vol_nodes, fac_nodes, size = (600, 300))
    savefig(plt, title)
    run(`pdfcrop $title $title`)
end
