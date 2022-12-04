function low_order_subdivision(reference_nodes::NTuple{3,Vector{Float64}},
    physical_nodes::NTuple{3,Matrix{Float64}})

    tet_in=TetGen.RawTetGenIO{Cdouble}(
        pointlist=permutedims(hcat(reference_nodes[1], 
            reference_nodes[2], reference_nodes[3])))
    tet_out = tetrahedralize(tet_in,"Q")
    connectivity = permutedims(tet_out.tetrahedronlist)
    points = permutedims(hcat(vec(physical_nodes[1]), 
                            vec(physical_nodes[2]),
                            vec(physical_nodes[3])))
    N_sub = size(connectivity,1)
    (N_p,N_e) = size(physical_nodes[1])

    cells = [MeshCell(VTKCellTypes.VTK_TETRA, 
                connectivity[mod1(i,N_sub),:] .+ N_p*div(i-1,N_sub)) 
                for i in 1:N_sub*N_e]

   return points, cells
end

function low_order_subdivision(p_vis::Int, p_map::Int, 
    reference_element::RefElemData{3},
    mesh::MeshData{3}, res=0.01)

    @unpack rq,sq,tq, wq, VDM, elementType = reference_element
    @unpack x,y,z = mesh

    r1,s1,t1 = nodes(elementType,1)
    facet_list = hcat(find_face_nodes(elementType, r1, s1, t1)...)

    tet_in=TetGen.RawTetGenIO{Cdouble}(
        pointlist=permutedims(hcat(r1,s1,t1)))
    TetGen.facetlist!(tet_in, facet_list)
    params = string("Qpq1.1a",res)
    tet_out = tetrahedralize(tet_in,params)
    connectivity = permutedims(tet_out.tetrahedronlist)
    N_sub = size(connectivity,1)
    rp = tet_out.pointlist[1,:]
    sp =  tet_out.pointlist[2,:]
    tp = tet_out.pointlist[3,:]

    V_map_to_plot = vandermonde(elementType, p_map, rp, sp, tp) / VDM
    Vp = vandermonde(elementType, p_vis, rp, sp, tp)
    Vq = vandermonde(elementType, p_vis, rq, sq, tq)
    V_quad_to_plot = Vp * inv(Vq' * diagm(wq) * Vq) * Vq' * diagm(wq)
    
    xp, yp, zp = (x -> V_map_to_plot * x).((x, y, z))

    points = permutedims(hcat(vec(xp), 
                            vec(yp),
                            vec(zp)))
    (N_p,N_e) = size(xp)

    cells = [MeshCell(VTKCellTypes.VTK_TETRA, 
                connectivity[mod1(i,N_sub),:] .+ N_p*div(i-1,N_sub)) 
                for i in 1:N_sub*N_e]

   return points, cells, V_quad_to_plot
end

function postprocess_vtk(
    spatial_discretization::SpatialDiscretization{3},
    filename::String, u::Array{Float64,3}; e=1, p_vis=nothing, p_map=nothing,
    variable_name="u")

    @unpack reference_element, V, V_plot = spatial_discretization.reference_approximation
    @unpack mesh, x_plot = spatial_discretization
    @unpack rstp = reference_element

    if isnothing(p_vis) || isnothing(p_map)
        points, cells = low_order_subdivision(rstp, x_plot)
        u_nodal = vec(Matrix(V_plot * u[:,e,:]))
    else
        points, cells, V_quad_to_plot  = low_order_subdivision(
            p_vis, p_map, reference_element,mesh)
        u_nodal = vec(Matrix(V_quad_to_plot * V * u[:,e,:]))
    end

    vtk_grid(filename, points, cells) do vtk
        vtk[variable_name] = u_nodal
    end
end

@recipe function plot(
    obj::Union{SpatialDiscretization{3},ReferenceApproximation{3,<:AbstractElemShape,<:AbstractApproximationType}};
    volume_quadrature=true,
    facet_quadrature=true,
    mapping_nodes=true,
    sketch=false,
    grid_connect=false,
    node_color = 1,
    facet_node_color=2,
    mapping_node_color=3,
    edge_line_width = 3.0,
    grid_line_width = 2.0,
    stride=nothing,
    facet_inds=nothing,
    element_inds=nothing,
    mark_vertices=false)

    aspect_ratio --> :equal
    legend --> false
    grid --> false
    xlabelfontsize --> 15
    ylabelfontsize --> 15
    zlabelfontsize --> 15
    windowsize --> (400,400)

    if obj isa SpatialDiscretization
        @unpack N_e = obj
        xlabel --> "\$x_1\$"
        ylabel --> "\$x_2\$"
        zlabel --> "\$x_2\$"
        @unpack reference_approximation, mesh = obj
        @unpack reference_element = reference_approximation
    else
        N_e = 1
        xlims --> [-1.1, 1.1]
        ylims --> [-1.1, 1.1]
        xlabel --> "\$\\xi_1\$"
        ylabel --> "\$\\xi_2\$"
        zlabel --> "\$\\xi_3\$"
        reference_approximation = obj
    end

    if sketch
        xlabel --> ""
        ylabel --> ""
        zlabel --> ""
        ticks --> false
        showaxis --> false
    end

    for k in 1:N_e
        if element_inds isa Vector{Int}
            if !(k in element_inds) continue end
        end

        if obj isa SpatialDiscretization
            X = function(ξ1,ξ2,ξ3)
                V = vandermonde(reference_element.element_type,
                    reference_element.N,ξ1,ξ2,ξ3) / reference_element.VDM
                return (sum(mesh.x[j,k]*V[:,j] for j in axes(mesh.x,1)),
                    sum(mesh.y[j,k]*V[:,j] for j in axes(mesh.y,1)),
                    sum(mesh.z[j,k]*V[:,j] for j in axes(mesh.z,1)))
            end
        else
            X = (x,y,z) -> (x,y,z)
        end

        @unpack element_type, r, s, t, rq, sq, tq, rf, sf, tf = reference_approximation.reference_element

        if element_type isa Tet
            up = collect(LinRange(-1.0,1.0, 40))
            down = up[end:-1:1]
            e = ones(40)

            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X([up; down; -e],[-e; -e; -e], [-e; up; down])
            end
            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X([up; down; -e],[-e; up; down], [down; -e; up])
            end
            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X([-e; -e; -e],[up; down; -e], [-e; up; down])
            end
            @series begin
                linewidth --> edge_line_width
                linecolor --> :black
                X([up; down; -e],[-e; up; down], [-e; -e; -e])
            end

            if mark_vertices
                @series begin
                    markersize --> 5
                    markerstrokewidth --> 0.0
                    color --> :red
                    markershape --> :utriangle
                    X([-0.9,-0.9],[-0.9,0.9],[0.9,-0.9])
                end
            end

        end

        if volume_quadrature
            @series begin 
                seriestype --> :scatter
                markerstrokewidth --> 0.0
                markersize --> 5
                color --> node_color
                X(rq, sq, tq)
            end
        end

        if facet_quadrature

            if grid_connect &&
                (reference_approximation.approx_type isa Union{NodalTensor, ModalTensor}) && (element_type isa Tet)

                nodes_per_facet = reference_approximation.N_f ÷ 4

                if isnothing(stride)
                    stride = Int(sqrt(nodes_per_facet))
                end

                N1 = stride
                N2 = nodes_per_facet ÷ stride
                
                for z in 1:4
                    if facet_inds isa Vector{Int}
                        if !(z in facet_inds) continue end
                    end
                    for i in 1:N1
                        @series begin
                            color --> facet_node_color
                            linewidth --> grid_line_width
                            start = i + nodes_per_facet*(z-1)
                            X(rf[start:N2:(N2*(N1-1) + start)], 
                                sf[start:N2:(N2*(N1-1) + start)],
                                tf[start:N2:(N2*(N1-1) + start)])
                        end
                    end

                    for i in 1:N2
                        @series begin
                            color --> facet_node_color
                            linewidth --> grid_line_width
                            X(rf[(i-1)*N1+1+nodes_per_facet*(z-1):i*N1+ nodes_per_facet*(z-1)], sf[(i-1)*N1+1+nodes_per_facet*(z-1):i*N1+ nodes_per_facet*(z-1)], tf[(i-1)*N1+1+nodes_per_facet*(z-1):i*N1+ nodes_per_facet*(z-1)])
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

        if mapping_nodes
            @series begin 
                seriestype --> :scatter
                markerstrokewidth --> 0.0
                markersize --> 4
                color --> mapping_node_color
                X(r, s, t)
            end
        end
    end
end