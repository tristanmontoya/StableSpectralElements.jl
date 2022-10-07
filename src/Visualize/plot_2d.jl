"""
Subdivide to low-order mesh for plotting 
(thanks to Yimin Lin for sharing this trick)
https://github.com/yiminllin/ESDG-PosLimit
"""
function low_order_subdivision(reference_nodes::NTuple{2,Vector{Float64}},
     physical_nodes::NTuple{2,Matrix{Float64}})

    tri_in = Triangulate.TriangulateIO()
    tri_in.pointlist = permutedims(hcat(reference_nodes...))
    tri_out, _ = Triangulate.triangulate("Q", tri_in)
    connectivity = permutedims(tri_out.trianglelist)
    points = transpose(hcat(vec(physical_nodes[1]), vec(physical_nodes[2])))
    N_sub = size(connectivity,1)
    (N_p,N_e) = size(physical_nodes[1])

    cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, 
                connectivity[mod1(i,N_sub),:] .+ N_p*div(i-1,N_sub)) 
                for i in 1:N_sub*N_e]

    return points, cells
end

function postprocess_vtk(spatial_discretization::SpatialDiscretization{2},
    filename::String, u::Array{Float64,3}; e=1, variable_name="u")
    @unpack V_plot, reference_element = spatial_discretization.reference_approximation
    @unpack x_plot = spatial_discretization
    @unpack rstp = reference_element

    points, cells = low_order_subdivision(rstp, x_plot)
    u_nodal = vec(Matrix(V_plot * u[:,e,:]))

    vtk_grid(filename, points, cells) do vtk
        vtk[variable_name] = u_nodal
    end
end

@recipe function plot(
    reference_approximation::ReferenceApproximation{2,<:AbstractElemShape,<:AbstractApproximationType}; 
    volume_quadrature=true,
    facet_quadrature=true,
    grid_connect=false,
    sketch=false,
    stride = nothing,
    node_color = 1,
    grid_line_width = 2.0,
    X=nothing)

    @unpack element_type, rq, sq, rf, sf = reference_approximation.reference_element

    if isnothing(X)
        xlims --> [-1.1, 1.1]
        ylims --> [-1.1, 1.1]
        X = (x,y) -> (x,y)
    end
    aspect_ratio --> 1.0
    legend --> false
    grid --> false
    xlabelfontsize --> 15
    ylabelfontsize --> 15

    if sketch
        xlabel --> ""
        ylabel --> ""
        ticks --> false
        showaxis --> false
    else
        xlabel --> "\$\\xi_1\$"
        ylabel --> "\$\\xi_2\$"
    end

    ref_edge_nodes = map_face_nodes(element_type,
        collect(LinRange(-1.0,1.0, 40)))
    edges = find_face_nodes(element_type, ref_edge_nodes...)
    
    for edge ∈ edges
        @series begin
            linewidth --> 3.0
            linecolor --> :black
            X(ref_edge_nodes[1][edge][1:end-1], ref_edge_nodes[2][edge][1:end-1])
        end
    end

    if grid_connect && 
        (reference_approximation.approx_type isa Union{DGSEM, CollapsedModal, CollapsedSEM})

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
                    X(rq[i:N2:(N2*(N1-1) + i)], sq[i:N2:(N2*(N1-1) + i)])
                end
            end

            for i in 1:N2
                @series begin
                    color --> node_color
                    linewidth --> grid_line_width
                    X(rq[(i-1)*N1+1:i*N1], sq[(i-1)*N1+1:i*N1])
                end
            end

        elseif element_type isa Quad

            for i in 1:N1
                @series begin
                    color --> node_color
                    linewidth --> grid_line_width
                    X(rq[i:N2:(N2*(N1-1) + i)], sq[i:N2:(N2*(N1-1) + i)])
                end
            end

            for i in 1:N2
                @series begin
                    color --> node_color
                    linewidth --> grid_line_width
                    X(rq[(i-1)*N1+1:i*N1], sq[(i-1)*N1+1:i*N1])
                end
            end
        end
        volume_quadrature = false
    end

    if volume_quadrature
        @series begin 
            seriestype --> :scatter
            markerstrokewidth --> 0.0
            markersize --> 5
            color --> node_color
            X(rq, sq)
        end
    end

    if facet_quadrature
        @series begin 
            seriestype --> :scatter
            markershape --> :circle
            markercolor --> node_color
            markerstrokewidth --> 0.0
            markersize --> 4
            X(rf, sf)
        end
    end


end