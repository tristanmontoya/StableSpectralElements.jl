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
    (N_p,N_el) = size(physical_nodes[1])

    cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, 
                connectivity[mod1(i,N_sub),:] .+ N_p*div(i-1,N_sub)) 
                for i in 1:N_sub*N_el]

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