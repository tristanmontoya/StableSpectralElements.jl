function low_order_subdivision(reference_nodes::NTuple{3,Vector{Float64}},
    physical_nodes::NTuple{3,Matrix{Float64}})

    tri_in=TetGen.RawTetGenIO{Cdouble}(
        pointlist=permutedims(hcat(reference_nodes[1], 
            reference_nodes[2], reference_nodes[3])))
    tet_out = tetrahedralize(tri_in,"Q")
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


function postprocess_vtk(
    spatial_discretization::SpatialDiscretization{3},
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