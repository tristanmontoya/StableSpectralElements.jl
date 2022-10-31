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