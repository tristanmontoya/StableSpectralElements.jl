module Mesh

    using StartUpDG: RefElemData, MeshData, make_periodic, Line
    using ..SpatialDiscretizations: volume_quadrature

    export reference_element, uniform_periodic_mesh

    function reference_element(elem_type::Line, quadrature_rule, num_quad_nodes)
        return RefElemData(Line(),1,quad_rule_vol=volume_quadrature(elem_type, quadrature_rule, num_quad_nodes))
    end

    function uniform_periodic_mesh(reference_element::RefElemData, 
        x_lim::NTuple{2,Float64}, K1D::Int)

        VX = collect(LinRange(x_lim[1], x_lim[2], K1D + 1))
        EToV = transpose(reshape(sort([1:K1D; 2:K1D+1]), 2, K1D))

        return make_periodic(MeshData((VX,), Matrix(EToV), reference_element))

    end

end