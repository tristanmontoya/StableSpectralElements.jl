module Mesh

    using StartUpDG: RefElemData, MeshData, make_periodic, Line
    export uniform_periodic_mesh

    function uniform_periodic_mesh(reference_element::RefElemData, 
        x_lim::NTuple{2,Float64}, K1D::Int)

        VX = collect(LinRange(x_lim[1], x_lim[2], K1D + 1))
        EToV = transpose(reshape(sort([1:K1D; 2:K1D+1]), 2, K1D))

        return make_periodic(MeshData((VX,), Matrix(EToV), reference_element))
    end
end