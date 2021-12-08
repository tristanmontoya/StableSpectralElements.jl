module Mesh

    using StartUpDG: RefElemData, MeshData, make_periodic, Line
    using LinearAlgebra: inv, det

    export GeometricFactors, uniform_periodic_mesh

    struct GeometricFactors{d}

        # first dimension is node index, second is element
        J::Matrix{Float64}

        # first dimension is node index, second and third are matrix indices mn,
        # fourth is element
        JinvG::Array{Float64,4}

        # d-tuple of matrices, where first is element index,
        nJf::NTuple{d, Matrix{Float64}}

    end

    function uniform_periodic_mesh(reference_element::RefElemData, 
        x_lim::NTuple{2,Float64}, K1D::Int)

        VX = collect(LinRange(x_lim[1], x_lim[2], K1D + 1))
        EToV = transpose(reshape(sort([1:K1D; 2:K1D+1]), 2, K1D))

        return make_periodic(MeshData((VX,), Matrix(EToV), reference_element))
    end

    function GeometricFactors(mesh::MeshData{d}, 
        reference_element::RefElemData{d}) where {d}

        # note, here we assume that mesh is same N_q, N_f every element
        N_q = size(mesh.xyzq[1],1)
        N_f = size(mesh.xyzf[1],1)
        N_el = size(mesh.xyzq[1],2)

        J = Matrix{Float64}(undef, N_q, N_el)
        nJf = Tuple(Matrix{Float64}(undef, N_f, N_el) for m in 1:d)

        # first dimension is node index, 
        # second and third are matrix indices mn,
        # fourth is element index.
        G = Array{Float64, 4}(undef, N_q, d, d, N_el)
        JinvG = Array{Float64, 4}(undef, N_q, d, d, N_el)
        G_at_facet = Array{Float64, 4}(undef, N_f, d, d, N_el)        

        for k in 1:N_el
            for m in 1:d, n in 1:d
                for n in 1:d
                    # evaluate metric at mapping nodes
                    dxdr = reference_element.Drst[n]*mesh.xyz[m][:,k]

                    # use mapping basis to interpolate to quadrature nodes
                    G[:,m,n,k] = reference_element.Vq*dxdr
                    G_at_facet[:,m,n,k] = reference_element.Vf*dxdr
                end
            end

            # loops over slower indices
            for i in 1:N_q
                J[i,k] = det(G[i,:,:,k])
                JinvG[i,:,:,k] = J[i,k]*inv(G[i,:,:,k])
            end
        
            # get scaled normal vectors - this includes scaling for ref. quadrature weights on long side of right-angled triangle.
            # don't need actual facet Jacobian for now, probably will.
            for i in 1:N_f
                JinvG_at_facet =  det(G_at_facet[i,:,:,k])*inv(G_at_facet[i,:,:,k])
                for m in 1:d
                    nJf[m][i,k] = sum(
                        JinvG_at_facet[n,m]*reference_element.nrstJ[n][i] 
                            for n in 1:d)
                end
            end
        end
        return GeometricFactors{d}(J,JinvG,nJf)
    end

end