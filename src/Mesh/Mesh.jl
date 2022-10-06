module Mesh

    using UnPack
    using StartUpDG: RefElemData, MeshData, uniform_mesh, connect_mesh, build_node_maps, diff, make_periodic, geometric_factors, compute_normals, AbstractElemShape, Line, Quad, Tri, Tet, Hex, Pyr
    using Random: rand, shuffle
    using LinearAlgebra: inv, det, transpose, diagm

    export GeometricFactors, uniform_periodic_mesh, warp_mesh, cartesian_mesh, Uniform, ZigZag

    abstract type AbstractMeshGenStrategy end
    struct Uniform <: AbstractMeshGenStrategy end
    struct ZigZag <: AbstractMeshGenStrategy end

    struct GeometricFactors{d}
        # first dimension is node index, second is element
        J_q::Matrix{Float64}

        # first dimension is node index, second and third are matrix indices mn,
        # fourth is element
        Λ_q::Array{Float64,4}

        # d-tuple of matrices, where first is element index,
        nJf::NTuple{d, Matrix{Float64}}
    end

    function warp_mesh(mesh::MeshData{2}, reference_element::RefElemData{2}, 
        factor::Float64=0.2)
        @unpack x, y = mesh

        x = x .+ factor*sin.(π*x).*sin.(π*y)
        y = y .+ factor*exp.(1.0.-y).*sin.(π*x).*sin.(π*y)
        
        return MeshData(reference_element, mesh, x, y)
    end
  
    function uniform_periodic_mesh(reference_element::RefElemData{1}, 
        limits::NTuple{2,Float64}, M::Int)

        VX, EtoV = uniform_mesh(reference_element.elementType, M)
        mesh =  MeshData(limits[1] .+ 0.5*(limits[2]-limits[1])*(VX[1] .+ 1.0), EtoV,reference_element)

        return make_periodic(mesh)
    end

    function uniform_periodic_mesh(
        reference_element::RefElemData{d}, 
        limits::NTuple{d,NTuple{2,Float64}}, 
        M::NTuple{d,Int};
        random_rotate::Bool=false, 
        strategy::AbstractMeshGenStrategy=ZigZag()) where {d}

        VXY, EtoV = cartesian_mesh(reference_element.elementType, 
            M, strategy)
        N_el = size(EtoV,1)
        
        if random_rotate
            for k in 1:N_el
                len = size(EtoV,2)
                step = rand(0:len-1)
                row = EtoV[k,:]
                EtoV[k,:] = vcat(row[end-step+1:end], row[1:end-step])
            end
        end

        return make_periodic(MeshData([limits[m][1] .+ 
            0.5*(limits[m][2]-limits[m][1])*(VXY[m] .+ 1.0) for m in 1:d]..., EtoV, reference_element))
    end

    function cartesian_mesh(element_type::AbstractElemShape, 
        M::NTuple{d,Int}, ::Uniform) where {d}
        return uniform_mesh(element_type, [M[m] for m in 1:d]...)
    end

    function cartesian_mesh(element_type::Union{Quad,Hex,Tet},
        M::NTuple{d,Int}, ::ZigZag) where {d}
        # zigzag not implemented for quad/hex/tet etc.
        return uniform_mesh(element_type, [M[m] for m in 1:d]...)
    end

    function cartesian_mesh(::Tri,  M::NTuple{2,Int}, ::ZigZag)
        if !(iseven(M[1]) && iseven(M[2]))
            error("ERROR: ZigZag mesh must have even number of elements in each direction")
        end

        (VX,VY), _ = uniform_mesh(Quad(), M[1], M[2])
        EtoV = Matrix{Int64}(undef, 0, 3)
        for i in 1:2:(M[1]-1)
            for j in 1:2:(M[2]-1)
                bot_left = (j-1)*(M[2]+1) + i
                bot_mid = j*(M[2]+1) + i
                bot_right = (j+1)*(M[2]+1) + i
                EtoV =vcat(EtoV,[
                    bot_left+1 bot_left bot_mid ;
                    bot_mid bot_mid+1 bot_left+1 ;
                     bot_mid bot_right bot_right+1;
                    bot_right+1 bot_mid+1 bot_mid ;
                    bot_left+1 bot_mid+1 bot_mid+2 ;
                    bot_mid+2 bot_left+2 bot_left+1 ;
                    bot_mid+2 bot_mid+1 bot_right+1;
                    bot_right+1 bot_right+2 bot_mid+2][:,end:-1:1])
            end
        end
        return (VX, VY), EtoV
    end
    
    function GeometricFactors(mesh::MeshData{d}, 
        reference_element::RefElemData{d}) where {d}

        # note, here we assume that mesh is same N_q, N_f every element
        N_q = size(mesh.xyzq[1],1)
        N_f = size(mesh.xyzf[1],1)
        N_el = size(mesh.xyzq[1],2)

        J_q = Matrix{Float64}(undef, N_q, N_el)
        nJf = Tuple(Matrix{Float64}(undef, N_f, N_el) for m in 1:d)

        # first dimension is node index, 
        # second and third are matrix indices mn,
        # fourth is element index.
        dxdr_q = Array{Float64, 4}(undef, N_q, d, d, N_el)
        Λ_q = Array{Float64, 4}(undef, N_q, d, d, N_el)
        dxdr_f = Array{Float64, 4}(undef, N_f, d, d, N_el)        

        for k in 1:N_el
            for m in 1:d, n in 1:d
                # evaluate metric at mapping nodes
                dxdr = reference_element.Drst[n]*mesh.xyz[m][:,k]

                # use mapping basis to interpolate to quadrature nodes
                dxdr_q[:,m,n,k] = reference_element.Vq*dxdr
                dxdr_f[:,m,n,k] = reference_element.Vf*dxdr
            end

            # loops over slower indices
            @inbounds for i in 1:N_q
                J_q[i,k] = det(dxdr_q[i,:,:,k])
                Λ_q[i,:,:,k] = J_q[i,k]*inv(dxdr_q[i,:,:,k])
            end
        
            # get scaled normal vectors - this includes scaling for ref. quadrature weights on long side of right-angled triangle.
            # don't need actual facet Jacobian for now, probably will at some point.
            for i in 1:N_f
                Jdrdx_f = det(dxdr_f[i,:,:,k]) *
                    inv(dxdr_f[i,:,:,k])
                for m in 1:d
                    nJf[m][i,k] = sum(
                        Jdrdx_f[n,m]*reference_element.nrstJ[n][i] 
                            for n in 1:d)
                end
            end
        end
        return GeometricFactors{d}(J_q, Λ_q, nJf)
    end
end