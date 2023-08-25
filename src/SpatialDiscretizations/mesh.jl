abstract type AbstractMeshGenStrategy end
struct Uniform <: AbstractMeshGenStrategy end
struct ZigZag <: AbstractMeshGenStrategy end

abstract type AbstractMeshWarping{d} end
struct DelReyWarping{d} <: AbstractMeshWarping{d}
    factor::Float64
    L::NTuple{d,Float64}
end
struct ChanWarping{d} <: AbstractMeshWarping{d} 
    # Chan, Del Rey Fernandez, Carpenter (2018)
    # Rueda-Ramirez, Hindenlang, Chan, Gassner (2023)
    factor::Float64
    L::NTuple{d,Float64}
end

struct UniformWarping{d}  <: AbstractMeshWarping{d}
    # Chan, Del Rey Fernandez, Carpenter (2019)
    # Shadpey and Zingg (2020)
    factor::Float64
    L::NTuple{d,Float64}
end

function warp_mesh(mesh::MeshData{d}, 
    reference_element::RefElemData{d}, 
    factor::Float64=0.2, L::Float64=1.0) where {d}
    return warp_mesh(mesh,reference_element,
        DelReyWarping(factor, Tuple(L for m in 1:d)))
end

function warp_mesh(mesh::MeshData{2}, 
    reference_element::RefElemData{2}, 
    mesh_warping::DelReyWarping{2})
    
    (; x, y) = mesh
    (; factor, L) = mesh_warping

    x_new = x .+ L[1]*factor*sin.(π*x./L[1]).*sin.(π*y/L[2])
    y_new = y .+ L[2]*factor*exp.(1.0.-y/L[2]).*sin.(π*x/L[1]).*
        sin.(π*y/L[2])
    return MeshData(reference_element, mesh, x_new, y_new)
end

function warp_mesh(mesh::MeshData{2}, 
    reference_element::RefElemData{2}, 
    mesh_warping::ChanWarping{2})
    
    (; x, y) = mesh
    (; factor, L) = mesh_warping
    
    x_new = x .+ L[1]*factor*cos.(π/L[1]*(x.-0.5*L[1])) .*
        cos.(3π/L[2]*(y.-0.5*L[2]))
    y_new = y .+ L[2]*factor*sin.(4π/L[1]*(x_new.-0.5*L[1])) .*
        cos.(π/L[2]*(y.-0.5*L[2]))

    return MeshData(reference_element, mesh, x_new, y_new)
end

function warp_mesh(mesh::MeshData{3}, 
    reference_element::RefElemData{3}, 
    mesh_warping::ChanWarping{3})
    
    (; x, y, z) = mesh
    (; factor, L) = mesh_warping
    
    y_new = y .+ L[2]*factor*cos.(3π/L[1]*(x.-0.5*L[1])) .*
        cos.(π/L[2]*(y.-0.5*L[2])) .* cos.(π/L[3]*(z.-0.5*L[3]))
    x_new = x .+ L[1]*factor*cos.(π/L[1]*(x.-0.5*L[1])) .*
    sin.(4π/L[2]*(y_new.-0.5*L[2])) .* cos.(π/L[3]*(z.-0.5*L[3]))
    z_new = z .+ L[3]*factor*cos.(π/L[1]*(x_new.-0.5*L[1])) .*
        cos.(2π/L[2]*(y_new.-0.5*L[2])) .* cos.(π/L[3]*(z.-0.5*L[3]))

    return MeshData(reference_element, mesh, x_new, y_new, z_new)
end

function warp_mesh(mesh::MeshData{2}, 
    reference_element::RefElemData{2}, 
    mesh_warping::UniformWarping{2})
    
    (; x, y) = mesh
    (; factor, L) = mesh_warping

    eps = factor * sin.(2π*(x.-L[1]/2)/L[1]) .* sin.(2π*(y.-L[2]/2)/L[2])
    x_new = x .+ L[1]*eps
    y_new = y .+ L[2]*eps

    return MeshData(reference_element, mesh, x_new, y_new)
end

function warp_mesh(mesh::MeshData{3}, 
    reference_element::RefElemData{3}, 
    mesh_warping::UniformWarping{3})
    
    (; x, y, z) = mesh
    (; factor, L) = mesh_warping

    eps = factor * sin.(2π*(x.-L[1]/2)/L[1]) .* sin.(2π*(y.-L[2]/2)/L[2]) .*
        sin.(2π*(z.-L[3]/2)/L[3])
    x_new = x .+ L[1]*eps
    y_new = y .+ L[2]*eps
    z_new = z .+ L[3]*eps

    return MeshData(reference_element, mesh, x_new, y_new, z_new)
end

function warp_mesh(mesh::MeshData{3}, reference_element::RefElemData{3}, 
    mesh_warping::DelReyWarping{3})

    (; x, y, z) = mesh
    (; factor, L) = mesh_warping

    x_new = x .+ L[1]*factor*sin.(π*x/L[1]).*sin.(π*y/L[2])
    y_new = y .+ L[2]*
        factor*exp.((1.0.-y)/L[2]).*sin.(π*x/L[1]).*sin.(π*y/L[2])
    z_new = z .+ 0.25*L[3]*
        factor*(sin.(2π*x/L[1]).*sin.(2π*y/L[2])).*sin.(2π*z/L[3])
    return MeshData(reference_element, mesh, x_new, y_new, z_new)
end

function uniform_periodic_mesh(reference_element::RefElemData{1}, 
    limits::NTuple{2,Float64}, M::Int)

    VX, EtoV = uniform_mesh(reference_element.element_type, M)
    mesh = MeshData(limits[1] .+ 0.5*(limits[2]-limits[1])*(VX[1] .+ 1.0),
        EtoV,reference_element)

    return make_periodic(mesh)
end

function uniform_periodic_mesh(
    reference_element::RefElemData{d}, 
    limits::NTuple{d,NTuple{2,Float64}}, 
    M::NTuple{d,Int};
    random_rotate::Bool=false, 
    collapsed_orientation::Bool=false,
    strategy::AbstractMeshGenStrategy=Uniform()) where {d}

    VXY, EtoV = cartesian_mesh(reference_element.element_type, 
        M, strategy)
    N_e = size(EtoV,1)
    
    if random_rotate
        for k in 1:N_e
            len = size(EtoV,2)
            step = rand(0:len-1)
            row = EtoV[k,:]
            EtoV[k,:] = vcat(row[end-step+1:end], row[1:end-step])
        end
    elseif reference_element.element_type isa Tet && collapsed_orientation

        # Second algorithm from Warburton's PhD thesis
        EtoV_new = Vector{Float64}(undef,4)

        for k in 1:N_e
            EtoV_new = sort(EtoV[k,:], rev=true)
            X = hcat([[VXY[1][EtoV_new[m]] - VXY[1][EtoV_new[1]];
                VXY[2][EtoV_new[m]] - VXY[2][EtoV_new[1]];
                VXY[3][EtoV_new[m]] - VXY[3][EtoV_new[1]]] for m in 2:4]...)
            
            if det(X) < 0 
                EtoV[k,:] = [EtoV_new[2]; 
                            EtoV_new[1]; EtoV_new[3]; EtoV_new[4]]
            else 
                EtoV[k,:] = EtoV_new 
            end
        end
    end

    return make_periodic(MeshData([limits[m][1] .+ 
        0.5*(limits[m][2]-limits[m][1])*(VXY[m] .+ 1.0) for m in 1:d]...,
        EtoV, reference_element))
end

function cartesian_mesh(element_type::AbstractElemShape, 
    M::NTuple{d,Int}, ::AbstractMeshGenStrategy) where {d}
    return uniform_mesh(element_type, [M[m] for m in 1:d]...)
end

function cartesian_mesh(::Tri,  M::NTuple{2,Int}, ::ZigZag)

    if !(iseven(M[1]) && iseven(M[2]))
        @error "ERROR: ZigZag mesh must have even number of elements in each direction"
    end
    (VX,VY), _ = uniform_mesh(Quad(), M[1], M[2])
    EtoV = Matrix{Int64}(undef, 0, 3)

    for i in 1:2:(M[1]-1), j in 1:2:(M[2]-1)
        bot_left = (j-1)*(M[2]+1) + i
        bot_mid = j*(M[2]+1) + i
        bot_right = (j+1)*(M[2]+1) + i
        EtoV =vcat(EtoV,[
            bot_mid bot_left bot_left+1 ;
            bot_left+1 bot_mid+1 bot_mid ;
            bot_right+1 bot_right bot_mid ;
            bot_mid bot_mid+1 bot_right+1 ;
            bot_mid+2 bot_mid+1 bot_left+1 ;
            bot_left+1 bot_left+2 bot_mid+2 ;
            bot_right+1 bot_mid+1 bot_mid+2  ;
            bot_mid+2 bot_right+2 bot_right+1])
    end
    return (VX, VY), EtoV
end


function metrics(dxdr::SMatrix{1,1})
    J = dxdr[1,1]
    Λ = SMatrix{1,1}(1.0) 
    return J, Λ
end

function metrics(dxdr::SMatrix{2,2})
    J = dxdr[1,1]*dxdr[2,2] - dxdr[1,2]*dxdr[2,1]
    Λ = SMatrix{2,2}([dxdr[2,2] -dxdr[1,2]; -dxdr[2,1] dxdr[1,1]]) 
    return J, Λ
end

function metrics(dxdr::SMatrix{3,3})
    J = det(dxdr)
    Λ = J*inv(dxdr)
    return J, Λ
end

function GeometricFactors(mesh::MeshData{d}, 
    reference_element::RefElemData{d}, 
    metric_type::ExactMetrics=ExactMetrics()) where {d}

    (; nrstJ) = reference_element

    # note, here we assume that mesh is same N_q, N_f every element
    N_q = size(mesh.xyzq[1],1)
    N_f = size(mesh.xyzf[1],1)
    N_e = size(mesh.xyzq[1],2)

    # here we assume same number of nodes per face
    N_fac = num_faces(reference_element.element_type)
    nodes_per_face = N_f ÷ N_fac

    J_q = Matrix{Float64}(undef, N_q, N_e)
    J_f = Matrix{Float64}(undef, N_f, N_e)

    nJf = Array{Float64, 3}(undef, d, N_f, N_e)
    nJq = Array{Float64, 4}(undef, d, N_fac, N_q, N_e)

    # first dimension is node index, 
    # second and third are matrix indices mn,
    # fourth is element index.
    dxdr_q = Array{Float64, 4}(undef, N_q, d, d, N_e)
    Λ_q = Array{Float64, 4}(undef, N_q, d, d, N_e)
    dxdr_f = Array{Float64, 4}(undef, N_f, d, d, N_e)   

    for k in 1:N_e
        @inbounds for m in 1:d, n in 1:d
            # evaluate metric at mapping nodes
            dxdr = reference_element.Drst[n]*mesh.xyz[m][:,k]

            # use mapping basis to interpolate to quadrature nodes (exact)
            dxdr_q[:,m,n,k] = reference_element.Vq*dxdr
            dxdr_f[:,m,n,k] = reference_element.Vf*dxdr
        end

        # loops over slower indices
        @inbounds for i in 1:N_q
            J_q[i,k], Λ_q[i,:,:,k] = metrics(SMatrix{d,d}(dxdr_q[i,:,:,k]))
            for f in 1:N_fac
                n_ref = Tuple(nrstJ[m][nodes_per_face*(f-1)+1] for m in 1:d)
                for n in 1:d
                    nJq[n,f,i,k] = sum(Λ_q[i,m,n,k]*n_ref[m] for m in 1:d)
                end
            end
        end

        # get scaled normal vectors - this includes scaling for ref. quadrature weights on long side of right-angled triangle.
        @inbounds for i in 1:N_f
            _, Jdrdx_f = metrics(SMatrix{d,d}(dxdr_f[i,:,:,k]))
            @inbounds for m in 1:d
                nJf[m,i,k] = sum(Jdrdx_f[n,m]*nrstJ[n][i] for n in 1:d)
            end
            J_f[i,k] = sqrt(sum(nJf[m,i,k]^2 for m in 1:d))
        end
    end
    return GeometricFactors(J_q, Λ_q, J_f, nJf, nJq)
end


function GeometricFactors(mesh::MeshData{2}, 
    reference_element::RefElemData{2}, 
    ::ChanWilcoxMetrics)

    (; x, y) = mesh
    (; nrstJ, Dr, Ds, Vq, Vf) = reference_element


    # note, here we assume that mesh is same N_q, N_f every element
    N_q = size(mesh.xyzq[1],1)
    N_f = size(mesh.xyzf[1],1)
    N_e = size(mesh.xyzq[1],2)

    # here we assume same number of nodes per face
    N_fac = num_faces(reference_element.element_type)
    nodes_per_face = N_f ÷ N_fac

    J_q = Matrix{Float64}(undef, N_q, N_e)
    J_f = Matrix{Float64}(undef, N_f, N_e)

    nJf = Array{Float64, 3}(undef, 2, N_f, N_e)
    nJq = Array{Float64, 4}(undef, 2, N_fac, N_q, N_e)

    # first dimension is node index, 
    # second and third are matrix indices mn,
    # fourth is element index.
    Λ_q = Array{Float64, 4}(undef, N_q, 2, 2, N_e)

    @inbounds @views for k in 1:N_e

        rxJ, sxJ, ryJ, syJ, J = geometric_factors(x[:,k], y[:,k], Dr, Ds)

        mul!(Λ_q[:,1,1,k], Vq, rxJ)
        mul!(Λ_q[:,2,1,k], Vq, sxJ)
        mul!(Λ_q[:,1,2,k], Vq, ryJ)
        mul!(Λ_q[:,2,2,k], Vq, syJ)
        mul!(J_q[:,k], Vq, J)
        
        Λ_f = Array{Float64, 3}(undef, N_f, 2, 2)
        mul!(Λ_f[:,1,1], Vf, rxJ)
        mul!(Λ_f[:,2,1], Vf, sxJ)
        mul!(Λ_f[:,1,2], Vf, ryJ)
        mul!(Λ_f[:,2,2], Vf, syJ)

        # loops over slower indices
        @inbounds for i in 1:N_q
            for f in 1:N_fac
                n_ref = Tuple(nrstJ[m][nodes_per_face*(f-1)+1] for m in 1:2)
                for n in 1:2
                    nJq[n,f,i,k] = sum(Λ_q[i,m,n,k]*n_ref[m] for m in 1:2)
                end
            end
        end

        # get scaled normal vectors - this includes scaling for ref. quadrature weights on long side of right-angled triangle.
        @inbounds for i in 1:N_f
            @inbounds for m in 1:2
                nJf[m,i,k] = sum(Λ_f[i,n,m]*nrstJ[n][i] for n in 1:2)
            end
            J_f[i,k] = sqrt(sum(nJf[m,i,k]^2 for m in 1:2))
        end
    end
    return GeometricFactors(J_q, Λ_q, J_f, nJf, nJq)
end

function uniform_periodic_mesh(
    reference_approximation::ReferenceApproximation{3, Tet, <:Union{NodalTensor,ModalTensor}},
    limits::NTuple{3,NTuple{2,Float64}}, 
    M::NTuple{3,Int};
    random_rotate::Bool=false, 
    strategy::AbstractMeshGenStrategy=Uniform())

    return uniform_periodic_mesh(
        reference_approximation.reference_element, limits, M,
        random_rotate=random_rotate,
        collapsed_orientation=true,
        strategy=strategy)
end

function uniform_periodic_mesh(
    reference_approximation::ReferenceApproximation{d, <:AbstractElemShape, <:AbstractApproximationType}, 
    limits::NTuple{d,NTuple{2,Float64}}, 
    M::NTuple{d,Int};
    random_rotate::Bool=false, 
    strategy::AbstractMeshGenStrategy=Uniform()) where {d}

    return uniform_periodic_mesh(
        reference_approximation.reference_element, limits, M,
        random_rotate=random_rotate,
        strategy=strategy)
end

@inline uniform_periodic_mesh(
    reference_approximation::ReferenceApproximation{1, <:AbstractElemShape, <:AbstractApproximationType}, 
    limits::NTuple{2,Float64}, 
    M::Int) = uniform_periodic_mesh(reference_approximation.reference_element,
        limits, M)

function warp_mesh(mesh::MeshData{d}, reference_approximation::ReferenceApproximation{d,<:AbstractElemShape, <:AbstractApproximationType},
    factor::Float64=0.2, L::Float64=1.0) where {d}
    return warp_mesh(mesh,reference_approximation.reference_element,
        DelReyWarping(factor, Tuple(L for m in 1:d)))
end

function warp_mesh(mesh::MeshData{d}, reference_approximation::ReferenceApproximation{d,<:AbstractElemShape, <:AbstractApproximationType}, mesh_warping::AbstractMeshWarping{d}) where {d}
    return warp_mesh(mesh, reference_approximation.reference_element, mesh_warping)
end