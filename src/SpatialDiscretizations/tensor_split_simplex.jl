using LinearAlgebra
using SparseArrays
"""
### SummationByParts.tensor_lgl_quad_nodes

Computes tensor-product nodes on a quadrilateral 
**Inputs**
* `p`: degree of the operator 
* `opertype`: the type of 1d operator used to construct the SST-SBP method (lgl or csbp)
* `n1d`: number of nodes in the 1D operator 

**Outputs** 
* `xy`: node coordinates
* `w`: weights of the 1D operator 
"""
function tensor_quad_nodes(p::Int;opertype::String="lgl", n1d::Int=-1)
    if opertype=="lgl"
        z,w = quadrature(Line(), GaussLobattoQuadrature(p,0,0))

        Q = length(z)
        x = repeat(z, Q)                 # x-coordinates: repeated for each row
        y = repeat(z, inner=Q)           # y-coordinates: tiled for each column
        xy = [x'; y']                   # 2 × n^2 matrix: each column is a 2D node
        return xy, w
    else 
        error("Operator not implemented. Must be 'lgl.")
    end
end
"""
### SummationByParts.square_quad_map

Maps points in the standard square domain, [-1,1]^2, to any quadrilateral 

**Inputs** 
* `xp`: points in the standard square domain 
* `quad_vert`: coordinates of the vertices of the general quadrilateral element

**Outputs** 
* `x`: the mapped points in the quadrilateral element 
"""
function square_quad_map(xp::Array{T},quad_vert::Array{T}) where T
    xi = xp[1]
    eta = xp[2]
    psi = []
    push!(psi, 1/4*(1-xi)*(1-eta))
    push!(psi, 1/4*(1+xi)*(1-eta))
    push!(psi, 1/4*(1-xi)*(1+eta))
    push!(psi, 1/4*(1+xi)*(1+eta))
    
    x = zeros(2,1)
    for i=1:2
        for j=1:4
            x[i] += quad_vert[j,i]*psi[j]
        end
    end
    return x 
end
"""
### SummationByParts.square_to_tri_map

Maps points from the standard square domain, [-1,1]^2, to the quadrilaterals
generated in the split-triangle

**Inputs** 
* `xi`: the point in the stadard square element 

**Outputs**
* `x`: the correspoinding points in the 3 quadrilaterals in the split-triangle 
"""
function square_to_tri_map(xi::Array{T}) where T
    quad_vert = get_quad_vert()
    n = size(xi,2)
    x = zeros(2,3*n)
    for i=0:2 
        for j=1:n
            xp = square_quad_map(xi[:,j], quad_vert[i+1])
            x[:,i*n+j] = xp 
        end
    end
    return x
end

"""
### SummationByParts.get_quad_vert

Returns the vertices of the 3 quadrilaterals obtained by splitting the 
standard triangle with vertices [-1 -1; 1 -1; -1 1]

**Outputs** 
* `quad_vert`: vertices of the 3 quadrilaterals 
"""
function get_quad_vert()
    quad_vert = [[-1 -1; 0 -1; -1 0; -1/3 -1/3], 
                 [0 -1; 1 -1; -1/3 -1/3; 0 0], 
                 [-1 0; -1/3 -1/3; -1 1; 0 0]]
    return quad_vert
end

"""
### SummationByParts.perp_to_equi_tri_map 

Maps nodes from the standard right triangle to an equilateral triangle 

**Inputs** 
* `x`: matrix containing coordinates of nodes in the right triange 

**Outputs** 
* `xequi`: matrix containing coordinates of nodes in the equilateral triangle 
"""
function perp_to_equi_tri_map(x::Array{T}) where T
    vtx = T[-1 -1/sqrt(3); 1 -1/sqrt(3); 0 2/sqrt(3)]
    xequi = zeros(size(x))
    for i = 1:2
        xequi[i,:] .= -0.5 .* (x[1,:].+x[2,:])*vtx[1,i] .+
                    0.5 .*(x[1,:].+1.0)*vtx[2,i] .+
                    0.5 .*(x[2,:].+1.0)*vtx[3,i]
    end
    return xequi
end

"""
### SummationByParts.metric_tri

Computes the metric terms for a point mapped from the standard square domain 
to a quadrilateral in the split-triangle 

**Inputs** 
* `xp`: the coordinates of the point in the standard square domain 
* `quad_vert`: A matrix containing the vertices of the quadrilateral in the split-triangle 

**Outputs** 
* `dxi`: A column vector containing the metric terms [dx/dξ,dx/dη,dy/dξ,dy/dη] 
* `dx`: A column vector containing the metric terms [dξ/dx,dξ/dy,dη/dx,dη/dy]
* `Jac`: The metric Jacobian 
"""
function metric_tri!(xp::Array{T},quad_vert::Array{T},dxi::SubArray{T},dx::SubArray{T},Jac::SubArray{T}) where T
    ξ = xp[1]
    η = xp[2]

    ∂Ψ∂ξ = []
    ∂Ψ∂η = []
    push!(∂Ψ∂ξ, -1/4*(1-η))
    push!(∂Ψ∂ξ, 1/4*(1-η))
    push!(∂Ψ∂ξ, -1/4*(1+η))
    push!(∂Ψ∂ξ, 1/4*(1+η))

    push!(∂Ψ∂η, -1/4*(1-ξ))
    push!(∂Ψ∂η, -1/4*(1+ξ))
    push!(∂Ψ∂η, 1/4*(1-ξ))
    push!(∂Ψ∂η, 1/4*(1+ξ))

    ∂x∂ξ = 0.0
    ∂x∂η = 0.0
    ∂y∂ξ = 0.0
    ∂y∂η = 0.0

    for j=1:4
        ∂x∂ξ += quad_vert[j,1]*∂Ψ∂ξ[j]
        ∂x∂η += quad_vert[j,1]*∂Ψ∂η[j]
        ∂y∂ξ += quad_vert[j,2]*∂Ψ∂ξ[j]
        ∂y∂η += quad_vert[j,2]*∂Ψ∂η[j]
    end

    J = ∂x∂ξ*∂y∂η - ∂x∂η*∂y∂ξ
    ∂ξ∂x = ∂y∂η/J 
    ∂ξ∂y = -∂x∂η/J 
    ∂η∂x = -∂y∂ξ/J 
    ∂η∂y = ∂x∂ξ/J 

    Jac[1,1]=J
    dxi[1,1]=∂x∂ξ
    dxi[2,1]=∂x∂η
    dxi[3,1]=∂y∂ξ
    dxi[4,1]=∂y∂η

    dx[1,1]=∂ξ∂x
    dx[2,1]=∂ξ∂y
    dx[3,1]=∂η∂x
    dx[4,1]=∂η∂y
end

"""
### SummationByParts.tensor_operators

Computes tensor-product operators using the 1D LGL or CSBP operators

**Inputs** 
* `p`: The polynomial degree 
* `dim`: The spatial dimension 
* `opertype`: the operator type, either "lgl" or "csbp" (optional)
* `n1d`: number of nodes in the 1D operator 

**Outputs** 
* `H`: The tensor-product norm matrix 
* `Q`: The tensor-product Q matrix 
* `D`: The tensor-product D matrix 
* `E`: The tensor-product boundary integration operator
* `R`: The extrapolation operator 
"""
function tensor_operators(p::Int, dim::Int; opertype::String="lgl", n1d::Int=-1, T=Float64)
    if opertype=="lgl"
        LGL_quadrature = LGLQuadrature(p)
        _,w = quadrature(Line(),LGL_quadrature)
        _, _, _, D1, _, _, _ = operators_1d(LGL_quadrature)
        D1 = Matrix(D1)
        H1 = Diagonal(w)
        Q1 = H1*D1
        E1 = Q1 + Q1'
        n = length(w)
    else 
        error("Operator not implemented. Must be 'lgl'.")
    end
    In = I(n)
    tR = zeros(T,(n,1)); tR[end] = 1.0
    tL = zeros(T,(n,1)); tL[1] = 1.0

    H = zeros(T, (n^dim,n^dim))
    Q = zeros(T, (n^dim,n^dim,dim))
    D = zeros(T, (n^dim,n^dim,dim))
    E = zeros(T, (n^dim,n^dim,dim))
    R = zeros(T, (n^(dim-1),n^dim,2*dim))

    if dim==2
        H[:,:] = kron(H1, H1)
        Q[:,:,1] = kron(H1, Q1)
        Q[:,:,2] = kron(Q1, H1)
        D[:,:,1] = kron(In, D1)
        D[:,:,2] = kron(D1, In)
        E[:,:,1] = kron(In, E1)
        E[:,:,2] = kron(E1, In)
        R[:,:,1] = kron(In, tL') #left facet (x=-1)
        R[:,:,2] = kron(In, tR') #right facet (x=1)
        R[:,:,3] = kron(tL', In) #bottom facet (y=-1)
        R[:,:,4] = kron(tR', In) #top facet (y=1)
    elseif dim==3 
        H[:,:] = kron(H1,kron(H1, H1))
        Q[:,:,1] = kron(H1, kron(H1, Q1))
        Q[:,:,2] = kron(H1, kron(Q1, H1))
        Q[:,:,3] = kron(Q1, kron(H1, H1))
        D[:,:,1] = kron(In, kron(In, D1))
        D[:,:,2] = kron(In, kron(D1, In))
        D[:,:,3] = kron(D1, kron(In, In))
        E[:,:,1] = kron(In, kron(In, E1))
        E[:,:,2] = kron(In, kron(E1, In))
        E[:,:,3] = kron(E1, kron(In, In))
        R[:,:,1] = kron(In,kron(In, tL')) #left facet (x=-1)
        R[:,:,2] = kron(In,kron(In, tR')) #right facet (x=1)
        R[:,:,3] = kron(In,kron(tL', In)) #back facet (y=-1)
        R[:,:,4] = kron(In,kron(tR', In)) #front facet (y=1)
        R[:,:,5] = kron(tL',kron(In, In)) #bottom facet (z=-1) 
        R[:,:,6] = kron(tR',kron(In, In)) #top facet (z=1)
    end
    return H, Q, D, E, R
end

"""
### SummationByParts.normals_square

Returns the normals on the standard square domain [-1,1]^2 

**Inputs** 
* `nf`: the number of facet nodes 

**Outputs** 
* `N`: the normals at each facet node 
"""
function normals_square(nf::Int;T=Float64)
    dim=2
    N = zeros(T, (dim,nf,4)) # normal vector for each facet

    N[1,:,1] .= -1.0
    N[1,:,2] .= 1.0
    N[2,:,3] .= -1.0
    N[2,:,4] .= 1.0

    return N
end

"""
### SummationByParts.facet_nodes_square

Returns the global node index of each facet node a square element

**Inputs** 
* `n`: The number of nodes in the element 

**Outputs**
* `facet_node_idx`: The global node index for the facet nodes  
"""
function facet_nodes_square(n::Int) 
    nf = convert(Int, sqrt(n))
    facet_node_idx = zeros(Int, (4,nf))

    facet_node_idx[1,:] = 1:nf:n
    facet_node_idx[2,:] = nf:nf:n
    facet_node_idx[3,:] = 1:nf 
    facet_node_idx[4,:] = (n+1-nf):n

    return facet_node_idx
end

"""
### SummationByParts.map_tensor_operators_to_tri

Maps the tensor-product operator to the quadrilateral elements in the split-triangle
**Inputs**
* `p`: degree of the operator 
* `opertype`: the type of 1d operator used to construct the SST-SBP method (lgl or csbp)
* `n1d`: number of nodes in the 1D operator 

**Outputs** 
* `Hs`: A list of the norm matrices
* `Qs`: A list of the Q matrices
* `Ds`: A list of the D matrices 
* `Es`: A list of the boundary integration operators
* `Ns`: A list of the normal matrices 
"""
function map_tensor_operators_to_tri(p::Int; opertype::String="lgl", n1d::Int=-1, T=Float64)
    dim = 2
    xs, B = tensor_quad_nodes(p,opertype=opertype,n1d=n1d) #nodes on square
    n = size(xs,2)
    nf = convert(Int, sqrt(n))
    
    quad_vert = get_quad_vert()
    Nhat = normals_square(convert(Int, sqrt(n)))
    facet_node_idx = facet_nodes_square(n)
    
    dxis = []
    dxs = []
    Js = []
    Ns = []
    for i = 1:3
        dxi = zeros(4,n)
        dx = zeros(4,n)
        J = zeros(1,n)
        for j=1:n
            metric_tri!(xs[:,j],quad_vert[i],view(dxi,:,j),view(dx,:,j),view(J,:,j))
        end
        push!(dxis, dxi)
        push!(dxs, dx)
        push!(Js, J)

        N = zeros(T,(dim,nf,4))
        for k=1:4
            N[1,:,k] = J[facet_node_idx[k,:]].*(dx[1,facet_node_idx[k,:]].*Nhat[1,:,k] .+ dx[3,facet_node_idx[k,:]].*Nhat[2,:,k])
            N[2,:,k] = J[facet_node_idx[k,:]].*(dx[2,facet_node_idx[k,:]].*Nhat[1,:,k] .+ dx[4,facet_node_idx[k,:]].*Nhat[2,:,k])
        end
        push!(Ns, N)
    end

    Hhat, Qhat, Dhat, Ehat, Rhat = tensor_operators(p, dim, opertype=opertype, n1d=n1d, T=T)
    Es = []
    for k=1:dim+1
        E = zeros(T, (n,n,dim))
        for i=1:dim
            for j=1:4
                E[:,:,i] += Rhat[:,:,j]'*diagm(Ns[k][i,:,j].*B)*Rhat[:,:,j]
            end
        end
        push!(Es,E)
    end

    Hs = []
    Qs = []
    Ds = []
    Ss = []
    for i=1:dim+1
        S = zeros(T, (n,n,dim))
        Q = zeros(T, (n,n,dim))
        E = Es[i]
        D = zeros(T, (n,n,dim))
        H = diagm(vec(Js[i]))*Hhat 
        push!(Hs, H)
        S[:,:,1] = 0.5*(diagm(vec(Js[i]).*dxs[i][1,:]) * Qhat[:,:,1] + diagm(vec(Js[i]).*dxs[i][3,:]) * Qhat[:,:,2]) - 
                   0.5*(Qhat[:,:,1]' * diagm(vec(Js[i]).*dxs[i][1,:]) + Qhat[:,:,2]' * diagm(vec(Js[i]).*dxs[i][3,:]))
        S[:,:,2] = 0.5*(diagm(vec(Js[i]).*dxs[i][2,:]) * Qhat[:,:,1] + diagm(vec(Js[i]).*dxs[i][4,:]) * Qhat[:,:,2]) - 
                   0.5*(Qhat[:,:,1]' * diagm(vec(Js[i]).*dxs[i][2,:]) + Qhat[:,:,2]' * diagm(vec(Js[i]).*dxs[i][4,:]))
        push!(Ss,S)
        Q[:,:,1] = S[:,:,1] + 0.5.*E[:,:,1]
        Q[:,:,2] = S[:,:,2] + 0.5.*E[:,:,2]
        push!(Qs, Q)
        D[:,:,1] = inv(H)*Q[:,:,1]
        D[:,:,2] = inv(H)*Q[:,:,2]
        # D[:,:,1] = diagm(vec(dxs[i][1,:]))*Dhat[:,:,1] + diagm(vec(dxs[i][3,:]))*Dhat[:,:,2]
        # D[:,:,2] = diagm(vec(dxs[i][2,:]))*Dhat[:,:,1] + diagm(vec(dxs[i][4,:]))*Dhat[:,:,2]
        push!(Ds, D)
    end 

    return Hs,Qs,Ds,Es,Ns,Ss
end

"""
### SummationByParts.global_node_index_tri

Returns the local to global node index on the split-triangle element 

**Inputs**
* `p`: degree of the operator 
* `opertype`: the type of 1d operator used to construct the TSS-SBP operator (lgl or csbp)
* `n1d`: number of nodes in the 1D operator 

**Outputs** 
* `xg`: Coordinates of global nodes in the split-triangle
* `loc_glob_idx`: Local to global index matching 
"""
function global_node_index_tri(p::Int;opertype::String="lgl",n1d::Int=-1, T=Float64)
    xs,_ = tensor_quad_nodes(p,opertype=opertype,n1d=n1d)
    xt = square_to_tri_map(xs)
    n = size(xs,2)

    facet_node_idx = facet_nodes_square(n)
    xg = copy(xt)
    remove_idx = collect(Iterators.flatten([n.+facet_node_idx[1,:], (2*n).+facet_node_idx[2,:], (2*n).+facet_node_idx[3,:]]))
    xg = xg[:, filter(x -> !(x in remove_idx), 1:size(xg, 2))]
    
    loc_glob_idx = []
    for k = 1:3
        x = zeros(Int,(2,n))
        x[1,:] = 1:n
        for i=1:n
            xgidx = xg .- xt[:,(k-1)*n+i]
            col_norm = [norm(xgidx[:, j]) for j in 1:size(xg,2)]
            ig = argmin(col_norm)
            x[2,i] = ig
        end
        push!(loc_glob_idx, x)
    end
    return xg, loc_glob_idx
end

"""
### SummationByParts.construct_zmatrix

Constructs the Z matrix required to apply continuous Galerkin type patching 
as described by Hicken et. al. Multidimensional Summation-by-Parts Operators: General Theory and Application to Simplex Elements (2016)

**inputs** 
* `glob_idx`: A matrix containing the local and global indices of each node of a split-element
* `i`: Row index corresponding to the ith local node of a split-element
* `j`: Column index corresponding to the jth local index of a split-element
* `nglob`: Total number of global nodes 

**Outputs** 
* `Z`: The Z matrix used to patch the split-elements in continuous Galerkin fashion
"""
function construct_zmatrix(glob_idx::Array{T},i::Int,j::Int,nglob::Int) where T
    ihat = glob_idx[2,i]
    jhat = glob_idx[2,j]
    ei = sparse(I,nglob,nglob)[:, ihat]
    ej = sparse(I,nglob,nglob)[:, jhat]
    Z = ei*ej'
    return Z
end

"""
### SummationByParts.construct_split_operator_tri

Returns TSS-SBP operators on the reference triangle 

**Inputs**
* `p`: degree of the operator 
* `opertype`: the type of 1d operator used to construct the TSS-SBP operator (lgl or csbp)
* `n1d`: number of nodes in the 1D operator 

**Outputs**
* `H`: The TSS norm matrix 
* `Q`: The TSS Q matrix 
* `D`: The TSS D matrix 
* `E`: The TSS boundary integration operator
"""
function construct_split_operator_tri(p::Int; opertype::String="lgl", n1d::Int=-1, T=Float64)
    Hs,Qs,Ds,Es,_,Ss = map_tensor_operators_to_tri(p, opertype=opertype, n1d=n1d)
    xg, loc_glob_idx = global_node_index_tri(p, opertype=opertype, n1d=n1d)
    n = size(Hs[1],1)
    nglob = size(xg,2) 
    dim = 2

    H = spzeros(nglob,nglob)
    S = [spzeros(nglob,nglob),spzeros(nglob,nglob)]
    Q = [spzeros(nglob,nglob),spzeros(nglob,nglob)]
    D = [spzeros(nglob,nglob),spzeros(nglob,nglob)]
    E = [spzeros(nglob,nglob),spzeros(nglob,nglob)]
    for k=1:dim+1
        for i=1:n
            for j=1:n 
                glob_idx = loc_glob_idx[k]
                Z = construct_zmatrix(glob_idx,i,j,nglob)
                if Hs[k][i,j]!=0.0
                    H[:,:] += (Hs[k][i,j]*Z)
                end
                for id=1:dim
                    if Ss[k][i,j,id]!=0.0
                        S[id] += (Ss[k][i,j,id]*Z)
                    end
                    if Qs[k][i,j,id]!=0.0
                        Q[id] += (Qs[k][i,j,id]*Z)
                    end
                    if Es[k][i,j,id]!=0.0
                        E[id] += (Es[k][i,j,id]*Z)
                    end
                end
            end
        end
    end
    
    for id=1:dim 
        D[id] = inv(Matrix(H))*Q[id]
    end
    S = [Matrix(m) for m in S]
    Q = [Matrix(m) for m in Q]
    D = [Matrix(m) for m in D]
    E = [Matrix(m) for m in E]
    return Matrix(H),Q,D,E,S
end


"""
### SummationByParts.global_node_index_tri_facet

Returns the coordinates of the facet nodes and the 
local and global node index for the facet nodes on the triangle  

**Inputs**
* `p`: degree of the operator 
* `opertype`: the type of 1d operator used to construct the TSS-SBP operator (lgl or csbp)
* `n1d`: number of nodes in the 1D operator 

**Outputs** 
* `xf`: Coordinates of the facet nodes on the triangle 
* `loc_glob_facet_idx`: The local to global index mapping of the facet nodes
"""
function global_node_index_tri_facet(p::Int;opertype::String="lgl",n1d::Int=-1, T=Float64)
    dim = 2
    xs,_ = tensor_quad_nodes(p, opertype=opertype, n1d=n1d)
    xt = square_to_tri_map(xs)
    xg,_ = global_node_index_tri(p, opertype=opertype, n1d=n1d)
    n = size(xs,2)
    n1 = convert(Int, round(n^(1/dim)))
    nf = n1^(dim-1)

    facet_node_idx = facet_nodes_square(n)
    xf = zeros(T, (dim, dim*nf-1, dim+1))
    keep_idx = []
    push!(keep_idx, collect(Iterators.flatten([(n).+facet_node_idx[2,:], reverse((2*n).+facet_node_idx[4,1:end-1])])))
    push!(keep_idx, collect(Iterators.flatten([reverse((2*n).+facet_node_idx[1,:]), reverse(facet_node_idx[1,1:end-1])])))
    push!(keep_idx, collect(Iterators.flatten([facet_node_idx[3,:], (n).+facet_node_idx[3,2:end]])))
    for k=1:dim+1
        xf[:,:,k] = xt[:,keep_idx[k]]
    end
    xfall= zeros(T,(dim,nf,dim*(dim+1)))
    xfall[:,:,1] = xt[:,(n).+facet_node_idx[2,:]]
    xfall[:,:,2] = xt[:,((2*n).+facet_node_idx[4,:])]
    xfall[:,:,3] = xt[:,((2*n).+facet_node_idx[1,:])]
    xfall[:,:,4] = xt[:,(facet_node_idx[1,:])]
    xfall[:,:,5] = xt[:,facet_node_idx[3,:]]
    xfall[:,:,6] = xt[:,(n).+facet_node_idx[3,:]]

    loc_glob_facet_idx = []
    for k = 1:(dim*(dim+1))
        x = zeros(Int,(3,nf))
        x[1,:] = 1:nf
        for i=1:nf
            xfidx = xf[:,:,convert(Int,ceil(k/dim))] .- xfall[:,i,k] 
            col_norm = [norm(xfidx[:, j]) for j in 1:size(xf,2)]
            ig = argmin(col_norm)
            x[2,i] = ig
        end
        for i=1:nf
            xgidx = xg .- xfall[:,i,k]
            col_norm = [norm(xgidx[:, j]) for j in 1:size(xg,2)]
            ig = argmin(col_norm)
            x[3,i] = ig
        end
        push!(loc_glob_facet_idx, x)
    end
    return xf, loc_glob_facet_idx
end

"""
### SummationByParts.construct_split_facet_operator_tri

Constructs the TSS-SBP facet operators on the triangle

**Inputs**
* `p`: degree of the operator 
* `opertype`: the type of 1d operator used to construct the TSS-SBP operator (lgl or csbp)
* `n1d`: number of nodes in the 1D operator 

**Outputs** 
* `B`: The TSS facet quadrature weights 
* `N`: The TSS normal matrix 
* `R`: The TSS extrapolation matrix operator 
* `E`: The TSS boundary integration operator 
"""
function construct_split_facet_operator_tri(p::Int; opertype::String="lgl", n1d::Int=-1, T=Float64)
    dim = 2 
    # qf = 2*(p+1)-3 
    # cub_lgl, vtx_lgl = SummationByParts.Cubature.quadrature(qf, internal=false)
    # perm = sortperm(vec(SymCubatures.calcnodes(cub_lgl, vtx_lgl)))
    # Bhat = SymCubatures.calcweights(cub_lgl)[perm]
    _, Bhat = tensor_quad_nodes(p,opertype=opertype,n1d=n1d)

    _,_,_,_,Nhat,_ = map_tensor_operators_to_tri(p, opertype=opertype, n1d=n1d)
    nf = length(Bhat)
    n = (dim+1)*nf^dim - (dim+dim^(dim-2))*nf^(dim-1) + (dim-2)*((dim+1)*nf - 2) + 1
    # N_idx = [2 2; 3 1; 1 3] #first column contains element number, and second column contains facet number of the element

    xf, loc_glob_idx = global_node_index_tri_facet(p, opertype=opertype, n1d=n1d)
    nglob = size(xf,2) 
    B = zeros(T,(nglob,nglob,dim+1))
    for k=1:dim+1
        for i=1:dim
            glob_idx = loc_glob_idx[(k-1)*dim+i]
            for j=1:nf
                Z = construct_zmatrix(glob_idx,j,j,nglob)
                B[:,:,k] += (Bhat[j]*Z)
            end
        end
    end
    
    # N = ones(T, (dim, size(xf,2),dim+1))
    # for k=1:dim+1
    #     for i=1:dim 
    #         N[i,:,k] = Nhat[N_idx[k,1]][i,1,N_idx[k,2]] * N[i,:,k]
    #     end
    # end

    N_idx = [[2 2; 3 4],[3 1; 1 1],[1 3; 2 3]]
    N = zeros(T, (dim, nglob,dim+1))
    for id=1:dim
        for k=1:dim+1
            jj = dim*(k-1)
            idx_vec=(collect(Iterators.flatten([loc_glob_idx[jj+1][2,:],loc_glob_idx[jj+2][2,:]])))
            idx = [findfirst(isequal(num), idx_vec) for num in 1:nglob]
            N[id,:,k] = collect(Iterators.flatten([Nhat[N_idx[k][1,1]][id,:,N_idx[k][1,2]],
                                                   Nhat[N_idx[k][2,1]][id,:,N_idx[k][2,2]]]))[idx]
        end
    end

    R = zeros(T, (nglob,n,dim+1))
    for k=1:dim+1 
        jj = dim*(k-1)
        loc_idx = unique(collect(Iterators.flatten([loc_glob_idx[jj+1][2,:],loc_glob_idx[jj+2][2,:]])))
        glob_idx = unique(collect(Iterators.flatten([loc_glob_idx[jj+1][3,:],loc_glob_idx[jj+2][3,:]])))
        for i=1:nglob
            R[loc_idx[i],glob_idx[i],k] = 1.0
        end 
    end

    E = zeros(T, (n,n,dim))
    for i=1:dim 
        for k=1:dim+1
            E[:,:,i] += R[:,:,k]'*diagm(N[i,:,k])*B[:,:,k]*R[:,:,k]
        end
    end
    return B, N, R, E
end

function construct_line_tss_quadrature(p::Int; opertype::String="lgl", n1d::Int=-1, T=Float64)
    dim = 2 
    _, Bhat = tensor_quad_nodes(p,opertype=opertype,n1d=n1d)
    nf = length(Bhat)
    
    xf, loc_glob_idx = global_node_index_tri_facet(p, opertype=opertype, n1d=n1d)
    nglob = size(xf,2) 
    B = zeros(T,(nglob,nglob,dim+1))
    for k=1:dim+1
        for i=1:dim
            glob_idx = loc_glob_idx[(k-1)*dim+i]
            for j=1:nf
                Z = construct_zmatrix(glob_idx,j,j,nglob)
                B[:,:,k] += (Bhat[j]*Z)
            end
        end
    end

    _,_,_,_,Nhat,_ = map_tensor_operators_to_tri(p, opertype=opertype, n1d=n1d)
    N_idx = [[2 2; 3 4],[3 1; 1 1],[1 3; 2 3]]
    N = zeros(T, (dim, nglob,dim+1))
    for id=1:dim
        for k=1:dim+1
            jj = dim*(k-1)
            idx_vec=(collect(Iterators.flatten([loc_glob_idx[jj+1][2,:],loc_glob_idx[jj+2][2,:]])))
            idx = [findfirst(isequal(num), idx_vec) for num in 1:nglob]
            N[id,:,k] = collect(Iterators.flatten([Nhat[N_idx[k][1,1]][id,:,N_idx[k][1,2]],
                                                   Nhat[N_idx[k][2,1]][id,:,N_idx[k][2,2]]]))[idx]
        end
    end

    return vec(xf[1,:,3]), diag(abs.(B[:,:,3].*N[2,:,3]))
end



"""
### SummationByParts.stat_sst_sbp

Returns DOF and numbor of nonzero elements for SST SBP operators

**Inputs**
* `p`: degree of the operator 
* `dim`: dimesion (2d or 3d)
* `opertype`: the type of 1d operator used to construct the SST-SBP method (lgl or csbp)
* `n1d`: number of nodes in the 1D operator 

**Outputs** 
* `dof`: degrees of freedom in the element (number of nodes)
* `nnz`: number of nonzero elements in the derivative matrix
"""
function stat_sst_sbp(p::Int,dim::Int; opertype::String="lgl", n1d::Int=-1)
    if opertype=="lgl"
        n1=p+1
        nn = n1
    elseif opertype=="csbp"
        if n1d==-1
            n1d=4*p 
        end
        n1=n1d 
        nn = 2*p
    else 
        error("Operator not implemented. Should choose between 'lgl' and 'csbp'.")
    end
    # n1 = p+1
    dof = (dim+1)*n1^dim - (dim+dim^(dim-2))*n1^(dim-1) + (dim-2)*((dim+1)*n1 - 2) + 1
    # nnz = (dim*p + 1)*dof
    if opertype=="lgl"
        nnz = (nn+(nn-1)*(dim-1))*dof
    elseif opertype=="csbp"
        nnz = (nn+(nn-1)*(dim-1))*dof #(4*p*(dim+1)*(2*dim)) + (nn+(nn-1)*(dim-1))*(dof-(4*p*(dim+1)*(2*dim)))
    end
    return dof, nnz
end