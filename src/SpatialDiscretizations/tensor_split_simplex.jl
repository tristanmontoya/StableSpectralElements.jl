using LinearAlgebra
using SparseArrays
"""
### SummationByParts.tensor_lgl_quad_nodes

Computes tensor-product nodes on a quadrilateral 
**Inputs**
* `p`: degree of the operator 
* `opertype`: the type of 1d operator used to construct the SST-SBP method (lgl)
* `n1d`: number of nodes in the 1D operator 

**Outputs** 
* `xy`: node coordinates
* `w`: weights of the 1D operator 
"""
function tensor_quad_nodes(p::Int;opertype::String="lgl", n1d::Int=-1)
    if opertype=="lgl"
        q,w = quadrature(Line(), GaussLobattoQuadrature(p,0,0))

        Q = length(q)
        x = repeat(q, Q)          
        y = repeat(q, inner=Q)   
        xy = [x'; y']                 
        return xy, w
    else 
        error("Operator not implemented. Must be 'lgl.")
    end
end

"""
### SummationByParts.tensor_lgl_hex_nodes

Computes tensor-product nodes on a hexahedron
**Inputs**
* `p`: degree of the operator 
* `opertype`: the type of 1d operator used to construct the SST-SBP method (lgl)
* `n1d`: number of nodes in the 1D operator 

**Outputs** 
* `xyz`: node coordinates
* `w`: weights of the 1D operator 
"""
function tensor_hex_nodes(p::Int; opertype::String="lgl", n1d::Int=-1)

    if opertype=="lgl"
        q,w = quadrature(Line(), GaussLobattoQuadrature(p,0,0))

        Q = length(q)

        x = repeat(q, Q * Q)                
        y = repeat(repeat(q, inner=Q), outer=Q)   
        z = repeat(q, inner=Q^2)                     

        xyz = [x'; y'; z']

        return xyz, w
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
### SummationByParts.cube_hex_map.jl 

Maps points in the standard cube domain, [-1,1]^3, to any hexahedron 

**Inputs** 
* `xp`: points in the standard cube domain 
* `hex_vert`: coordinates of the vertices of the general hexahedron element

**Outputs** 
* `x`: the mapped points in the hexahedral element 
"""
function cube_hex_map(xp::Array{T},hex_vert::Array{T}) where T
    xi = xp[1]
    eta = xp[2]
    zeta = xp[3]
    psi = []

    push!(psi, 1/8*(1-xi)*(1-eta)*(1-zeta))
    push!(psi, 1/8*(1+xi)*(1-eta)*(1-zeta))
    push!(psi, 1/8*(1-xi)*(1+eta)*(1-zeta))
    push!(psi, 1/8*(1+xi)*(1+eta)*(1-zeta))
    push!(psi, 1/8*(1-xi)*(1-eta)*(1+zeta))
    push!(psi, 1/8*(1+xi)*(1-eta)*(1+zeta))
    push!(psi, 1/8*(1-xi)*(1+eta)*(1+zeta))
    push!(psi, 1/8*(1+xi)*(1+eta)*(1+zeta))

    x = zeros(3,1)
    for i=1:3
        for j=1:8
            x[i] += hex_vert[j,i]*psi[j]
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
### SummationByParts.get_quad_vert

Returns the vertices of the 4 hexahedra obtained by splitting the 
standard tetrahedron with vertices [-1 -1 -1; 1 -1 -1; -1 1 -1; -1 -1 1]

**Outputs** 
* `hex_vert`: vertices of the 4 hexahedra 
"""
function get_hex_vert(;T=Float64)
    v1 = T[-1 -1 -1]
    v2 = T[1 -1 -1]
    v3 = T[-1 1 -1]
    v4 = T[-1 -1 1]
    v5 = T[0 0 -1]
    v6 = T[-1 0 -1]
    v7 = T[0 -1 -1]
    v8 = T[-1 0 0]
    v9 = T[-1 -1 0]
    v10 = T[0 -1 0]
    v11 = T[-1/3 -1/3 -1]
    v12 = T[-1/3 -1/3 -1/3]
    v13 = T[-1 -1/3 -1/3]
    v14 = T[-1/3 -1 -1/3]
    v15 = T[-1/2 -1/2 -1/2]
    hex_vert = [[v2; v5; v7; v11; v10; v12; v14; v15],
                [v5; v3; v11; v6; v12; v8; v15; v13],
                [v7; v11; v1; v6; v14; v15; v9; v13],
                [v10; v4; v12; v8; v14; v9; v15; v13]]
    return hex_vert
end

"""
### SummationByParts.cube_to_tet_map.jl 
Maps points from the standard cube domain, [-1,1]^3, to the hexahedra
generated in the split-tetrahedron

**Inputs** 
* `xi`: the point in the stadard cube element 

**Outputs**
* `x`: the correspoinding points in the 4 hexahedra in the split-tetrahedron
"""
function cube_to_tet_map(xi::Array{T}) where T
    hex_vert = get_hex_vert()
    n = size(xi,2)
    x = zeros(3,4*n)
    for i=0:3 
        for j=1:n
            xp = cube_hex_map(xi[:,j], Matrix(hex_vert[i+1]))
            x[:,i*n+j] = xp 
        end
    end
    return x
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
### SummationByParts.metric_tet

Computes the metric terms for a point mapped from the standard cube domain 
to a hexahedron in the split-tetrahedron

**Inputs** 
* `xp`: the coordinates of the point in the standard cube domain 
* `hex_vert`: A matrix containing the vertices of the hexahedron in the split-tetrahedron

**Outputs** 
* `dxi`: A column vector containing the metric terms [dx/dxi,dx/deta,dx/dzeta,dy/dxi,dy/deta,dy/dzeta,dz/dxi,dz/deta,dz/dzeta]
* `dx`: A column vector containing the metric terms [dxi/dx,dxi/dy,dxi/dz,deta/dx,deta/dy,deta/dz,dzeta/dx,dzeta/dy,dzeta/dz]
* `Jac`: The metric Jacobian 
"""
function metric_tet!(xp::Array{T},hex_vert::Array{T},dxi::SubArray{T},dx::SubArray{T},Jac::SubArray{T}) where T
    ξ = xp[1]
    η = xp[2]
    ζ = xp[3]
    ∂Ψ∂ξ = []
    ∂Ψ∂η = []
    ∂Ψ∂ζ = []

    push!(∂Ψ∂ξ, -1/8*(1-η)*(1-ζ))
    push!(∂Ψ∂ξ, 1/8*(1-η)*(1-ζ))
    push!(∂Ψ∂ξ, -1/8*(1+η)*(1-ζ))
    push!(∂Ψ∂ξ, 1/8*(1+η)*(1-ζ))
    push!(∂Ψ∂ξ, -1/8*(1-η)*(1+ζ))
    push!(∂Ψ∂ξ, 1/8*(1-η)*(1+ζ))
    push!(∂Ψ∂ξ, -1/8*(1+η)*(1+ζ))
    push!(∂Ψ∂ξ, 1/8*(1+η)*(1+ζ))

    push!(∂Ψ∂η, -1/8*(1-ξ)*(1-ζ))
    push!(∂Ψ∂η, -1/8*(1+ξ)*(1-ζ))
    push!(∂Ψ∂η, 1/8*(1-ξ)*(1-ζ))
    push!(∂Ψ∂η, 1/8*(1+ξ)*(1-ζ))
    push!(∂Ψ∂η, -1/8*(1-ξ)*(1+ζ))
    push!(∂Ψ∂η, -1/8*(1+ξ)*(1+ζ))
    push!(∂Ψ∂η, 1/8*(1-ξ)*(1+ζ))
    push!(∂Ψ∂η, 1/8*(1+ξ)*(1+ζ))

    push!(∂Ψ∂ζ, -1/8*(1-ξ)*(1-η))
    push!(∂Ψ∂ζ, -1/8*(1+ξ)*(1-η))
    push!(∂Ψ∂ζ, -1/8*(1-ξ)*(1+η))
    push!(∂Ψ∂ζ, -1/8*(1+ξ)*(1+η))
    push!(∂Ψ∂ζ, 1/8*(1-ξ)*(1-η))
    push!(∂Ψ∂ζ, 1/8*(1+ξ)*(1-η))
    push!(∂Ψ∂ζ, 1/8*(1-ξ)*(1+η))
    push!(∂Ψ∂ζ, 1/8*(1+ξ)*(1+η))

    ∂x∂ξ = 0.0
    ∂x∂η = 0.0
    ∂x∂ζ = 0.0
    ∂y∂ξ = 0.0
    ∂y∂η = 0.0
    ∂y∂ζ = 0.0 
    ∂z∂ξ = 0.0 
    ∂z∂η = 0.0
    ∂z∂ζ = 0.0

    for j=1:8
        ∂x∂ξ += hex_vert[j,1]*∂Ψ∂ξ[j]
        ∂x∂η += hex_vert[j,1]*∂Ψ∂η[j]
        ∂x∂ζ += hex_vert[j,1]*∂Ψ∂ζ[j]

        ∂y∂ξ += hex_vert[j,2]*∂Ψ∂ξ[j]
        ∂y∂η += hex_vert[j,2]*∂Ψ∂η[j]
        ∂y∂ζ += hex_vert[j,2]*∂Ψ∂ζ[j]

        ∂z∂ξ += hex_vert[j,3]*∂Ψ∂ξ[j]
        ∂z∂η += hex_vert[j,3]*∂Ψ∂η[j]
        ∂z∂ζ += hex_vert[j,3]*∂Ψ∂ζ[j]
    end

    J = (∂x∂ξ*∂y∂η*∂z∂ζ + ∂x∂η*∂y∂ζ*∂z∂ξ + ∂x∂ζ*∂y∂ξ*∂z∂η - 
            ∂x∂ζ*∂y∂η*∂z∂ξ - ∂x∂η*∂y∂ξ*∂z∂ζ - ∂x∂ξ*∂y∂ζ*∂z∂η)
    
    ∂ξ∂x = 1/J * (∂y∂η*∂z∂ζ - ∂y∂ζ*∂z∂η) 
    ∂η∂x = 1/J * (∂y∂ζ*∂z∂ξ - ∂y∂ξ*∂z∂ζ)
    ∂ζ∂x = 1/J * (∂y∂ξ*∂z∂η - ∂y∂η*∂z∂ξ)
    
    ∂ξ∂y = 1/J * (∂x∂ζ*∂z∂η - ∂x∂η*∂z∂ζ)
    ∂η∂y = 1/J * (∂x∂ξ*∂z∂ζ - ∂x∂ζ*∂z∂ξ)
    ∂ζ∂y = 1/J * (∂x∂η*∂z∂ξ - ∂x∂ξ*∂z∂η)

    ∂ξ∂z = 1/J * (∂x∂η*∂y∂ζ - ∂x∂ζ*∂y∂η)
    ∂η∂z = 1/J * (∂x∂ζ*∂y∂ξ - ∂x∂ξ*∂y∂ζ)
    ∂ζ∂z = 1/J * (∂x∂ξ*∂y∂η - ∂x∂η*∂y∂ξ)

    Jac[1,1]=J
    dxi[1,1]=∂x∂ξ
    dxi[2,1]=∂x∂η
    dxi[3,1]=∂x∂ζ
    dxi[4,1]=∂y∂ξ
    dxi[5,1]=∂y∂η
    dxi[6,1]=∂y∂ζ
    dxi[7,1]=∂z∂ξ
    dxi[8,1]=∂z∂η
    dxi[9,1]=∂z∂ζ

    dx[1,1]=∂ξ∂x
    dx[2,1]=∂ξ∂y
    dx[3,1]=∂ξ∂z
    dx[4,1]=∂η∂x
    dx[5,1]=∂η∂y
    dx[6,1]=∂η∂z
    dx[7,1]=∂ζ∂x
    dx[8,1]=∂ζ∂y
    dx[9,1]=∂ζ∂z
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
### SummationByParts.normals_cube

Returns the normals on the standard cube domain [-1,1]^3 

**Inputs** 
* `nf`: the number of facet nodes 

**Outputs** 
* `N`: the normals at each facet node 
"""
function normals_cube(nf::Int; T=Float64)
   
    dim=3
    N = zeros(T, (dim,nf,6)) # normal vector for each facet

    N[1,:,1] .= -1.0
    N[1,:,2] .= 1.0
    N[2,:,3] .= -1.0
    N[2,:,4] .= 1.0
    N[3,:,5] .= -1.0
    N[3,:,6] .= 1.0

    return N
end

"""
### SummationByParts.facet_nodes_cube

Returns the global node index of each facet node in a cube element

**Inputs** 
* `n`: The number of nodes in the element 

**Outputs**
* `facet_node_idx`: The global node index for the facet nodes 
"""
function facet_nodes_cube(n::Int)
 
    n1 = convert(Int, round(n^(1/3)))
    nf = convert(Int, round(n^(2/3)))
    facet_node_idx = zeros(Int, (6,nf))

    facet_node_idx[1,:] = 1:n1:n
    facet_node_idx[2,:] = n1:n1:n
    facet_node_idx[3,:] = collect(Iterators.flatten([1:n1...] .+ (i-1)*nf for i in 1:n1))'
    facet_node_idx[4,:] = collect(Iterators.flatten([nf+1-n1:nf...] .+ (i-1)*nf for i in 1:n1))'
    facet_node_idx[5,:] = 1:nf 
    facet_node_idx[6,:] = n+1-nf:n

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
### SummationByParts.map_tensor_operators_to_tet

Maps the tensor-product operator to the hexahedral elements in the split-tetrahedron

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
function map_tensor_operators_to_tet(p::Int; opertype::String="lgl", n1d::Int=-1, T=Float64)
    dim = 3
    xc, B = tensor_hex_nodes(p, opertype=opertype, n1d=n1d)   #nodes on cube
    n = size(xc,2)
    n1 = round(n^(1/3))
    nf = convert(Int, round(n1^2))

    # qf = 2*(p+1)-3 
    # cub_lgl, vtx_lgl = SummationByParts.Cubature.quadrature(qf, internal=false)
    # perm = sortperm(vec(SymCubatures.calcnodes(cub_lgl, vtx_lgl)))
    # B = SymCubatures.calcweights(cub_lgl)[perm]
    B = vec(kron(B,B))

    hex_vert = get_hex_vert()
    Nhat = normals_cube(nf) 
    facet_node_idx = facet_nodes_cube(n) 

    dxis = []
    dxs = []
    Js = []
    Ns = []
    for i = 1:4
        dxi = zeros(9,n)
        dx = zeros(9,n)
        J = zeros(1,n)
        N = zeros(3,n)
        for j=1:n
            metric_tet!(xc[:,j],hex_vert[i],view(dxi,:,j),view(dx,:,j),view(J,:,j))
        end
        push!(dxis, dxi)
        push!(dxs, dx)
        push!(Js, J)

        N = zeros(T,(dim,nf,2*dim))
        for k=1:6
            for id=1:dim
                N[id,:,k] = J[facet_node_idx[k,:]].*(dx[id,facet_node_idx[k,:]].*Nhat[1,:,k] .+ 
                                                     dx[dim+id,facet_node_idx[k,:]].*Nhat[2,:,k] .+
                                                     dx[2*dim+id,facet_node_idx[k,:]].*Nhat[3,:,k])
            end
        end
        push!(Ns, N)
    end

    Hhat, Qhat, Dhat, Ehat, Rhat = tensor_operators(p, dim, opertype=opertype, n1d=n1d, T=T)
    Es = []
    for k=1:dim+1
        E = zeros(T, (n,n,dim))
        for i=1:dim
            for j=1:2*dim
                E[:,:,i] += Rhat[:,:,j]'*diagm(Ns[k][i,:,j].*B)*Rhat[:,:,j]
            end
        end
        push!(Es,E)
    end

    Hs = []
    Qs = []
    Ds = []
    for i=1:dim+1
        Q = zeros(T, (n,n,dim))
        E = Es[i]
        D = zeros(T, (n,n,dim))
        H = diagm(vec(Js[i]))*Hhat 
        push!(Hs, H)
        Sx = 0.5*(diagm(vec(Js[i]).*dxs[i][1,:]) * Qhat[:,:,1] + diagm(vec(Js[i]).*dxs[i][1+dim,:]) * Qhat[:,:,2] + diagm(vec(Js[i]).*dxs[i][1+2*dim,:]) * Qhat[:,:,3]) - 
             0.5*(Qhat[:,:,1]' * diagm(vec(Js[i]).*dxs[i][1,:]) + Qhat[:,:,2]' * diagm(vec(Js[i]).*dxs[i][1+dim,:]) + Qhat[:,:,3]' * diagm(vec(Js[i]).*dxs[i][1+2*dim,:]))
        Sy = 0.5*(diagm(vec(Js[i]).*dxs[i][2,:]) * Qhat[:,:,1] + diagm(vec(Js[i]).*dxs[i][2+dim,:]) * Qhat[:,:,2] + diagm(vec(Js[i]).*dxs[i][2+2*dim,:]) * Qhat[:,:,3]) - 
             0.5*(Qhat[:,:,1]' * diagm(vec(Js[i]).*dxs[i][2,:]) + Qhat[:,:,2]' * diagm(vec(Js[i]).*dxs[i][2+dim,:]) + Qhat[:,:,3]' * diagm(vec(Js[i]).*dxs[i][2+2*dim,:]))
        Sz = 0.5*(diagm(vec(Js[i]).*dxs[i][3,:]) * Qhat[:,:,1] + diagm(vec(Js[i]).*dxs[i][3+dim,:]) * Qhat[:,:,2] + diagm(vec(Js[i]).*dxs[i][3+2*dim,:]) * Qhat[:,:,3]) - 
             0.5*(Qhat[:,:,1]' * diagm(vec(Js[i]).*dxs[i][3,:]) + Qhat[:,:,2]' * diagm(vec(Js[i]).*dxs[i][3+dim,:]) + Qhat[:,:,3]' * diagm(vec(Js[i]).*dxs[i][3+2*dim,:]))
        
        Q[:,:,1] = Sx + 0.5.*E[:,:,1]
        Q[:,:,2] = Sy + 0.5.*E[:,:,2]
        Q[:,:,3] = Sz + 0.5.*E[:,:,3]
        push!(Qs, Q)
        D[:,:,1] = inv(H)*Q[:,:,1]
        D[:,:,2] = inv(H)*Q[:,:,2]
        D[:,:,3] = inv(H)*Q[:,:,3]
        # D[:,:,1] = diagm(vec(dxs[i][1,:]))*Dhat[:,:,1] + diagm(vec(dxs[i][4,:]))*Dhat[:,:,2] + diagm(vec(dxs[i][7,:]))*Dhat[:,:,3]
        # D[:,:,2] = diagm(vec(dxs[i][2,:]))*Dhat[:,:,1] + diagm(vec(dxs[i][5,:]))*Dhat[:,:,2] + diagm(vec(dxs[i][8,:]))*Dhat[:,:,3]
        # D[:,:,3] = diagm(vec(dxs[i][3,:]))*Dhat[:,:,1] + diagm(vec(dxs[i][6,:]))*Dhat[:,:,2] + diagm(vec(dxs[i][9,:]))*Dhat[:,:,3]
        push!(Ds, D)
    end 

    return Hs,Qs,Ds,Es,Ns
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
### SummationByParts.global_node_index_tet

Returns the local to global node index on the split-tetrahedron element 

**Inputs**
* `p`: degree of the operator 
* `opertype`: the type of 1d operator used to construct the TSS-SBP operator (lgl or csbp)
* `n1d`: number of nodes in the 1D operator 

**Outputs** 
* `xg`: Coordinates of global nodes in the split-tetrahedron
* `loc_glob_idx`: Local to global index matching 
"""
function global_node_index_tet(p::Int; opertype::String="lgl",n1d::Int=-1, T=Float64)
    xh,_ = tensor_hex_nodes(p,opertype=opertype,n1d=n1d)
    xt = cube_to_tet_map(xh)
    n = size(xh,2)

    facet_node_idx = facet_nodes_cube(n)
    xg = copy(xt)
    remove_idx = collect(Iterators.flatten([n.+facet_node_idx[1,:], 
                                            (2*n).+facet_node_idx[2,:],
                                            (2*n).+facet_node_idx[3,:],
                                            (3*n).+facet_node_idx[1,:],
                                            (3*n).+facet_node_idx[4,:],
                                            (3*n).+facet_node_idx[6,:]]))
    xg = xg[:, filter(x -> !(x in remove_idx), 1:size(xg, 2))]

    loc_glob_idx = []
    for k = 1:4
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
### SummationByParts.construct_split_operator_tet

Returns TSS-SBP operators on the reference tetrahedron 

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
function construct_split_operator_tet(p::Int; opertype::String="lgl", n1d::Int=-1, T=Float64)
    Hs,Qs,Ds,Es,_ = map_tensor_operators_to_tet(p, opertype=opertype, n1d=n1d)
    xg, loc_glob_idx = global_node_index_tet(p, opertype=opertype, n1d=n1d)
    n = size(Hs[1],1)
    nglob = size(xg,2) 
    dim = 3

    H = spzeros(nglob,nglob)
    Q = [spzeros(nglob,nglob),spzeros(nglob,nglob),spzeros(nglob,nglob)]
    D = [spzeros(nglob,nglob),spzeros(nglob,nglob),spzeros(nglob,nglob)]
    E = [spzeros(nglob,nglob),spzeros(nglob,nglob),spzeros(nglob,nglob)]
    for k=1:dim+1
        for i=1:n
            for j=1:n 
                glob_idx = loc_glob_idx[k]
                Z = construct_zmatrix(glob_idx,i,j,nglob)
                if Hs[k][i,j]!=0.0
                    H[:,:] += (Hs[k][i,j]*Z)
                end
                for id=1:dim
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
    Q = [Matrix(m) for m in Q]
    D = [Matrix(m) for m in D]
    E = [Matrix(m) for m in E]
    return Matrix(H),Q,D,E
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

"""
### SummationByParts.global_node_index_tet_facet

Returns the coordinates of the facet nodes and the 
local and global node index for the facet nodes on the tetrahedron 

**Inputs**
* `p`: degree of the operator 
* `opertype`: the type of 1d operator used to construct the TSS-SBP operator (lgl or csbp)
* `n1d`: number of nodes in the 1D operator 

**Outputs** 
* `xf`: Coordinates of the facet nodes on the tetrahedron
* `loc_glob_facet_idx`: The local to global index mapping of the facet nodes
"""
function global_node_index_tet_facet(p::Int;opertype::String="lgl",n1d::Int=-1, T=Float64)
    dim = 3
    xh,_= tensor_hex_nodes(p, opertype=opertype, n1d=n1d)
    xt = cube_to_tet_map(xh)
    xg,_ = global_node_index_tet(p, opertype=opertype, n1d=n1d)
    n = size(xh,2)
    n1 = convert(Int, round(n^(1/dim)))
    nf = n1^(dim-1)
    
    facet_node_idx = facet_nodes_cube(n)
    xf = zeros(T, (dim, dim*nf-3*n1+1, dim+1))
    keep_idx = []
    push!(keep_idx, collect(Iterators.flatten([facet_node_idx[3,:],
                                               (n.+facet_node_idx[3,:]), 
                                               ((3*n).+facet_node_idx[5,:])])))
    # push!(keep_idx, collect(Iterators.flatten([(n.+facet_node_idx[2,:]), 
    #                                             (2*n).+(facet_node_idx[4,:]),
    #                                             (3*n).+(facet_node_idx[2,:])])))
    push!(keep_idx, collect(Iterators.flatten([(n.+facet_node_idx[2,:]), 
                                                (2*n).+ collect(Iterators.flatten([reverse(facet_node_idx[4,:][i:min(i+n1-1, end)]) for i in 1:n1:length(facet_node_idx[4,:])])),
                                                (3*n).+ collect(Iterators.flatten([reverse(facet_node_idx[2,:][i:min(i+n1-1, end)]) for i in 1:n1:length(facet_node_idx[2,:])]))])))                                            
    push!(keep_idx, collect(Iterators.flatten([(facet_node_idx[1,:]), 
                                                (2*n).+(facet_node_idx[1,:]),
                                                (3*n).+(facet_node_idx[3,:])])))
    # push!(keep_idx, collect(Iterators.flatten([(facet_node_idx[5,:]), 
    #                                             (n).+(facet_node_idx[5,:]),
    #                                             (2*n).+(facet_node_idx[5,:])])))
    push!(keep_idx, collect(Iterators.flatten([(facet_node_idx[5,:]), 
                                                (n).+(facet_node_idx[5,:]),
                                                (2*n).+ reverse(facet_node_idx[5,:])])))

    unique_idx = [[keep_idx[1][1]],[keep_idx[2][1]],[keep_idx[3][1]],[keep_idx[4][1]]]
    for k=1:dim+1
        xtf = xt[:,keep_idx[k]]
        for i=2:nf*dim
            xtf_temp = xtf .- xtf[:,i]
            col_norm = [norm(xtf_temp[:, j]) for j in 1:size(xtf,2)]
            idx = argmin(col_norm[1:(i-1)])
            if norm(xtf_temp[:,idx]) > 1e-14
                push!(unique_idx[k],keep_idx[k][i])
            end
        end
    end

    for k=1:dim+1
        xf[:,:,k] = xt[:,unique_idx[k]]
    end
    xfall= zeros(T,(dim,nf,dim*(dim+1)))
    xfall[:,:,1] = xt[:,facet_node_idx[3,:]]
    xfall[:,:,2] = xt[:,(n.+facet_node_idx[3,:])]
    xfall[:,:,3] = xt[:,((3*n).+facet_node_idx[5,:])]
    xfall[:,:,4] = xt[:,(n.+facet_node_idx[2,:])]
    xfall[:,:,5] = xt[:,(2*n).+(facet_node_idx[4,:])]
    xfall[:,:,6] = xt[:,(3*n).+(facet_node_idx[2,:])]
    xfall[:,:,7] = xt[:,(facet_node_idx[1,:])]
    xfall[:,:,8] = xt[:,(2*n).+(facet_node_idx[1,:])]
    xfall[:,:,9] = xt[:,(3*n).+(facet_node_idx[3,:])]
    xfall[:,:,10] = xt[:,(facet_node_idx[5,:])]
    xfall[:,:,11] = xt[:,(n).+(facet_node_idx[5,:])]
    xfall[:,:,12] = xt[:,(2*n).+(facet_node_idx[5,:])]

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
### SummationByParts.construct_split_facet_operator_tet

Constructs the TSS-SBP facet operators on the tetrahedron

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
function construct_split_facet_operator_tet(p::Int; opertype::String="lgl", n1d::Int=-1, T=Float64)
    dim = 3 
    # qf = 2*(p+1)-3 
    # cub_lgl, vtx_lgl = SummationByParts.Cubature.quadrature(qf, internal=false)
    # perm = sortperm(vec(SymCubatures.calcnodes(cub_lgl, vtx_lgl)))
    # B1 = SymCubatures.calcweights(cub_lgl)[perm]
    _, B1 = tensor_quad_nodes(p,opertype=opertype,n1d=n1d)

    Bhat = kron(diagm(B1),diagm(B1))
    _,_,_,_,Nhat = map_tensor_operators_to_tet(p,opertype=opertype,n1d=n1d)
    n1 = length(B1)
    nf = n1^(dim-1)
    n = (dim+1)*n1^dim - (dim+dim^(dim-2))*n1^(dim-1) + (dim-2)*((dim+1)*n1 - 2) + 1
    # N_idx = [1 3; 2 2; 3 1; 1 5] #first column contains element number, and second column contains facet number of the element
    # N_idx = [[1 3; 2 3; 4 5],[2 2; 3 4; 4 2],[1 1; 3 1; 4 3],[1 5; 2 5; 3 5]]
    N_idx = [[1 3; 2 3; 4 5],[2 2; 3 4; 4 2],[1 1; 3 1; 4 3],[1 5; 2 5; 3 5]]
    
    xf, loc_glob_idx = global_node_index_tet_facet(p,opertype=opertype,n1d=n1d)
    nglob = size(xf,2) 
    B = zeros(T,(nglob,nglob,dim+1))
    N = zeros(T, (dim, nglob,dim+1))
    for k=1:dim+1
        for i=1:dim
            glob_idx = loc_glob_idx[(k-1)*dim+i]
            for j=1:nf
                Z = construct_zmatrix(glob_idx,j,j,nglob)
                B[:,:,k] += (Bhat[j,j]*Z)
            end
        end
    end
    
    for id=1:dim
        for k=1:dim+1
            jj = dim*(k-1)
            idx_vec=(collect(Iterators.flatten([loc_glob_idx[jj+1][2,:],loc_glob_idx[jj+2][2,:],loc_glob_idx[jj+3][2,:]])))
            idx = [findfirst(isequal(num), idx_vec) for num in 1:nglob]
            N[id,:,k] = collect(Iterators.flatten([Nhat[N_idx[k][1,1]][id,:,N_idx[k][1,2]],
                                                   Nhat[N_idx[k][2,1]][id,:,N_idx[k][2,2]],
                                                   Nhat[N_idx[k][3,1]][id,:,N_idx[k][3,2]]]))[idx]
        end
    end
    # N = ones(T, (dim, size(xf,2),dim+1))
    # for k=1:dim+1
    #     for i=1:dim 
    #         N[i,:,k] = Nhat[N_idx[k,1]][i,:,N_idx[k,2]] * N[i,:,k]
    #     end
    # end

    R = zeros(T, (nglob,n,dim+1))
    for k=1:dim+1 
        jj = dim*(k-1)
        loc_idx = unique(collect(Iterators.flatten([loc_glob_idx[jj+1][2,:],loc_glob_idx[jj+2][2,:],loc_glob_idx[jj+3][2,:]])))
        glob_idx = unique(collect(Iterators.flatten([loc_glob_idx[jj+1][3,:],loc_glob_idx[jj+2][3,:],loc_glob_idx[jj+3][3,:]])))
        for i=1:nglob
            R[loc_idx[i],glob_idx[i],k] = 1.0
        end 
    end

    # xg, _ = global_node_index_tet(p)
    E = zeros(T, (n,n,dim))
    for i=1:dim 
        for k=1:dim+1
            E[:,:,i] += R[:,:,k]'*diagm(N[i,:,k])*B[:,:,k]*R[:,:,k]
        end
    end
    return B, N, R, E
end