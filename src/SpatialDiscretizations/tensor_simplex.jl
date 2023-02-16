"""Duffy transform from the square to triangle"""
@inline function χ(::Tri, 
    η::Union{NTuple{2,Float64},NTuple{2,Vector{Float64}}})
    return (0.5.*(1.0 .+ η[1]).*(1.0 .- η[2]) .- 1.0, η[2])
end

"""Duffy transform from the cube to tetrahedron"""
@inline function χ(::Tet, 
    η::Union{NTuple{3,Float64},NTuple{3,Vector{Float64}}})
    ξ_pri = (0.5.*(1.0 .+ η[1]).*(1.0 .- η[3]) .- 1.0, η[2], η[3])
    ξ_pyr = (ξ_pri[1], 0.5.*(1.0 .+ η[2]).*(1.0 .- η[3]) .- 1.0, ξ_pri[3])
    return (0.5.*(1.0 .+ ξ_pri[1]).*(1.0 .- η[2]) .- 1.0, ξ_pyr[2] , ξ_pyr[3])
end

function reference_geometric_factors(::Tri, 
    η::NTuple{2,Vector{Float64}})

    N = size(η[1],1)
    Λ_ref = Array{Float64, 3}(undef, N, 2, 2)
    
    J_ref = (x->0.5*(1.0-x)).(η[2])

    Λ_ref[:,1,1] = ones(N) # Jdη1/dξ1
    Λ_ref[:,1,2] = (x->0.5*(1.0+x)).(η[1]) # Jdη1/dξ2
    Λ_ref[:,2,1] = zeros(N) # Jdη2/dξ1
    Λ_ref[:,2,2] = (x->0.5*(1.0-x)).(η[2]) # Jdη2/dξ2

    return J_ref, Λ_ref
end

function reference_geometric_factors(::Tet, 
    η::NTuple{3,Vector{Float64}})

    N = size(η[1],1)
    Λ_ref = Array{Float64, 3}(undef, N, 3, 3)
    
    J_ref = (x->0.5*(1.0-x)).(η[2]) .* (x->(0.5*(1.0-x))^2).(η[3])

    Λ_ref[:,1,1] = (x->0.5*(1.0-x)).(η[3]) # Jdη1/dξ1
    Λ_ref[:,1,2] = (x->0.5*(1.0+x)).(η[1]) .* 
        (x->0.5*(1.0-x)).(η[3])  # Jdη1/dξ2
    Λ_ref[:,1,3] = (x->0.5*(1.0+x)).(η[1]) .* 
        (x->0.5*(1.0-x)).(η[3])  # Jdη1/dξ3
    
    Λ_ref[:,2,1] = zeros(N) # Jdη2/dξ1
    Λ_ref[:,2,2] = (x->0.5*(1.0-x)).(η[2]) .*
        (x->0.5*(1.0-x)).(η[3]) # Jdη2/dξ2
    Λ_ref[:,2,3] = (x->0.5*(1.0+x)).(η[2]) .* 
        (x->0.5*(1.0-x)).(η[2]) .*
        (x->0.5*(1.0-x)).(η[3]) # Jdη2/dξ3

    Λ_ref[:,3,1] = zeros(N) # Jdη3/dξ1
    Λ_ref[:,3,2] = zeros(N) # Jdη3/dξ2
    Λ_ref[:,3,3] = (x->0.5*(1.0-x)).(η[2]) .*
        (x->(0.5*(1.0-x))^2).(η[3])# Jdη3/dξ3

    return J_ref, Λ_ref
end

function warped_product(::Tri, p, η1D::NTuple{2,Vector{Float64}})

    (M1, M2) = (length(η1D[1]), length(η1D[2]))
    σₒ = [M2*(i-1) + j for i in 1:M1, j in 1:M2]
    σᵢ = zeros(Int,p+1,p+1)

    A = Matrix{Float64}(undef, M1, p+1)
    B = [Matrix{Float64}(undef, M2, p-i+1) for i = 0:p]

    k = 1
    for i = 0:p
        for j = 0:p-i
            σᵢ[i+1,j+1] = k
            k = k + 1
            for α1 in 1:M1, α2 in 1:M2
                A[α1,i+1] = sqrt(2) * jacobiP(η1D[1][α1],0,0,i)
                B[i+1][α2,j+1] = (1-η1D[2][α2])^i * jacobiP(η1D[2][α2],2i+1,0,j)
            end
        end
    end

    return WarpedTensorProductMap2D(A,B,σᵢ,σₒ)
end

function operators_1d(
    quadrature_rule::NTuple{d,AbstractQuadratureRule}) where {d}

    η_1D, q, V_1D, D_1D, R_L, R_R = fill((),6)

    for m in 1:d
        η_1D = (η_1D..., quadrature(Line(),quadrature_rule[m])[1])
        q = (q..., length(η_1D[m]) - 1)
        V_1D = (V_1D..., vandermonde(Line(),q[m],η_1D[m] ))
        D_1D = (D_1D..., grad_vandermonde(Line(),q[m], η_1D[m]) / V_1D[m])
        R_L = (R_L..., vandermonde(Line(),q[m],[-1.0]) / V_1D[m])
        R_R = (R_R..., vandermonde(Line(),q[m],[1.0]) / V_1D[m])
    end
    
    return η_1D, q, V_1D, D_1D, R_L, R_R
end

function ReferenceApproximation(
    approx_type::Union{NodalTensor,ModalTensor}, 
    ::Tri; mapping_degree::Int=1, 
    N_plot::Int=10, volume_quadrature_rule=(LGQuadrature(approx_type.p),
    LGRQuadrature(approx_type.p)), 
    facet_quadrature_rule=LGQuadrature(approx_type.p))

    # one-dimensional operators
    η_1D, q, V_1D, D_1D, R_L, R_R = operators_1d(volume_quadrature_rule)
    
    # nodes and weights on the square
    η1, η2, w_η = quadrature(Quad(), volume_quadrature_rule)

    # differentiation operator on the square
    σ = [(q[2]+1)*(i-1) + j for i in 1:q[1]+1, j in 1:q[2]+1]
    D = (TensorProductMap2D(D_1D[1], I, σ, σ), 
        TensorProductMap2D(I, D_1D[2], σ, σ))

    J_ref, Λ_ref = reference_geometric_factors(Tri(),(η1,η2))

    reference_element = RefElemData(Tri(), approx_type, mapping_degree,
        volume_quadrature_rule=volume_quadrature_rule,
        facet_quadrature_rule=facet_quadrature_rule,  Nplot=N_plot)
        
    mortar_nodes, _ = quadrature(Line(), facet_quadrature_rule)

    # bottom
    σ_1 = [i for i in 1:q[1]+1, j in 1:1]
    if volume_quadrature_rule[1] == facet_quadrature_rule
        P_1 = LinearMap(I, q[1]+1)
    else
        P_1 = LinearMap(vandermonde(Line(),q[1],mortar_nodes) / V_1D[1])
    end
    if (volume_quadrature_rule[2] isa LGRQuadrature ||
        volume_quadrature_rule[2] isa JGRQuadrature)
        R_1 = SelectionMap([(q[2]+1)*(i-1)+1 
            for i in 1:q[1]+1], (q[1]+1)*(q[2]+1))
    else
        R_1 = TensorProductMap2D(I, R_L[2], σ, σ_1)
    end

    # hypotenuse
    σ_2 = [j for i in 1:1, j in 1:q[2]+1]
    if volume_quadrature_rule[2] == facet_quadrature_rule
        P_2 = LinearMap(I, q[2]+1)
    else
        P_2 = LinearMap(vandermonde(Line(),q[2],mortar_nodes) / V_1D[2])
    end
    R_2 = TensorProductMap2D(R_R[1], I, σ, σ_2)

    # left
    σ_3 = [j for i in 1:1, j in 1:q[2]+1]
    if volume_quadrature_rule[2] == facet_quadrature_rule
        P_3 = LinearMap(I, q[2]+1)
    else
        P_3 = LinearMap(vandermonde(Line(),q[2],mortar_nodes) / V_1D[2])
    end
    R_3 = TensorProductMap2D(R_L[1], I, σ, σ_3)

    # combine to extrapolate to all facets
    R = [P_1 * R_1; P_2 * R_2; P_3 * R_3]

    # construct nodal or modal scheme
    if approx_type isa ModalTensor
        V = warped_product(Tri(),approx_type.p, η_1D)
        V_plot = LinearMap(vandermonde(Tri(), 
            approx_type.p, reference_element.rstp...))
        new_approx_type = approx_type
    else
        V = LinearMap(I, (q[1]+1)*(q[2]+1))
        VDM_plot_1D = (vandermonde(Line(), q[1], equi_nodes(Line(),N_plot)),
            vandermonde(Line(), q[2], equi_nodes(Line(),N_plot)))
        V_plot = LinearMap(kron(VDM_plot_1D[1]/V_1D[1], VDM_plot_1D[2]/V_1D[2]))
        new_approx_type = NodalTensor(min(q[1],q[2]))
    end

    return ReferenceApproximation(new_approx_type, size(V,2), 
        length(reference_element.wq), length(reference_element.wf),
        reference_element, D, V, R * V, R, Diagonal(J_ref .* w_η), 
        Diagonal(reference_element.wf), V_plot, 
        ReferenceMapping(J_ref, Λ_ref))
end

function ReferenceApproximation(
    approx_type::Union{NodalTensor,ModalTensor}, 
    ::Tet; mapping_degree::Int=1, N_plot::Int=10,
    volume_quadrature_rule=(LGQuadrature(approx_type.p),
        LGQuadrature(approx_type.p), LGQuadrature(approx_type.p)), 
    facet_quadrature_rule=(LGQuadrature(approx_type.p), 
        LGQuadrature(approx_type.p)))

    # one-dimensional operators
    η_1D, q, V_1D, D_1D, R_L, R_R = operators_1d(volume_quadrature_rule)
    N_q = prod(q[m]+1 for m in 1:3)

    # nodes and weights on the cube
    η1, η2, η3, w_η = quadrature(Hex(), volume_quadrature_rule)

    # differentiation operator on the cube
    σ =  [(i-1)*(q[2]+1)*(q[3]+1) + (j-1)*(q[3]+1) + k 
        for i in 1:(q[1]+1), j in 1:(q[2]+1), k in 1:(q[3]+1)]
    D = (TensorProductMap3D(D_1D[1], I, I, σ, σ),
         TensorProductMap3D(I, D_1D[2], I, σ, σ),
         TensorProductMap3D(I, I, D_1D[3], σ, σ))

    # reference geometric factors for cube-to-tetrahedron mapping
    J_ref, Λ_ref = reference_geometric_factors(Tet(),(η1, η2, η3))

    # reference element data
    reference_element = RefElemData(Tet(), approx_type, mapping_degree,
        volume_quadrature_rule=volume_quadrature_rule,
        facet_quadrature_rule=facet_quadrature_rule,  Nplot=N_plot)

    σ_13 =  [(i-1)*(q[3]+1) + k  for i in 1:(q[1]+1), j in 1:1, k in 1:(q[3]+1)]
    σ_23 =  [(j-1)*(q[3]+1) + k  for i in 1:1, j in 1:(q[2]+1), k in 1:(q[3]+1)]
    σ_12 =  [(i-1)*(q[2]+1) + j  for i in 1:(q[1]+1), j in 1:(q[2]+1), k in 1:1]
    
    # 2D facet quadrature nodes and weights
    η_2d_1, _ = quadrature(Line(), facet_quadrature_rule[1])
    η_2d_2, _ = quadrature(Line(), facet_quadrature_rule[2])
    q_f = (length(η_2d_1)-1, length(η_2d_2)-1)
    σ_f = [(q_f[2]+1)*(i-1) + j for i in 1:q_f[1]+1, j in 1:q_f[2]+1]

    # front (η2 = -1)
    if (volume_quadrature_rule[1] == facet_quadrature_rule[1]) &&
        (volume_quadrature_rule[3] == facet_quadrature_rule[2])
        P_1 = LinearMap(I, (q[1]+1)*(q[3]+1) )
    else
        P_1 = TensorProductMap2D(vandermonde(Line(),q[1],η_2d_1) / V_1D[1],
            vandermonde(Line(),q[3],η_2d_2) / V_1D[3], σ_13[:,1,:], σ_f)
    end
    R_1 = TensorProductMap3D(I, R_L[2], I, σ, σ_13)

    # hypotenuse (η1 = +1)
    if (volume_quadrature_rule[2] == facet_quadrature_rule[1]) &&
        (volume_quadrature_rule[3] == facet_quadrature_rule[2])
        P_2 = LinearMap(I, (q[2]+1)*(q[3]+1) )
    else
        P_2 = TensorProductMap2D(vandermonde(Line(),q[2],η_2d_1) / V_1D[2],
            vandermonde(Line(),q[3],η_2d_2) / V_1D[3], σ_23[1,:,:], σ_f)
    end
    R_2 = TensorProductMap3D(R_R[1], I, I, σ, σ_23)

    # left (η1 = -1)
    if (volume_quadrature_rule[2] == facet_quadrature_rule[1]) &&
        (volume_quadrature_rule[3] == facet_quadrature_rule[2])
        P_3 = LinearMap(I, (q[2]+1)*(q[3]+1) )
    else
        P_3 = TensorProductMap2D(vandermonde(Line(),q[2],η_2d_1) / V_1D[2],
            vandermonde(Line(),q[3],η_2d_2) / V_1D[3], σ_23[1,:,:], σ_f)
    end
    R_3 = TensorProductMap3D(R_L[1], I, I, σ, σ_23)

    # bottom (η3 = -1)
    if (volume_quadrature_rule[1] == facet_quadrature_rule[1]) &&
        (volume_quadrature_rule[2] == facet_quadrature_rule[2])
        P_4 = LinearMap(I, (q[1]+1)*(q[2]+1) )
    else
        P_4 = TensorProductMap2D(vandermonde(Line(),q[1],η_2d_1) / V_1D[1],
            vandermonde(Line(),q[2],η_2d_2) / V_1D[2], σ_12[:,:,1], σ_f)
    end
    R_4 = TensorProductMap3D(I, I, R_L[3], σ, σ_12)

    # combine to extrapolate to all facets
    R = [P_1 * R_1; P_2 * R_2; P_3 * R_3; P_4 * R_4]

    @unpack rstq, rstf, rstp, wq, wf = reference_element

    if approx_type isa ModalTensor
        V = LinearMap(vandermonde(Tet(),approx_type.p, rstq...))
        V_plot = LinearMap(vandermonde(Tet(), 
            approx_type.p, rstp...))
        new_approx_type = approx_type
    else
        V = LinearMap(I, (q[1]+1)*(q[2]+1)*(q[3]+1))
        VDM_plot_1D = (vandermonde(Line(), q[1], equi_nodes(Line(),N_plot)),
            vandermonde(Line(), q[2], equi_nodes(Line(),N_plot)),
            vandermonde(Line(), q[3], equi_nodes(Line(),N_plot)) )
        V_plot = LinearMap(kron(VDM_plot_1D[1]/V_1D[1], VDM_plot_1D[2]/V_1D[2],
            VDM_plot_1D[3]/V_1D[3]))
        new_approx_type = NodalTensor(min(q[1],q[2],q[3]))
    end
    
    return ReferenceApproximation(new_approx_type, size(V,2), 
        length(wq), length(wf),reference_element, D, V, R * V, R, 
        Diagonal(J_ref .* w_η),  Diagonal(wf), V_plot, 
        ReferenceMapping(J_ref, Λ_ref))
end