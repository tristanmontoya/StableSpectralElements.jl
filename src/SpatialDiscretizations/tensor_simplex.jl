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
    J_ref = (x->0.5*(1.0-x)).(η[2])
    Λ_ref = Array{Float64, 3}(undef, N, 2, 2)
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
    Λ_ref[:,1,2] = (x->0.5*(1.0+x)).(η[1]) .* 
        (x->0.5*(1.0-x)).(η[3])  # Jdη1/dξ3
    
    Λ_ref[:,2,1] = zeros(N) # Jdη2/dξ1
    Λ_ref[:,2,2] = (x->0.5*(1.0-x)).(η[2]) .*
        (x->0.5*(1.0-x)).(η[3]) # Jdη2/dξ2
    Λ_ref[:,2,3] = (x->0.5*(1.0+x)).(η[1]) .* 
        (x->0.5*(1.0-x)).(η[2]) .*
        (x->0.5*(1.0-x)).(η[3]) # Jdη2/dξ3

    Λ_ref[:,3,1] = zeros(N) # Jdη3/dξ1
    Λ_ref[:,3,2] = zeros(N) # Jdη3/dξ2
    Λ_ref[:,3,3] = (x->0.5*(1.0-x)).(η[2]) .*
        ((x->0.5*(1.0-x)).(η[3])).^2 # Jdη3/dξ3

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

    # reference geometric factors for square-to-triangle mapping
    reference_mapping = ReferenceMapping(
        reference_geometric_factors(Tri(),(η1,η2))...)

    mortar_nodes, mortar_weights = quadrature(Line(), facet_quadrature_rule)

    P = (LinearMap(vandermonde(Line(),q[1],mortar_nodes) / V_1D[1]), 
        LinearMap(vandermonde(Line(),q[2],mortar_nodes) / V_1D[2]),
        LinearMap(vandermonde(Line(),q[2],mortar_nodes[end:-1:1]) / V_1D[2]))
    
    reference_element = RefElemData(Tri(), Polynomial(), mapping_degree,
        quad_rule_vol=quadrature(Tri(),volume_quadrature_rule),
        quad_rule_face=(mortar_nodes,mortar_weights), Nplot=N_plot)

    # ordering of facet nodes
    σ_1 = [i for i in 1:q[1]+1, j in 1:1] # bottom
    σ_2 = [j for i in 1:1, j in 1:q[2]+1] # hypotenuse
    σ_3 = σ_2 # left

    # interpolation/extrapolation operators
    if volume_quadrature_rule[2] isa LGRQuadrature
        R = [P[1] * SelectionMap([(q[2]+1)*(i-1)+1 
                for i in 1:q[1]+1], (q[1]+1)*(q[2]+1));
            P[2] * TensorProductMap2D(R_R[1], I, σ, σ_2); 
            P[3] * TensorProductMap2D(R_L[1], I, σ, σ_3)]
    else
        R = [P[1] * TensorProductMap2D(I, R_L[2], σ, σ_1); 
            P[2] * TensorProductMap2D(R_R[1], I, σ, σ_2); 
            P[3] * TensorProductMap2D(R_L[1], I, σ, σ_3)]
    end

    if approx_type isa ModalTensor
        V = warped_product(Tri(),approx_type.p, η_1D)
        V_plot = LinearMap(vandermonde(Tri(), 
            approx_type.p, reference_element.rstp...))
        new_approx_type = approx_type
    elseif approx_type isa NodalTensor
        V = LinearMap(I, (q[1]+1)*(q[2]+1))
        VDM_plot_1D = (vandermonde(Line(), q[1], equi_nodes(Line(),N_plot)),
            vandermonde(Line(), q[2], equi_nodes(Line(),N_plot)))
        V_plot = kron(VDM_plot_1D[1]/V_1D[1], VDM_plot_1D[2]/V_1D[2])
        new_approx_type = NodalTensor(min(q[1],q[2]))
    end

    return ReferenceApproximation(new_approx_type, size(V,2), 
        length(reference_element.wq), length(reference_element.wf),
        reference_element, D, V, R * V, R, Diagonal(w_η), 
        Diagonal(reference_element.wf), V_plot, reference_mapping)
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
    σ = permutedims(reshape(collect(1:N_q),q[1]+1,q[2]+1,q[3]+1), [3,2,1])

    D = (TensorProductMap3D(D_1D[1], I, I, σ, σ),
         TensorProductMap3D(I, D_1D[2], I, σ, σ),
         TensorProductMap3D(I, I, D_1D[3], σ, σ))

    # reference geometric factors for cube-to-tetrahedron mapping
    ref_geo_facs = reference_geometric_factors(Tet(),(η1, η2, η3))

    # 2D facet quadrature nodes and weights
    mortar_nodes, mortar_weights = quadrature(Tri(), facet_quadrature_rule)
    nodes_per_facet = length(mortar_nodes)
    
    # reference element data
    reference_element = RefElemData(Tet(), approx_type, mapping_degree,
        volume_quadrature_rule=volume_quadrature_rule,
        facet_quadrature_rule=facet_quadrature_rule,  Nplot=N_plot)

    σ_13 = permutedims(reshape(collect(1:nodes_per_facet),q[1]+1,1,q[3]+1),
        [3,2,1])
    σ_23 = permutedims(reshape(collect(1:nodes_per_facet),1,q[2]+1,q[3]+1),
        [3,2,1])
    σ_12 = permutedims(reshape(collect(1:nodes_per_facet),q[1]+1,q[2]+1,1),
        [3,2,1])

    # interpolation/extrapolation operators
    R = [TensorProductMap3D(I, R_L[2], I, σ, σ_13);
        TensorProductMap3D(R_R[1], I, I, σ, σ_23);
        TensorProductMap3D(R_L[1], I, I, σ, σ_23);
        TensorProductMap3D(I, I, R_L[3], σ, σ_12)]

    @unpack rstq, rstf, rstp, wq, wf = reference_element

    if approx_type isa ModalTensor
        V = LinearMap(vandermonde(Tet(),approx_type.p, rstq...))
        V_plot = LinearMap(vandermonde(Tet(), 
            approx_type.p, reference_element.rstp...))
        new_approx_type = approx_type
    elseif approx_type isa NodalTensor
        V = LinearMap(I, (q[1]+1)*(q[2]+1)*(q[3]+1))
        VDM_plot_1D = (vandermonde(Line(), q[1], equi_nodes(Line(),N_plot)),
            vandermonde(Line(), q[2], equi_nodes(Line(),N_plot)),
            vandermonde(Line(), q[3], equi_nodes(Line(),N_plot)) )
        V_plot = kron(VDM_plot_1D[1]/V_1D[1], VDM_plot_1D[2]/V_1D[2],
            VDM_plot_1D[3]/V_1D[3])
        new_approx_type = NodalTensor(min(q[1],q[2],q[3]))
    end

    return ReferenceApproximation(new_approx_type, size(V,2), 
        length(reference_element.wq), length(reference_element.wf),
        reference_element, D, V, R * V, R, Diagonal(w_η), 
        Diagonal(reference_element.wf), V_plot, 
        ReferenceMapping(ref_geo_facs...))
end