"""Duffy transform from the square to triangle"""
@inline function χ(::Tri, 
    η::Union{NTuple{2,Float64},NTuple{2,Vector{Float64}}})
    return (0.5.*(1.0 .+ η[1]).*(1.0 .- η[2]) .- 1.0, η[2])
end

"""Duffy transform from the cube to tetrahedron"""
@inline function χ(::Tet, 
    η::Union{NTuple{3,Float64},NTuple{3,Vector{Float64}}})

    (η1bar, η3bar) = χ(Tri(),(η[1], η[3]))
    (η2tilde, η3tilde) = χ(Tri(),(η[2], η3bar))
    return χ(Tri(), (η1bar, η2tilde))..., η3tilde
end

"""Geometric factors of the Duffy transform"""
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
    ::Tri; mortar::Bool=true, mapping_degree::Int=1, 
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

    if mortar 
        mortar_nodes, mortar_weights = quadrature(Line(), facet_quadrature_rule)

        P = (LinearMap(vandermonde(Line(),q[1],mortar_nodes)/V_1D[1]), 
            LinearMap(vandermonde(Line(),q[2],mortar_nodes)/V_1D[2]),
            LinearMap(vandermonde(Line(),q[2],mortar_nodes[end:-1:1])/V_1D[2]))
        
        reference_element = RefElemData(Tri(), Polynomial(), mapping_degree,
            quad_rule_vol=quadrature(Tri(),volume_quadrature_rule),
            quad_rule_face=(mortar_nodes,mortar_weights), Nplot=N_plot)
    else 
        P = (LinearMap(I, q[1]+1), LinearMap(I, q[2]+1),
            SelectionMap([i for i in q[2]+1:-1:1], q[2]+1))

        reference_element = RefElemData(Tri(), approx_type, mapping_degree;
            quadrature_rule=volume_quadrature_rule, Nplot=N_plot)
    end

    # ordering of facet nodes
    σ_1 = [i for i in 1:q[1]+1, j in 1:1]
    σ_2 = [j for i in 1:1, j in 1:q[2]+1]

    # interpolation/extrapolation operators
    if volume_quadrature_rule[2] isa LGRQuadrature
        R = [P[1] * SelectionMap([(q[2]+1)*(i-1)+1 
                for i in 1:q[1]+1], (q[1]+1)*(q[2]+1));
            P[2] * TensorProductMap2D(R_R[1], I, σ, σ_2); 
            P[3] * TensorProductMap2D(R_L[1], I, σ, σ_2)]
    else
        R = [P[1] * TensorProductMap2D(I, R_L[2], σ, σ_1); 
            P[2] * TensorProductMap2D(R_R[1], I, σ, σ_2); 
            P[3] * TensorProductMap2D(R_L[1], I, σ, σ_2)]
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
    else
        error("Invalid approximation type")
    end

    return ReferenceApproximation(new_approx_type, size(V,2), 
        length(reference_element.wq), length(reference_element.wf),
        reference_element, D, V, R * V, R, Diagonal(w_η), 
        Diagonal(reference_element.wf), V_plot, reference_mapping)
end

function ReferenceApproximation(
    approx_type::Union{NodalTensor,ModalTensor}, 
    ::Tet; mortar::Bool=true, mapping_degree::Int=1, N_plot::Int=10,
    volume_quadrature_rule=(LGQuadrature(approx_type.p),
        LGQuadrature(approx_type.p), LGQuadrature(approx_type.p)), 
    facet_quadrature_rule=(LGQuadrature(approx_type.p), 
        LGQuadrature(approx_type.p)))

    @unpack p = approx_type
    d = 3
    
    if mortar
        reference_element = RefElemData(Tet(), mapping_degree,
            quad_rule_vol=quadrature(Tet(), volume_quadrature_rule),
            quad_rule_face=quadrature(Tri(), facet_quadrature_rule),
            Nplot=N_plot)
    else
        reference_element = RefElemData(Tet(), approx_type, mapping_degree,
            quadrature_rule=(LGQuadrature(p), LGQuadrature(p), LGQuadrature(p)),  Nplot=N_plot)
    end

    @unpack rstq, rstf, rstp, wq, wf = reference_element
    
    VDM, ∇VDM... = basis(Tet(), p, rstq...) 
    ∇V = Tuple(LinearMap(∇VDM[m]) for m in 1:d)
    V = LinearMap(VDM)
    Vf = LinearMap(vandermonde(Tet(),p,rstf...))
    V_plot = LinearMap(vandermonde(Tet(), p, rstp...))
    W = Diagonal(wq)
    B = Diagonal(wf)
    P = inv(VDM' * Diagonal(wq) * VDM) * V' * W
    R = Vf * P
    D = Tuple(∇V[m] * P for m in 1:d)

    return ReferenceApproximation(approx_type, binomial(p+d, d), length(wq),
        length(wf), reference_element, D, V, Vf, R, W, B, V_plot, NoMapping())

end