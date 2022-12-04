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
    return χ(Tri(), η1bar, η2tilde)..., η3tilde
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

function ReferenceApproximation(
    approx_type::Union{NodalTensor,ModalTensor}, 
    ::Tri; quadrature_rule::NTuple{2,AbstractQuadratureRule{1}}=(
        LGQuadrature(),LGRQuadrature()), mortar::Bool=true,
    mapping_degree::Int=1, N_plot::Int=10)

    @unpack p = approx_type

    # one-dimensional operators
    nodes_1D = Tuple(quadrature(Line(),quadrature_rule[m],p+1)[1] for m in 1:2)
    V_1D = Tuple(vandermonde(Line(),p,nodes_1D[m]) for m in 1:2)
    D_1D = Tuple(grad_vandermonde(Line(),p,nodes_1D[m]) / V_1D[m] for m in 1:2)
    R_L = Tuple(vandermonde(Line(),p,[-1.0]) / V_1D[m] for m in 1:2)
    R_R = Tuple(vandermonde(Line(),p,[1.0]) / V_1D[m] for m in 1:2)

    # nodes and weights on the square
    η1, η2, w_η = quadrature(Quad(), quadrature_rule, (p+1, p+1))

    # differentiation operator on the square
    σ = [(p+1)*(i-1) + j for i in 1:p+1, j in 1:p+1]
    D = (TensorProductMap2D(D_1D[1], I, σ, σ), 
        TensorProductMap2D(I, D_1D[2], σ, σ))

    # reference geometric factors for square-to-triangle mapping
    reference_mapping = ReferenceMapping(
        reference_geometric_factors(Tri(),(η1,η2))...)

    if mortar 
        mortar_nodes, mortar_weights = quad_nodes(Line(),p)

        P = (LinearMap(vandermonde(Line(),p,mortar_nodes) / V_1D[1]), 
            LinearMap(vandermonde(Line(),p,mortar_nodes) / V_1D[2]),
            LinearMap(vandermonde(Line(),p,mortar_nodes[end:-1:1]) /  V_1D[2]))
        
        reference_element = RefElemData(Tri(), Polynomial(), mapping_degree,
            quad_rule_vol=quadrature(Tri(), quadrature_rule, (p+1, p+1)),
            quad_rule_face=(mortar_nodes,mortar_weights), Nplot=N_plot)
    else 
        P = (LinearMap(I, p+1), LinearMap(I, p+1),
            SelectionMap([i for i in p+1:-1:1], p+1))

        reference_element = RefElemData(Tri(), approx_type, mapping_degree;
            quadrature_rule=quadrature_rule, Nplot=N_plot)
    end

    # ordering of facet nodes
    (σ_1, σ_2) = ([i for i in 1:p+1, j in 1:1], [j for i in 1:1, j in 1:p+1])

    # interpolation/extrapolation operators
    if quadrature_rule[2] isa LGRQuadrature
        R = [P[1] * SelectionMap([(p+1)*(i-1)+1 for i in 1:p+1], (p+1)*(p+1));
            P[2] * TensorProductMap2D(R_R[1], I, σ, σ_2); 
            P[3] * TensorProductMap2D(R_L[1], I, σ, σ_2)]
    else
        R = [P[1] * TensorProductMap2D(I, R_L[2], σ, σ_1); 
            P[2] * TensorProductMap2D(R_R[1], I, σ, σ_2); 
            P[3] * TensorProductMap2D(R_L[1], I, σ, σ_2)]
    end

    if approx_type isa ModalTensor
        V = warped_product(Tri(),p, (nodes_1D[1],nodes_1D[2]))
        V_plot = LinearMap(vandermonde(Tri(), p, reference_element.rstp...))
    else
        V = LinearMap(I, (p+1)*(p+1))
        V_plot = LinearMap(vandermonde(Quad(), p, reference_element.rstp...) / 
            kron(V_1D[1], V_1D[2]))
    end

    return ReferenceApproximation(approx_type, size(V,2), (p+1)*(p+1), 3*(p+1), 
        reference_element, D, V, R * V, R, Diagonal(w_η), 
        Diagonal(reference_element.wf), V_plot, reference_mapping)
end

function ReferenceApproximation(
    approx_type::Union{NodalTensor,ModalTensor}, 
    ::Tet; quadrature_rule::NTuple{3,AbstractQuadratureRule{1}}=(
        LGQuadrature(),LGQuadrature(), LGQuadrature()),
    mapping_degree::Int=1, N_plot::Int=10)

    @unpack p = approx_type
    d = 3
    
    reference_element = RefElemData(Tet(), approx_type, mapping_degree,
        quadrature_rule=quadrature_rule, Nplot=N_plot)

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