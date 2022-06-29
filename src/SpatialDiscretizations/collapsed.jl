"""Nodal spectral element method in collapsed coordinates"""
struct CollapsedSEM <:AbstractApproximationType
    p::Int  # polynomial degree
end

"""Nodal spectral element method projected onto modal basis"""
struct CollapsedModal <:AbstractApproximationType
    p::Int  # polynomial degree
end

"""Duffy transform from the square to triangle"""
function χ(::Tri, 
    η::Union{NTuple{2,Float64},NTuple{2,Vector{Float64}}})
    return (0.5.*(1.0 .+ η[1]).*(1.0 .- η[2]) .- 1.0, η[2])
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

function init_face_data(::Tri, p, 
    facet_quadrature_rule::Tuple{LegendreQuadrature,JacobiQuadrature})
    r_1d_1, w_1d_1 = quadrature(face_type(Tri()), 
    facet_quadrature_rule[1], p+1)
    r_1d_2, w_1d_2 = quadrature(face_type(Tri()), 
        facet_quadrature_rule[2], p+1)
    wf = [w_1d_1; (x->2.0/(1.0-x)).(r_1d_2) .* w_1d_2; 
        ((x->2.0/(1.0-x)).(r_1d_2) .* w_1d_2)[end:-1:1]]
    (rf, sf) = ([r_1d_1; -r_1d_2; -ones(size(r_1d_2))], 
        [-ones(size(r_1d_1)); r_1d_2; r_1d_2[end:-1:1]])
    nrJ = [zeros(size(r_1d_1)); ones(size(r_1d_2)); -ones(size(r_1d_2))]
    nsJ = [-ones(size(r_1d_1)); ones(size(r_1d_2)); zeros(size(r_1d_2))]
    return rf,sf,wf,nrJ,nsJ
end

function init_face_data(::Tri, p, 
    facet_quadrature_rule::NTuple{2,LegendreQuadrature})
    r_1d_1, w_1d_1 = quadrature(face_type(Tri()), 
    facet_quadrature_rule[1], p+1)
    r_1d_2, w_1d_2 = quadrature(face_type(Tri()), 
        facet_quadrature_rule[2], p+1)
    wf = [w_1d_1; w_1d_2; w_1d_2[end:-1:1]]
    (rf, sf) = ([r_1d_1; -r_1d_2; -ones(size(r_1d_2))], 
        [-ones(size(r_1d_1)); r_1d_2; r_1d_2[end:-1:1]])
    nrJ = [zeros(size(r_1d_1)); ones(size(r_1d_2)); -ones(size(r_1d_2))]
    nsJ = [-ones(size(r_1d_1)); ones(size(r_1d_2)); zeros(size(r_1d_2))]
    return rf,sf,wf,nrJ,nsJ
end

function ReferenceApproximation(
    approx_type::Union{CollapsedSEM,CollapsedModal}, 
    elem_type::Tri;
    volume_quadrature_rule::NTuple{2,AbstractQuadratureRule}=(
        LGQuadrature(),LGRQuadrature()),
    facet_quadrature_rule::NTuple{2,AbstractQuadratureRule}=(
        LGQuadrature(),LGRQuadrature()),
    mapping_degree::Int=1, N_plot::Int=10)

    @unpack p = approx_type
    N_q = (p+1)*(p+1)
    N_f = 3*(p+1)

    # set up quadrature rules and nodes
    rq, sq, wq = quadrature(Tri(), volume_quadrature_rule, (p+1, p+1))
    rf, sf, wf, nrJ, nsJ = init_face_data(elem_type, p, facet_quadrature_rule)
    rp, sp = χ(Tri(),equi_nodes(Quad(),N_plot)) 

    # set up mapping nodes and interpolation matrices
    r,s = nodes(Tri(), mapping_degree)  
    VDM, Vr, Vs = basis(Tri(), mapping_degree, r, s)
    # from mapping interpolation nodes to quadrature nodes
    Vq_map = vandermonde(Tri(), mapping_degree, rq, sq) / VDM
    Vf_map = vandermonde(Tri(), mapping_degree, rf, sf) / VDM
    # reference element data structure from StartUpDG
    reference_element = RefElemData(Tri(), Polynomial(), mapping_degree,
                    face_vertices(Tri()), 
                    vandermonde(Tri(), 1, r, s)/
                        vandermonde(Tri(), 1, nodes(Tri(), 1)...),
                    tuple(r, s), VDM, 
                    vec(hcat(find_face_nodes(Tri(), r, s)...)),
                    N_plot, (rp, sp),
                    vandermonde(Tri(), mapping_degree, rp, sp) / VDM,
                    tuple(rq, sq), wq, Vq_map, tuple(rf, sf), wf, Vf_map,
                    tuple(nrJ, nsJ),  Vq_map' * diagm(wq) * Vq_map, 
                    ( Vq_map' * diagm(wq) * Vq_map) \ (Vq_map' * diagm(wq)), 
                    (Vr / VDM, Vs / VDM), 
                    (Vq_map' * diagm(wq) * Vq_map) \  (Vf_map' * diagm(wf)))

    # one-dimensional operators
    rd_1 = RefElemData(Line(), p,
        quad_rule_vol=quadrature(Line(), 
        volume_quadrature_rule[1], p+1), Nplot=N_plot) # horizontal
    rd_2 = RefElemData(Line(), p,
        quad_rule_vol=quadrature(Line(), 
        volume_quadrature_rule[2], p+1), Nplot=N_plot) # vertical

    VDM_1, ∇VDM_1 = basis(Line(), p, rd_1.rstq[1])
    D_1 = ∇VDM_1 / VDM_1
    VDM_2, ∇VDM_2 = basis(Line(), p, rd_2.rstq[1])
    D_2 = ∇VDM_2 / VDM_2
    R_1 = vandermonde(Line(),p, rd_1.rstf[1]) / VDM_1 # horizontal
    R_2 = vandermonde(Line(),p, rd_2.rstf[1]) / VDM_2 # vertical

    # ordering
    sigma = [(p+1)*(i-1) + j for i in 1:p+1, j in 1:p+1]

    # if a 1D extrapolation can be used along each line of nodes
    if (typeof(volume_quadrature_rule[1]) ==
            typeof(facet_quadrature_rule[1])) &&
        (typeof(volume_quadrature_rule[2]) ==
            typeof(facet_quadrature_rule[2]))

        if volume_quadrature_rule[2] isa LGRQuadrature
            bottom_map = SelectionMap([i for i in 1:p+1], N_q)
        else
            bottom_map =TensorProductMap(
                R_2[1:1,:], I, sigma, [j for i in 1:1, j in 1:p+1])
        end

        R = [bottom_map; # bottom, left to right
            TensorProductMap( # right, upwards
                I, R_1[2:2,:], sigma, [i for i in 1:p+1, j in 1:1]); 
            TensorProductMap( # left, downwards
                I, R_1[1:1,:],  sigma, [i for i in p+1:-1:1, j in 1:1])]
                
    # otherwise use successive 1D extrapolations
    else 
        r_1d_1, _ = quadrature(face_type(Tri()), 
            facet_quadrature_rule[1], p+1)
        r_1d_2, _ = quadrature(face_type(Tri()), 
            facet_quadrature_rule[2], p+1)
            
        R = [LinearMap(vandermonde(Line(), p, r_1d_1) / rd_1.VDM) *
            TensorProductMap(R_2[1:1,:], I, sigma, 
                [j for i in 1:1, j in 1:p+1]);
        LinearMap(vandermonde(Line(), p, r_1d_2) / rd_2.VDM) * 
            TensorProductMap(I, R_1[2:2,:], sigma, 
                [i for i in 1:p+1, j in 1:1]); 
        LinearMap((vandermonde(Line(), p, r_1d_2) / rd_1.VDM)[end:-1:1,:]) *   
            TensorProductMap(I, R_1[1:1,:], sigma,
                [i for i in p+1:-1:1, j in 1:1])]
    end

    # differentiation on the square
    D = (TensorProductMap(I, D_1, sigma, sigma), 
        TensorProductMap(D_2, I, sigma, sigma))

    # construct mapping to triangle
    η1, η2, w_η = quadrature(Quad(), volume_quadrature_rule, (p+1, p+1))
    B = LinearMap(Diagonal(wf))
    W = LinearMap(Diagonal(w_η))
    reference_mapping = ReferenceMapping(
        reference_geometric_factors(Tri(),(η1,η2))...)

    # construct modal or nodal scheme
    @unpack rstp, rstq = reference_element
    if approx_type isa CollapsedModal
        N_p = binomial(p+2, 2)
        V_modal = vandermonde(Tri(), p, rstq...)
        V_plot = LinearMap(vandermonde(Tri(), p, rstp...))
        V = LinearMap(V_modal)
        inv_M = LinearMap(inv(V_modal' * Diagonal(wq) * V_modal))
        P = inv_M * V' * W
        Vf = R * V
        ADVw = Tuple(V' * D[m]' * W for m in 1:2)
    else
        N_p = N_q
        V_plot = LinearMap(vandermonde(Quad(), p, rstp...) / 
            kron(rd_1.VDM, rd_2.VDM))
        V = LinearMap(I, N_q)
        P = LinearMap(I, N_q)
        Vf = R
        ADVw = Tuple(D[m]' * W for m in 1:2)
    end

    return ReferenceApproximation{2}(approx_type, N_p, N_q, N_f, 
        reference_element, D, V, Vf, R, P, W, B, ADVw, V_plot,
        reference_mapping)
end