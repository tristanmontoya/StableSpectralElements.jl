"""Duffy transform from the square to triangle"""
function χ(::CollapsedTri, 
    η::Union{NTuple{2,Float64},NTuple{2,Vector{Float64}}})
    return (0.5.*(1.0 .+ η[1]).*(1.0 .- η[2]) .- 1.0, η[2])
end

function init_face_data(::CollapsedTri, p, 
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

function init_face_data(::CollapsedTri, p, 
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

function ReferenceApproximation(approx_type::DGSEM, 
    elem_type::CollapsedTri;
    volume_quadrature_rule::NTuple{2,AbstractQuadratureRule}=(
        LGQuadrature(),LGQuadrature()),
    facet_quadrature_rule::NTuple{2,AbstractQuadratureRule}=(
        LGQuadrature(),LGQuadrature()),
    reference_mapping=ChainRuleMapping(),
    mapping_degree::Int=1, N_plot::Int=10)

    @unpack p = approx_type
    d = 2
    N_p = (p+1)*(p+1)
    N_q = N_p
    N_f = 3*(p+1)

    # reference element data structure
    fv = face_vertices(Tri())
    r,s = nodes(Tri(), mapping_degree)
    Fmask = hcat(find_face_nodes(Tri(), r, s)...)
    VDM, Vr, Vs = basis(Tri(), mapping_degree, r, s)
    Dr = Vr / VDM
    Ds = Vs / VDM
    r1, s1 = nodes(Tri(), 1)
    V1 = vandermonde(Tri(), 1, r, s) / vandermonde(Tri(), 1, r1, s1)
    rq, sq, wq = quadrature(CollapsedTri(), volume_quadrature_rule, (p+1, p+1))
    Vq = vandermonde(Tri(), mapping_degree, rq, sq) / VDM
    M = Vq' * diagm(wq) * Vq
    Pq = M \ (Vq' * diagm(wq))
    rp, sp = equi_nodes(Tri(), N_plot)
    Vp = vandermonde(Tri(), mapping_degree, rp, sp) / VDM
    rf, sf, wf, nrJ, nsJ = init_face_data(elem_type, p, facet_quadrature_rule)
    Vf = vandermonde(Tri(), mapping_degree, rf, sf) / VDM
    reference_element = RefElemData(Tri(), Polynomial(), mapping_degree, fv, V1,
                       tuple(r, s), VDM, vec(Fmask),
                       N_plot, tuple(rp, sp), Vp,
                       tuple(rq, sq), wq, Vq,
                       tuple(rf, sf), wf, 
                       Vf, tuple(nrJ, nsJ),
                       M, Pq, (Dr, Ds),
                       M \  (Vf' * diagm(wf)))
    @unpack rstp, rstq = reference_element

    # one-dimensional operators
    rd_1 = RefElemData(Line(), mapping_degree,
        quad_rule_vol=quadrature(Line(), 
        volume_quadrature_rule[1], p+1), Nplot=1) # horizontal
    rd_2 = RefElemData(Line(), mapping_degree,
        quad_rule_vol=quadrature(Line(), 
        volume_quadrature_rule[2], p+1), Nplot=1) # vertical

    VDM_1, ∇VDM_1 = basis(Line(), p, rd_1.rstq[1])
    D_1 = ∇VDM_1 / VDM_1
    VDM_2, ∇VDM_2 = basis(Line(), p, rd_2.rstq[1])
    D_2 = ∇VDM_2 / VDM_2

    # ordering
    sigma = [(p+1)*(i-1) + j for i in 1:p+1, j in 1:p+1]

    # extrapolation operators (not most efficient for e.g. LGL) 
    R_1 = vandermonde(Line(),p, rd_1.rstf[1]) / VDM_1 # horizontal
    R_2 = vandermonde(Line(),p, rd_2.rstf[1]) / VDM_2 # vertical
    R_B = R_2[1:1,:]
    R_R = R_1[2:2,:]
    R_L = R_1[1:1,:]

    # if a 1D extrapolation can be used along each line of nodes
    if (typeof(volume_quadrature_rule[1])==
            typeof(facet_quadrature_rule[1])) &&
        (typeof(volume_quadrature_rule[2])==
            typeof(facet_quadrature_rule[2]))    
        R = [TensorProductMap(
                R_B, I, sigma, [j for i in 1:1, j in 1:p+1]); # bottom
            TensorProductMap(I, R_R, 
                sigma, [i for i in 1:p+1, j in 1:1]); # right 
            TensorProductMap(I, R_L,  
                sigma, [i for i in p+1:-1:1, j in 1:1])] # left, downwards
    else 
        R = [LinearMap(vandermonde(Line(), p, r_1d_1) / VDM_1) *
            TensorProductMap( R_B, I, sigma, [j for i in 1:1, j in 1:p+1]);
        LinearMap(vandermonde(Line(), p, r_1d_2) / VDM_2) * 
            TensorProductMap(I, R_R, sigma, [i for i in 1:p+1, j in 1:1]); 
        LinearMap((vandermonde(Line(), p, r_1d_2) / VDM_2)[end:-1:1,:]) *   
            TensorProductMap(I, R_L, sigma,
                [i for i in p+1:-1:1, j in 1:1])]
    end
            
    W = LinearMap(Diagonal(wq))
    B = LinearMap(Diagonal(wf))

    # differentiation on the square
    D_η = (TensorProductMap(I, D_1, sigma, sigma), 
        TensorProductMap(D_2, I, sigma, sigma))
    
    if reference_mapping isa ChainRuleMapping
        η1, η2, _ = quadrature(Quad(), volume_quadrature_rule, (p+1, p+1))
        D = (Diagonal((x-> 2.0/(1.0-x)).(η2))*D_η[1], 
        Diagonal((x -> (1.0+x)).(η1) ./ (x -> (1.0-x)).(η2))*D_η[1] + D_η[2])
    else
        D = D_η
    end

    V_plot = LinearMap(vandermonde(Tri(), p, rstp...) / 
        vandermonde(Tri(), p, rstq...))
    V = LinearMap(I, N_q)
    P = LinearMap(I, N_q)
    ADVs = Tuple(-1.0*Diagonal(wq)*D[m] for m in 1:d)
    ADVw = Tuple(D[m]' * W for m in 1:d)

    return ReferenceApproximation{d}(approx_type, N_p, N_q, N_f, 
        reference_element, D, V, R, P, W, B, ADVs, ADVw, V_plot,
        reference_mapping)
end

"""Apply the transformation of Crean et al. (2018) to χ"""
function make_sbp_operator(::CollapsedTri, η::NTuple{2,Vector{Float64}}, 
    w_η::Vector{Float64}, wf::Vector{Float64}, 
    D_η::NTuple{2,<:LinearMap{Float64}}, R::LinearMap{Float64},
    nrstJ::NTuple{2,Vector{Float64}})

    S = (0.5*(Diagonal(w_η)*D_η[1] - D_η[1]'*Diagonal(w_η)),
        0.5*((Diagonal(w_η .* (x -> 0.5*(1+x)).(η[1]))*D_η[1] -
            D_η[1]'*Diagonal(w_η .* (x -> 0.5*(1+x)).(η[1]))) +
            (Diagonal(w_η .* (x -> 0.5*(1-x)).(η[2]))*D_η[2] - 
            D_η[2]'*(Diagonal(w_η .* (x -> 0.5*(1-x)).(η[2])))))
        )
    return Tuple(S[m] + 0.5*R'*Diagonal(wf .* nrstJ[m])*R for m in 1:2)
end