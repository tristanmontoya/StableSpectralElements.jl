function RefElemData(elem::Tri,  
    ::Union{ModalTensor,NodalTensor}, N;
    quadrature_rule::NTuple{2,AbstractQuadratureRule}=(
        LGQuadrature(),LGRQuadrature()),
        Nplot=10)

    fv = face_vertices(elem) # set faces for triangle

    # Construct matrices on reference elements
    r,s = nodes(elem, N)
    Fmask = hcat(find_face_nodes(elem, r, s)...)

    VDM, Vr, Vs = basis(elem, N, r, s)
    Dr = Vr / VDM
    Ds = Vs / VDM

    # low order interpolation nodes
    r1, s1 = nodes(elem, 1)
    V1 = vandermonde(elem, 1, r, s) / vandermonde(elem, 1, r1, s1)

    r_1d_1, w_1d_1 = quadrature(face_type(elem), quadrature_rule[1], N+1)
    r_1d_2, w_1d_2 = quadrature(face_type(elem), quadrature_rule[2], N+1)
    
    wf = [w_1d_1; w_1d_2; w_1d_2[end:-1:1]]
    (rf, sf) = ([r_1d_1; -r_1d_2; -ones(size(r_1d_2))], 
        [-ones(size(r_1d_1)); r_1d_2; r_1d_2[end:-1:1]])
    nrJ = [zeros(size(r_1d_1)); ones(size(r_1d_2)); -ones(size(r_1d_2))]
    nsJ = [-ones(size(r_1d_1)); ones(size(r_1d_2)); zeros(size(r_1d_2))]
    
    rq, sq, wq =  quadrature(elem, quadrature_rule, (N+1,N+1))

    Vq = vandermonde(elem, N, rq, sq) / VDM
    M = Vq' * diagm(wq) * Vq
    Pq = M \ (Vq' * diagm(wq))

    Vf = vandermonde(elem, N, rf, sf) / VDM # interpolates from nodes to face nodes
    LIFT = M \ (Vf' * diagm(wf)) # lift matrix used in rhs evaluation

    # plotting nodes
    rp, sp = χ(Tri(),equi_nodes(Quad(),Nplot)) 
    Vp = vandermonde(elem, N, rp, sp) / VDM

    return RefElemData(elem, Polynomial(), N, fv, V1,
                    tuple(r, s), VDM, vec(Fmask),
                    Nplot, tuple(rp, sp), Vp,
                    tuple(rq, sq), wq, Vq,
                    tuple(rf, sf), wf, Vf, tuple(nrJ, nsJ),
                    M, Pq, (Dr, Ds), LIFT)
end