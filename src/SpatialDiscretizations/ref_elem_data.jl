function RefElemData(elem::Tri,  
    approx_type::Union{ModalTensor,NodalTensor}, N; quadrature_rule=(
        LGQuadrature(approx_type.p),LGRQuadrature(approx_type.p)),
        Nplot=10)

    @unpack p = approx_type

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

    r_1d_1, w_1d_1 = quadrature(Line(), quadrature_rule[1])
    r_1d_2, w_1d_2 = quadrature(Line(), quadrature_rule[2])
    
    wf = [w_1d_1; w_1d_2; w_1d_2[end:-1:1]]
    (rf, sf) = ([r_1d_1; -r_1d_2; -ones(size(r_1d_2))], 
        [-ones(size(r_1d_1)); r_1d_2; r_1d_2[end:-1:1]])
    nrJ = [zeros(size(r_1d_1)); ones(size(r_1d_2)); -ones(size(r_1d_2))]
    nsJ = [-ones(size(r_1d_1)); ones(size(r_1d_2)); zeros(size(r_1d_2))]
    
    rq, sq, wq =  quadrature(elem, quadrature_rule)

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

function RefElemData(elem::Tet,  
    approx_type::Union{ModalTensor,NodalTensor}, N;
    quadrature_rule=(LGQuadrature(approx_type.p),LGQuadrature(approx_type.p),
        JGRQuadrature(approx_type.p)), Nplot=10)

    @unpack p = approx_type

    fv = face_vertices(elem) 

    # Construct matrices on reference elements
    r, s, t = nodes(elem, N)
    Fmask = hcat(find_face_nodes(elem, r, s, t)...)
    VDM, Vr, Vs, Vt = basis(elem, N, r, s, t)
    Dr, Ds, Dt = (A -> A / VDM).((Vr, Vs, Vt))

    # low order interpolation nodes
    r1, s1, t1 = nodes(elem, 1)
    V1 = vandermonde(elem, 1, r, s, t) / vandermonde(elem, 1, r1, s1, t1)

    r1, s1, w1 = quadrature(Tri(), (quadrature_rule[1], quadrature_rule[3]))
    r2, s2, w2 = quadrature(Tri(),  (quadrature_rule[2], quadrature_rule[3]))
    r3, s3, w3 = r2, s2, w2
    r4, s4, w4 = quadrature(Tri(), (quadrature_rule[1], quadrature_rule[2]))

    (e1, z1) = (ones(size(r1)), zeros(size(r1)))
    (e2, z2) = (ones(size(r2)), zeros(size(r2)))
    (e3, z3) = (ones(size(r3)), zeros(size(r3)))
    (e4, z4) = (ones(size(r4)), zeros(size(r4)))

    rf = [r1; -(e2 + r2 + s2); -e3; r4]
    sf = [-e1; r2; r3; s4]
    tf = [s1; s2; s3; -e4]
    wf = [w1; w2; w3; w4]
    nrJ = [z1; e2; -e3; z4]
    nsJ = [-e1; e2; z3; z4]
    ntJ = [z1; e2; z3; -e4]

    # quadrature nodes - build from 1D nodes.
    rq, sq, tq, wq = quad_nodes(Tet(), N) #temporarily do this
    Vq = vandermonde(elem, N, rq, sq, tq) / VDM
    M = Vq' * diagm(wq) * Vq
    Pq = M \ (Vq' * diagm(wq))

    Vf = vandermonde(elem, N, rf, sf, tf) / VDM
    LIFT = M \ (Vf' * diagm(wf))

    # plotting nodes
    rp, sp, tp = equi_nodes(elem, Nplot)
    Vp = vandermonde(elem, N, rp, sp, tp) / VDM

    return RefElemData(elem, Polynomial(), N, fv, V1,
                        tuple(r, s, t), VDM, vec(Fmask),
                        Nplot, tuple(rp, sp, tp), Vp,
                        tuple(rq, sq, tq), wq, Vq,
                        tuple(rf, sf, tf), wf, Vf, tuple(nrJ, nsJ, ntJ),
                        M, Pq, (Dr, Ds, Dt), LIFT)
end