function RefElemData(elem::Tri,
        approx_type::Union{ModalTensor, NodalTensor},
        N;
        volume_quadrature_rule = (LGQuadrature(approx_type.p),
            LGQuadrature(approx_type.p)),
        facet_quadrature_rule = LGQuadrature(approx_type.p),
        Nplot = 10)
    fv = face_vertices(elem) # set faces for triangle

    # Construct matrices on reference elements
    r, s = nodes(elem, N)
    Fmask = hcat(find_face_nodes(elem, r, s)...)

    VDM, Vr, Vs = basis(elem, N, r, s)
    Dr = Vr / VDM
    Ds = Vs / VDM

    # low order interpolation nodes
    r1, s1 = nodes(elem, 1)
    V1 = vandermonde(elem, 1, r, s) / vandermonde(elem, 1, r1, s1)

    r_1d, w_1d = quadrature(Line(), facet_quadrature_rule)

    wf = [w_1d; w_1d; w_1d]
    (rf, sf) = ([r_1d; -r_1d; -ones(size(r_1d))], [-ones(size(r_1d)); r_1d; r_1d])
    nrJ = [zeros(size(r_1d)); ones(size(r_1d)); -ones(size(r_1d))]
    nsJ = [-ones(size(r_1d)); ones(size(r_1d)); zeros(size(r_1d))]

    rq, sq, wq = quadrature(elem, volume_quadrature_rule)

    Vq = vandermonde(elem, N, rq, sq) / VDM
    M = Vq' * diagm(wq) * Vq
    Pq = M \ (Vq' * diagm(wq))

    Vf = vandermonde(elem, N, rf, sf) / VDM # 
    LIFT = M \ (Vf' * diagm(wf))

    # plotting nodes
    rp, sp = χ(Tri(), equi_nodes(Quad(), Nplot))
    Vp = vandermonde(elem, N, rp, sp) / VDM

    return RefElemData(elem,
        Polynomial(),
        N,
        fv,
        V1,
        tuple(r, s),
        VDM,
        vec(Fmask),
        tuple(rp, sp),
        Vp,
        tuple(rq, sq),
        wq,
        Vq,
        tuple(rf, sf),
        wf,
        Vf,
        tuple(nrJ, nsJ),
        M,
        Pq,
        (Dr, Ds),
        LIFT)
end

function RefElemData(elem::Tet,
        approx_type::Union{ModalTensor, NodalTensor},
        N;
        volume_quadrature_rule = (LGQuadrature(approx_type.p),
            LGQuadrature(approx_type.p),
            LGQuadrature(approx_type.p)),
        facet_quadrature_rule = (LGQuadrature(approx_type.p),
            LGQuadrature(approx_type.p)),
        Nplot = 10)
    fv = face_vertices(elem)

    # Construct matrices on reference elements
    r, s, t = nodes(elem, N)
    face_nodes = find_face_nodes(elem, r, s, t, 100 * eps())
    Fmask = hcat(face_nodes...)
    VDM, Vr, Vs, Vt = basis(elem, N, r, s, t)
    Dr, Ds, Dt = (A -> A / VDM).((Vr, Vs, Vt))

    # low order interpolation nodes
    r1, s1, t1 = nodes(elem, 1)
    V1 = vandermonde(elem, 1, r, s, t) / vandermonde(elem, 1, r1, s1, t1)

    r_2d, s_2d, w_2d = quadrature(Tri(),
        (facet_quadrature_rule[1], facet_quadrature_rule[2]))

    (ee, zz) = (ones(size(r_2d)), zeros(size(r_2d)))

    rf = [r_2d; -(ee + r_2d + s_2d); -ee; r_2d]
    sf = [-ee; r_2d; r_2d; s_2d]
    tf = [s_2d; s_2d; s_2d; -ee]

    wf = [w_2d; w_2d; w_2d; w_2d]
    nrJ = [zz; ee; -ee; zz]
    nsJ = [-ee; ee; zz; zz]
    ntJ = [zz; ee; zz; -ee]

    rq, sq, tq, wq = quadrature(Tet(), volume_quadrature_rule)
    Vq = vandermonde(elem, N, rq, sq, tq) / VDM
    M = Vq' * diagm(wq) * Vq
    Pq = M \ (Vq' * diagm(wq))

    Vf = vandermonde(elem, N, rf, sf, tf) / VDM
    LIFT = M \ (Vf' * diagm(wf))

    # plotting nodes
    rp, sp, tp = equi_nodes(elem, Nplot)
    Vp = vandermonde(elem, N, rp, sp, tp) / VDM

    return RefElemData(elem,
        Polynomial(),
        N,
        fv,
        V1,
        tuple(r, s, t),
        VDM,
        vec(Fmask),
        tuple(rp, sp, tp),
        Vp,
        tuple(rq, sq, tq),
        wq,
        Vq,
        tuple(rf, sf, tf),
        wf,
        Vf,
        tuple(nrJ, nsJ, ntJ),
        M,
        Pq,
        (Dr, Ds, Dt),
        LIFT)
end
