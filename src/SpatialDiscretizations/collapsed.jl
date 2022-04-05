abstract type AbstractTransformType end
struct SBPTransform <: AbstractTransformType end
struct NaiveTransform <: AbstractTransformType end

function χ(::CollapsedTri, 
    η::Union{NTuple{2,Float64},NTuple{2,Vector{Float64}}})
    return (0.5.*(1.0 .+ η[1]).*(1.0 .- η[2]) .- 1.0, η[2])
end

function make_standard_operator(::CollapsedTri, η::NTuple{2,Vector{Float64}}, 
    w_η::Vector{Float64}, wf::Vector{Float64}, 
    D_η::NTuple{2,<:LinearMap{Float64}})

    return (Diagonal(w_η)*D_η[1], 
            Diagonal(w_η .* (x -> 0.5*(1+x)).(η[1]))*D_η[1] + 
            Diagonal(w_η .* (x -> 0.5*(1-x)).(η[2]))*D_η[2])
end 

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

function ReferenceApproximation(approx_type::DGSEM, 
    elem_type::CollapsedTri;
    transform_type::AbstractTransformType=SBPTransform(),
    volume_quadrature_rule::NTuple{2,AbstractQuadratureRule}=(
        LGQuadrature(),LGQuadrature()),
    mapping_degree::Int=1, N_plot::Int=10)

    @unpack p = approx_type
    d = 2
    N_p = (p+1)*(p+1)
    N_q = N_p
    N_f = 3*(p+1)

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

    # extrapolation (this only will work for LG volume rule)
    # facet quadrature is currently defaulted to LG
    R_1 = vandermonde(Line(),p, rd_1.rstf[1]) / VDM_1
    R_2 = vandermonde(Line(),p, rd_2.rstf[1]) / VDM_2
    R_L = R_1[1:1,:] # get only one row
    R_R = R_1[2:2,:]
    R_B = R_2[1:1,:]
    R = [TensorProductMap(
            R_B, I, sigma, [j for i in 1:1, j in 1:p+1]); # bottom
        TensorProductMap(I, R_R, 
            sigma, [i for i in 1:p+1, j in 1:1]); # right 
        TensorProductMap(I, R_L,  
            sigma, [i for i in p+1:-1:1, j in 1:1])] # left, downwards

    # triangular reference element data
    reference_element = RefElemData(Tri(), 
        mapping_degree, quad_rule_vol=quadrature(
            elem_type, volume_quadrature_rule, (p+1, p+1)),
        quad_rule_face=quad_nodes(face_type(Tri()), p),
        Nplot=N_plot)
    @unpack rstp, rstq, rstf, wq, wf, nrstJ = reference_element

    # triangular operators
    W = LinearMap(Diagonal(wq))
    B = LinearMap(Diagonal(wf))
    η1, η2, w_η = quadrature(Quad(), 
        volume_quadrature_rule, (p+1, p+1))
    η = (η1, η2)

    if transform_type isa SBPTransform
        Q = make_sbp_operator(elem_type, η, w_η, wf,
            (TensorProductMap(I, D_1, sigma, sigma), 
            TensorProductMap(D_2, I, sigma, sigma)), R, nrstJ)
    else
        Q = make_standard_operator(elem_type, η, w_η, wf, 
            (TensorProductMap(I,  D_1, sigma, sigma), 
            TensorProductMap(D_2, I, sigma, sigma)))
    end

    D = Tuple(inv(Diagonal(wq))*Q[m] for m in 1:d)
 
    V_plot = LinearMap(vandermonde(Tri(), p, rstp...) / 
        vandermonde(Tri(), p, rstq...))
    V = LinearMap(I, N_q)
    P = LinearMap(I, N_q)
    ADVs = Tuple(-1.0*Q[m] for m in 1:d)
    ADVw = Tuple(D[m]' * W for m in 1:d)

    return ReferenceApproximation{d}(approx_type, N_p, N_q, N_f, 
        reference_element, D, V, R, P, W, B, ADVs, ADVw, V_plot)
end