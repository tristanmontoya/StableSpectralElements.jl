"""Duffy transform from the square to triangle"""
function χ(::CollapsedTri, 
    η::Union{NTuple{2,Float64},NTuple{2,Vector{Float64}}})
    return (0.5.*(1.0 .+ η[1]).*(1.0 .- η[2]) .- 1.0, η[2])
end

function ReferenceApproximation(approx_type::DGSEM, 
    elem_type::CollapsedTri;
    volume_quadrature_rule::NTuple{2,AbstractQuadratureRule}=(
        LGQuadrature(),LGQuadrature()),
    facet_quadrature_rule::AbstractQuadratureRule=LGQuadrature(),
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

    # extrapolation operators (not most efficient)
    R_1 = vandermonde(Line(),p, rd_1.rstf[1]) / VDM_1 # horizontal
    R_2 = vandermonde(Line(),p, rd_2.rstf[1]) / VDM_2 # vertical
    R_B = R_2[1:1,:]
    R_R = R_1[2:2,:]
    R_L = R_1[1:1,:]

    # if a 1D extrapolation can be used along each line of nodes
    if (typeof(volume_quadrature_rule[1]) == typeof(facet_quadrature_rule)) &&
        (typeof(volume_quadrature_rule[2]) == typeof(facet_quadrature_rule))    
        R = [TensorProductMap(
            R_B, I, sigma, [j for i in 1:1, j in 1:p+1]); # bottom
        TensorProductMap(I, R_R, 
            sigma, [i for i in 1:p+1, j in 1:1]); # right 
        TensorProductMap(I, R_L,  
            sigma, [i for i in p+1:-1:1, j in 1:1])] # left, downwards
    else # if a second 1D extrapolation is needed to get to the facet nodes
        rstf_1d, _ = quadrature(Line(),facet_quadrature_rule,p+1)
        R = [LinearMap(vandermonde(Line(), p, rstf_1d) / VDM_1) *
            TensorProductMap( R_B, I, sigma, [j for i in 1:1, j in 1:p+1]);
        LinearMap(vandermonde(Line(), p, rstf_1d) / VDM_2) * 
            TensorProductMap(I, R_R, sigma, [i for i in 1:p+1, j in 1:1]); 
        LinearMap(vandermonde(Line(), p, rstf_1d)[end:-1:1,:] / VDM_2) *   
            TensorProductMap(I, R_L, sigma, 
                [i for i in p+1:-1:1, j in 1:1])]
    end

    # triangular reference element data
    reference_element = RefElemData(Tri(), 
        mapping_degree, quad_rule_vol=quadrature(
            elem_type, volume_quadrature_rule, (p+1, p+1)),          quad_rule_face=quadrature(face_type(elem_type), 
                facet_quadrature_rule, p+1), 
        Nplot=N_plot)
    @unpack rstp, rstq, wq, wf = reference_element

    # discrete inner products
    W = LinearMap(Diagonal(wq))
    B = LinearMap(Diagonal(wf))

    # differentiation on the square
    D_η = (TensorProductMap(I, D_1, sigma, sigma), 
        TensorProductMap(D_2, I, sigma, sigma))

    # apply chain rule to differentiate on the triangle
    η1, η2, _ = quadrature(Quad(), volume_quadrature_rule, (p+1, p+1))
    D_ξ = (Diagonal((x-> 2.0/(1.0-x)).(η1))*D_η[1], 
            Diagonal((x -> (1+x)).(η1) ./ (x -> (1-x)).(η2))*D_η[1] + D_η[2])

    V_plot = LinearMap(vandermonde(Tri(), p, rstp...) / 
        vandermonde(Tri(), p, rstq...))
    V = LinearMap(I, N_q)
    P = LinearMap(I, N_q)
    ADVs = Tuple(-1.0*Diagonal(wq)*D_ξ[m] for m in 1:d)
    ADVw = Tuple(D_ξ[m]' * W for m in 1:d)

    return ReferenceApproximation{d}(approx_type, N_p, N_q, N_f, 
        reference_element, D_ξ, V, R, P, W, B, ADVs, ADVw, V_plot)
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