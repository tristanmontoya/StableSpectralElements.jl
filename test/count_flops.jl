import LinearAlgebra
using StableSpectralElements, OrdinaryDiffEq, .Threads
using Plots, Plots.PlotMeasures, TimerOutputs
using Profile, LinearAlgebra
using SparseArrays
using DelimitedFiles,Dates
using CountFlops, LinearMaps

const OPTYPE_MAP = Dict(
    "NodalTPSS" => NodalTPSS,
    "NodalTPSSOpt" => NodalTPSSOpt,
    "NodalTPSSLGL" => NodalTPSSLGL,
    "NodalTPSSMinimal" => NodalTPSSMinimal,
    "NodalTPSSOptimal" => NodalTPSSOptimal,
    "ModalTensor" => ModalTensor,
    "ModalOmega" => ModalMulti,
    "NodalTensor" => NodalTensor,
    "NodalOmega" => NodalMulti,
    "ModalDiagE" => ModalMultiDiagE,
    "NodalDiagE" => NodalMultiDiagE,
    "MFNodalTPSSLGL" => MatrixFreeTPSSLGL
)


function _skewsymmetric(opertype::String, form::String, p, d)
    if haskey(OPTYPE_MAP, opertype)
        operator = OPTYPE_MAP[opertype](p) 
    else
        error("Operator type '$opertype' not supported. Options are: $(keys(OPTYPE_MAP))")
    end    

    #define reference approximation
    if d == 2
        reference_approximation = ReferenceApproximation(operator, Tri(), mapping_degree=p)
    elseif d == 3
        reference_approximation = ReferenceApproximation(operator, Tet(), mapping_degree=p)
    end

    D = reference_approximation.D[1]
    Dᵀ = transpose(D)
    V = reference_approximation.V
    Vᵀ = transpose(V)
    R = reference_approximation.R
    Rᵀ = transpose(R)
    u_q = size(D,1)
    u_f = size(R,1)

    if form == "modal"
        if opertype == "ModalTensor"
            Vcnt = count_operations(V)
            VTcnt = count_operations(Vᵀ)
        else #dense flops = 2*n*m-n  for multiplying an n by m matrix by a vector
            Vcnt = 2*size(V,1)*size(V,2)-size(V,1) 
            VTcnt = 2*size(Vᵀ,1)*size(Vᵀ,2)-size(Vᵀ,1)
        end
    else # nodal schemes do not project
        Vcnt = 0
        VTcnt = 0
    end

    sparse_D=("NodalTPSSLGL", "NodalTPSSMinimal", "NodalTPSSOptimal", "ModalTensor", "NodalTensor")
    # sprase arrays for D if tensor product, otherwise dense 
    if opertype in sparse_D
        Dcnt = count_operations(sparse(D))
        DTcnt = count_operations(sparse(Dᵀ))
    else
        #Dcnt = count_operations(make_operator(D,GenericMatrixAlgorithm()))
        #DTcnt = count_operations(transpose(make_operator(D,GenericMatrixAlgorithm())))
        Dcnt = 2*size(D,1)*size(D,2)-size(D,1)
        DTcnt = 2*size(Dᵀ,1)*size(Dᵀ,2)-size(Dᵀ,1)
    end

    # assume R is sparse for diagE and TPSS and CC
    sparse_R=("NodalTPSSLGL", "NodalTPSSMinimal", "NodalTPSSOptimal", "ModalDiagE", "NodalDiagE", "NodalTensor", "ModalTensor")
    if opertype in sparse_R
        Rcnt = count_operations(sparse(R))
        RTcnt = count_operations(sparse(Rᵀ))
    else
        #Rcnt = count_operations(make_operator(R,GenericMatrixAlgorithm()))
        #RTcnt = count_operations(transpose(make_operator(R,GenericMatrixAlgorithm())))
        Rcnt = 2*size(R,1)*size(R,2)-size(R,1)
        RTcnt = 2*size(Rᵀ,1)*size(Rᵀ,2)-size(Rᵀ,1)
    end
    #println("FLOP count for ", opertype)
    #=     println(Dcnt, " flops for D which is size ", size(D))
    println(DTcnt, " flops for Dᵀ which is size ", size(Dᵀ))
    println(Vcnt, " flops for V which is size ", size(V))
    println(VTcnt, " flops for Vᵀ which is size ", size(Vᵀ))
    println(Rcnt, " flops for R which is size ", size(R))
    println(RTcnt, " flops for Rᵀ which is size ", size(Rᵀ)) =#

    return count_skew_symmetric(Dcnt, 
        DTcnt, 
        Vcnt,
        VTcnt,
        Rcnt,
        RTcnt, u_q, u_f, d, form)
end

function count_skew_symmetric(D,Dᵀ,V,Vᵀ,R,Rᵀ,u_q,u_f,d,form)
    # entires D to R^T are the flops to compute that matrix multiplied by a vector 

    # number of flops to perform a diagonal matrix times vector or an accumulation of vector
    diag = u_q

    # number of flops to perform diagonam matrix times vector of size # of facet nodes
    diag_facet = u_f

    # volume term
    volume = d * (d*(D+2*diag) + d*(Dᵀ+2*diag)) 

    #  surface terms (d+1) facets  
    # the 2 diag_facet accounts for the multiplicaiton with N and the addition/subtraction
    # then diag_facet is applying BJ, then multiply with R^T and then accum to residual which is size diag
    surface = (d*(R+2*diag_facet) + diag_facet + Rᵀ + diag) 

    # cost for weight-adjusted approx of inverse mass matrix
    if form == "modal"
        # assume quadrature strength of at least 2p, so the reference mass matrix M and M^-1 are identity
        mass_matrix = V + diag + Vᵀ
    else
        mass_matrix = diag
    end

    # cost for projection 
    if form == "modal"
        projection = Vᵀ
    else
        projection = 0
    end

    # cost to compute numerical and physical flux 
    # first term is physical flux (a*u) in each direction
    flux = d*u_q 

    # 0.5*(F(U+)+F(U-))*n (one mul, one add, and multipy by a over u_q nodes)
    # the 2*d is for the dot product with n
    flux = (2+d+2*d)*u_f

    #\lambda/2*|a*n|(u+ - u-), wavespeed computation is 2d (mul and add over d)
    # the three is to add u+ and u-, multiply by wave speed and subtract the entire term 
    flux = flux + (2d+1+3)*u_f

    # cost to get nodal values
    nodal_values = V + R
    
    total_flops = volume  + surface + mass_matrix + projection + flux + nodal_values
    
    volume_flops = volume/total_flops*100
    surface_flops = surface/total_flops*100
    mass_flops = (mass_matrix+projection)/total_flops*100
    flux_flops = flux/total_flops*100
    nodal_flops = nodal_values/total_flops*100

    println(volume_flops,", ", surface_flops,", ", mass_flops,", ", flux_flops,", ", nodal_flops)

    return total_flops
end 

function _fluxdifferencing(opertype::String, p, d, modal::String, diagE::String)
    if haskey(OPTYPE_MAP, opertype)
        operator = OPTYPE_MAP[opertype](p) 
    else
        error("Operator type '$opertype' not supported. Options are: $(keys(OPTYPE_MAP))")
    end    
    mach_number = 0.4
    angle = 0.0
    L = 1.0
    γ=1.4
    T = 0.01 # end time
    strength = sqrt(2/(γ-1)*(1-0.75^(γ-1))) # for central value of ρ=0.75
    conservation_law = EulerEquations{d}(γ)

    M = 1
    #define reference approximation
    if d == 2
        reference_approximation = ReferenceApproximation(operator, Tri(), mapping_degree=p)
        mesh = uniform_periodic_mesh(reference_approximation, ((0.0,L),(0.0,L)), (M,M))
        exact_solution = IsentropicVortex(conservation_law, θ=angle,
        Ma=mach_number, β=strength, R=1.0/10.0, x_0=(L/2,L/2))
    elseif d == 3
        reference_approximation = ReferenceApproximation(operator, Tet(), mapping_degree=p)
        mesh = uniform_periodic_mesh(reference_approximation, ((0.0,L),(0.0,L),(0.0,L)), (M,M,M))
        exact_solution = TaylorGreenVortex(conservation_law)
    end
    form = FluxDifferencingForm(inviscid_numerical_flux=LaxFriedrichsNumericalFlux(1.0))
    
    spatial_discretization = SpatialDiscretization(mesh, reference_approximation)
    ode = semidiscretize(conservation_law, spatial_discretization, 
    exact_solution, form, (0.0, T), ReferenceOperator(),DefaultOperatorAlgorithm(),parallelism=Serial());

    solver = ode.p
    S = solver.operators.S
    C = solver.operators.C
    V = reference_approximation.V
    Vᵀ = transpose(V)
    R = reference_approximation.R
    Rᵀ = transpose(R)    
    u_q = size(R,2)
    u_f = size(R,1)

    if modal == "modal"
        if opertype == "ModalTensor"
            Vcnt = count_operations(V)
            VTcnt = count_operations(Vᵀ)
        else #dense flops = 2*n*m-n  for multiplying an n by m matrix by a vector
            Vcnt = 2*size(V,1)*size(V,2)-size(V,1) 
            VTcnt = 2*size(Vᵀ,1)*size(Vᵀ,2)-size(Vᵀ,1)
        end
    else # nodal schemes do not project
        Vcnt = 0
        VTcnt = 0
    end

    # assume R is sparse for diagE and TPSS
    sparse_R=("NodalTPSSLGL", "NodalTPSSMinimal", "NodalTPSSOptimal", "ModalDiagE", "NodalDiagE")
    if opertype in sparse_R
        Rcnt = count_operations(sparse(R))
        RTcnt = count_operations(sparse(Rᵀ))
    else
        #Rcnt = count_operations(make_operator(R,GenericMatrixAlgorithm()))
        #RTcnt = count_operations(transpose(make_operator(R,GenericMatrixAlgorithm())))
        Rcnt = 2*size(R,1)*size(R,2)-size(R,1)
        RTcnt = 2*size(Rᵀ,1)*size(Rᵀ,2)-size(Rᵀ,1)
    end

    if d == 2
        return count_flux_differencing(S,C,u_q,u_f,Vcnt,VTcnt,Rcnt,RTcnt,d, form, diagE)/2 # divide by d to get flops per element
    elseif d == 3
        return count_flux_differencing(S,C,u_q,u_f,Vcnt,VTcnt,Rcnt,RTcnt,d, form, diagE)/6 # divide by d to get flops per element
    end
end  

function count_flux_differencing(S, C, u_q, u_f, V, VT, R, RT, d, nodal, diagE)
    # S = the m S matrices which should be size u_q by u_q
    # C = the C correction matrix
    # u_q = number of points in u
    N_c = d + 2
    two_point_flux_cost = stpff(d) # cost to compute the two point flux once

    # flops to compute numerical flux, the u_f is cost to scale the numerical flux
    numerical_flux = nff(d, u_q, two_point_flux_cost) + N_c*u_f

    if first(S) isa SparseMatrixCSC
        # flops to compute the volume terms
        # count the times the i < j condition is hit
        i_less_j = 0
        for m in 1:d
            row_index = rowvals(S[m])
            for j in 1:u_q
                for ii in nzrange(S[m],j)
                    i = row_index[ii]
                    if i < j
                        i_less_j += 1
                    end
                end
            end
        end

        # within each i < j, we have to compute the two point flux at the point i,J
        # then loop through N_c and d to perform the flux differencing
        #=     for e in 1:N_c
            Fm_ij = 0.0
            for n in 1:d
                Λ_ij = Λ_q[i, m, n] + Λ_q[j, m, n] < ----- one addition
                @muladd Fm_ij = Fm_ij + Λ_ij * F_ij[e, n]< ----- one multiplication and addition
            end
            diff_ij = Sm_ij * Fm_ij < ---- one multiplication
            r_q[i, e] -= diff_ij < ----- one subtraction
            r_q[j, e] += diff_ij < ---- one addition
        end    =#
        
        # we do the things in the brackets i_less_j times
        volume_flops = i_less_j * (two_point_flux_cost + N_c * (d*(3*d+2)+2))
    else
        N_q = u_q
        N_c = d + 2
        
        pair_count = N_q * (N_q - 1) ÷ 2

        volume_flops = pair_count * (two_point_flux_cost + N_c * (d*(3*d+2)+2))
    end
    # now the facet term, represents a multiplication of R^TBJ*f_num and accumulation to the residual
    facet_flops = N_c*(RT + u_f)

    # now the facet correction term
    if diagE == "diagE"
        facet_correction = 0
    else 
        if C isa SparseMatrixCSC
            row_index = rowvals(C)
            
            # count number of non zeros in C that we have to compute the two point flux
            ij_count = 0
            for j in 1:u_f
                for i in nzrange(C,j)
                    ij_count += 1
                end
            end
        #=         for e in 1:N_c
                F_dot_n_ij = 0.0
                for m in 1:d
                    nJ_ij = halfnJf[m, j] + halfnJq[m, f, i] -> addition
                    @muladd F_dot_n_ij = F_dot_n_ij + nJ_ij * F_ij[e, m] -> an addition and multiplication
                end
                diff_ij = C_ij * F_dot_n_ij
                r_q[i, e] -= diff_ij
                f_f[j, e] -= diff_ij -> 3 operations here
            end =#
            facet_correction = ij_count * (two_point_flux_cost + N_c * (3*d + 3))
        else
            N_q = u_q
            N_f = u_f
            N_c = d+2

            ij_count = N_q * N_f

            facet_correction = ij_count *(two_point_flux_cost + N_c * ((3*d) + 3))
        end
    end

    # entropy projection cost. nodal_values called once per residual evaluation, which calls entropy_projection once
    if diagE == "diagE" && nodal == "nodal" # nodal diagE
        #mul!(u_q, V, u)
        #mul!(u_f, R, u_q)
        entropy_projection = N_c*(V + R)
    
    elseif diagE != "diagE" && nodal == "nodal" # nodal non-diagE
    #=         mul!(u_q, V, u)
        @inbounds for i in axes(u, 1)
            conservative_to_entropy!(w_q[i, :], conservation_law, u_q[i, :])
        end
        mul!(w_f, R, w_q)
        @inbounds for i in axes(u_f, 1)
            entropy_to_conservative!(u_f[i, :], conservation_law, w_f[i, :])
        end =#
        
        entropy_projection = N_c*(V + R) + u_q*c2e(d) + u_f*e2c(d)
    else # modal schemes
    #=     # evaluate entropy variables in terms of nodal conservative variables
        mul!(u_q, V, u)
        @inbounds for i in axes(u_q, 1)
            conservative_to_entropy!(w_q[i, :], conservation_law, u_q[i, :])
        end =#
        entropy_projection = N_c*V + u_q*c2e(d)
        
        #= # project entropy variables and store modal coeffs in w
        lmul!(WJ, w_q)
        mul!(w, Vᵀ, w_q)
        mass_matrix_solve!(mass_solver, k, w, w_q) # w = M[k] \ w =#
        entropy_projection += N_c*((u_q + VT) + V + u_q + VT)
        
        #= # get nodal values of projected entropy variables
        mul!(w_q, V, w)
        mul!(w_f, R, w_q) =#
        entropy_projection += N_c*(V + R)

        #= # convert back to conservative variables
        @inbounds for i in axes(u_q, 1)
            entropy_to_conservative!(u_q[i, :], conservation_law, w_q[i, :])
        end
        @inbounds for i in axes(u_f, 1)
            entropy_to_conservative!(u_f[i, :], conservation_law, w_f[i, :])
        end =#
        entropy_projection += u_q*e2c(d) + u_f*e2c(d)
    end
    
    # modal projection and mass matrix
    if nodal == "nodal"
        mass_matrix = 0
        projection = 0
    else
        mass_matrix = N_c*(V + u_q + VT)
        projection = N_c*VT
    end
    total_flops = numerical_flux + volume_flops + facet_flops + facet_correction + entropy_projection + mass_matrix + projection
#=     println("Volume flops: ", volume_flops , " Percentage of FLOPs: ", volume_flops/total_FLOPs)
    println("Facet flops: ", facet_flops+facet_correction, " Percentage of FLOPs: ", (facet_flops+facet_correction)/total_FLOPs)
    println("Mass matrix flops: ", mass_matrix+projection, " Percentage of FLOPs: ", (mass_matrix+projection)/total_FLOPs)
    println("Numerical flux flops: ", numerical_flux, " Percentage of FLOPs: ", numerical_flux/total_FLOPs)
    println("Entropy projection/nodal valuesflops: ", entropy_projection, " Percentage of FLOPs: ", entropy_projection/total_FLOPs) =#
    volume_flops = volume_flops/total_flops*100
    surface_flops = (facet_flops+facet_correction)/total_flops*100
    mass_flops = (mass_matrix+projection)/total_flops*100
    flux_flops = numerical_flux/total_flops*100
    nodal_flops = entropy_projection/total_flops*100

    println(volume_flops,", ", surface_flops,", ", mass_flops,", ", flux_flops,", ", nodal_flops)
    return total_flops
end

function stpff(d)
    # the "sum" functions are counted, "logmean" functions are not, need to manually add back those counts
    conservation_law = EulerEquations{d}(1.4)
    u_L = ones(d+2)
    u_R = 2*u_L
    cnt = @count_ops compute_two_point_flux(conservation_law,EntropyConservativeFlux(),u_L,u_R)
    log = @count_ops logmean(1.0,2.0)
    invlog = @count_ops inv_logmean(1.0,2.0)
    cnt_total = cnt.fma64 + cnt.muladd64 + cnt.add64 + cnt.mul64 + cnt.div64 + cnt.sub64
    log_total = log.fma64 + log.muladd64 + log.add64 + log.mul64 + log.div64 + log.sub64
    invlog_total = invlog.fma64 + invlog.muladd64 + invlog.add64 + invlog.mul64 + invlog.div64 + invlog.sub64
    return cnt_total + log_total + invlog_total
end

# computes flops for entropy to conservative 
function e2c(d)
    cnt = @count_ops entropy_to_conservative!(-1*ones(Float64, d+2), EulerEquations{d}(1.4), -1*ones(Float64, d+2))
    # note that the count ops is able to count the sum and exp functions
    return cnt.fma64 + cnt.muladd64 + cnt.add64 + cnt.mul64 + cnt.div64 + cnt.sub64
end

function c2e(d)
    cnt = @count_ops conservative_to_entropy!(ones(Float64, d+2), EulerEquations{d}(1.4), [100;10*ones(Float64, d+1)])
    # note that the count ops is able to count the sum and log functions
    return cnt.fma64 + cnt.muladd64 + cnt.add64 + cnt.mul64 + cnt.div64 + cnt.sub64
end

function nff(d, u_q, tpff)
    #=     @inbounds for i in axes(u_in, 1)
        f_s = compute_two_point_flux(conservation_law,
            two_point_flux,
            u_in[i, :],
            u_out[i, :])
        a = numerical_flux.halfλ *
            wave_speed(conservation_law, u_in[i, :], u_out[i, :], n_f[:, i])
        for e in 1:N_c
            f_n_avg = 0.0
            for m in 1:d
                @muladd f_n_avg = f_n_avg + f_s[e, m] * n_f[m, i]
            end
            @muladd f_star[i, e] = f_n_avg + a * (u_in[i, e] - u_out[i, e])
        end
    end =#
    wsf = @count_ops wave_speed(EulerEquations{d}(1.4), [100;10*ones(Float64, d+2)], [100;10*ones(Float64, d+2)], ones(d))
    
    # the "1" is for multiplicaiton by the numerical flux parameter
    wsf_total = wsf.fma64 + wsf.muladd64 + wsf.add64 + wsf.mul64 + wsf.div64 + wsf.sub64 + wsf.sqrt64 + 1

    # d+2 = N_c
    return u_q *(tpff + wsf_total + (d+2)*(d*2+3))
end

function count_operations(M)
    x = ones(size(M,1))
    y = ones(size(M,2))
    cnt = @count_ops mul!(x,M,y)

    return cnt.add64 + cnt.mul64 + cnt.muladd64 + cnt.fma64 + cnt.div64 + cnt.sub64
end

function test_flops_driver(equation::String, opertype::String, form::String, diagE::String, p_vec, M_vec, d)
    flops = zeros(length(M_vec), length(p_vec))
    p_cnt = 1
    for p in p_vec
        if equation == "skewsymmetric"
            temp = _skewsymmetric(opertype, form, p, d)
        elseif equation == "fluxdifferencing"
            temp = _fluxdifferencing(opertype, p, d, form, diagE) # manually change this for now... 
        end
        M_cnt = 1
        for M in M_vec
            if d == 2
                flops[M_cnt,p_cnt] = 2*temp*M^2
            elseif d == 3
                flops[M_cnt,p_cnt] = 6*temp*M^3
            end
            M_cnt += 1
        end
        p_cnt += 1
    end
    open("ConvergenceStudies/DEC_1_Results/flops/curved1o32/"*string(equation)*string(d)*"D"*string(opertype)*"_M"*join(M_vec, "")*"_p"*join(p_vec, "")*".txt", "w") do io
        println(io, "Operator type: ", opertype)
        println(io, "p_vec: ", p_vec)
        println(io, "M_vec: ", M_vec)
        println(io, "FLOPS matrix (rows=M_vec, cols=p_vec):")
        writedlm(io, flops)
    end
end

function test_D_flops(opertype::String, p_vec,d)

    Dflop_vec = zeros(length(p_vec))
    pcnt = 1
    for p in p_vec
        if haskey(OPTYPE_MAP, opertype)
            operator = OPTYPE_MAP[opertype](p) 
        else
            error("Operator type '$opertype' not supported. Options are: $(keys(OPTYPE_MAP))")
        end    
        if d == 2
            reference_approximation = ReferenceApproximation(operator, Tri(), mapping_degree=p)
        elseif d == 3
            reference_approximation = ReferenceApproximation(operator, Tet(), mapping_degree=p)
        end
    D = reference_approximation.D[1]
    Dflop_vec[pcnt] = count_operations(sparse(D))
    pcnt += 1
    end
    return Dflop_vec
end

# time how long it takes to do 1 million multiplications with D
function timeDmatrix(opertype::String,p_vec,d)
    timevec = zeros(length(p_vec))
    pcnt = 1
    for p in p_vec
        if haskey(OPTYPE_MAP, opertype)
            operator = OPTYPE_MAP[opertype](p) 
        else
            error("Operator type '$opertype' not supported. Options are: $(keys(OPTYPE_MAP))")
        end    
        if d == 2
            reference_approximation = ReferenceApproximation(operator, Tri(), mapping_degree=p)
        elseif d == 3
            reference_approximation = ReferenceApproximation(operator, Tet(), mapping_degree=p)
        end
        D = reference_approximation.D[1]
        x = rand(size(D,2))
        y = rand(size(D,1))
        D = sparse(D)
        
        start_time = time()
        for i = 1:10^6
            mul!(y, D, x)
        end
        end_time = time()
        timevec[pcnt] = end_time - start_time
        pcnt +=1
    end
    return timevec
end

function test_V_flops(opertype::String, p_vec,d)

    Vflop_vec = zeros(length(p_vec))
    pcnt = 1
    for p in p_vec
        if haskey(OPTYPE_MAP, opertype)
            operator = OPTYPE_MAP[opertype](p) 
        else
            error("Operator type '$opertype' not supported. Options are: $(keys(OPTYPE_MAP))")
        end    
        if d == 2
            reference_approximation = ReferenceApproximation(operator, Tri(), mapping_degree=p)
        elseif d == 3
            reference_approximation = ReferenceApproximation(operator, Tet(), mapping_degree=p)
        end
    V = reference_approximation.V
    Vflop_vec[pcnt] = count_operations(sparse(V))
    pcnt += 1
    end
    return Vflop_vec
end

# time how long it takes to do 1 million multiplications with D
function timeVmatrix(opertype::String,p_vec,d)
    timevec = zeros(length(p_vec))
    pcnt = 1
    for p in p_vec
        if haskey(OPTYPE_MAP, opertype)
            operator = OPTYPE_MAP[opertype](p) 
        else
            error("Operator type '$opertype' not supported. Options are: $(keys(OPTYPE_MAP))")
        end    
        if d == 2
            reference_approximation = ReferenceApproximation(operator, Tri(), mapping_degree=p)
        elseif d == 3
            reference_approximation = ReferenceApproximation(operator, Tet(), mapping_degree=p)
        end
        V = reference_approximation.V
        x = rand(size(V,2))
        y = rand(size(V,1))
        V = sparse(V)
        
        start_time = time()
        for i = 1:10^6
            mul!(y, V, x)
        end
        end_time = time()
        timevec[pcnt] = end_time - start_time
        pcnt +=1
    end
    return timevec
end

function test_R_flops(opertype::String, p_vec,d)

    Rflop_vec = zeros(length(p_vec))
    pcnt = 1
    for p in p_vec
        if haskey(OPTYPE_MAP, opertype)
            operator = OPTYPE_MAP[opertype](p) 
        else
            error("Operator type '$opertype' not supported. Options are: $(keys(OPTYPE_MAP))")
        end    
        if d == 2
            reference_approximation = ReferenceApproximation(operator, Tri(), mapping_degree=p)
        elseif d == 3
            reference_approximation = ReferenceApproximation(operator, Tet(), mapping_degree=p)
        end
    R = reference_approximation.R
    Rflop_vec[pcnt] = count_operations(sparse(R))
    pcnt += 1
    end
    return Rflop_vec
end

# time how long it takes to do 1 million multiplications with D
function timeRmatrix(opertype::String,p_vec,d)
    timevec = zeros(length(p_vec))
    pcnt = 1
    for p in p_vec
        if haskey(OPTYPE_MAP, opertype)
            operator = OPTYPE_MAP[opertype](p) 
        else
            error("Operator type '$opertype' not supported. Options are: $(keys(OPTYPE_MAP))")
        end    
        if d == 2
            reference_approximation = ReferenceApproximation(operator, Tri(), mapping_degree=p)
        elseif d == 3
            reference_approximation = ReferenceApproximation(operator, Tet(), mapping_degree=p)
        end
        R = reference_approximation.R
        x = rand(size(R,2))
        y = rand(size(R,1))
        R = sparse(R)
        
        start_time = time()
        for i = 1:10^6
            mul!(y, R, x)
        end
        end_time = time()
        timevec[pcnt] = end_time - start_time
        pcnt +=1
    end
    return timevec
end
test_flops_driver("skewsymmetric","NodalTPSSLGL","nodal","Rerun",[1,2,3,4,5,6,7,8], [2,4,6,8,10,12],2)
#test_flops_driver("skewsymmetric","NodalTPSSOptimal","nodal","Rerun",[2,3,4,5,6],[1,2,3,4],2)
#test_flops_driver("skewsymmetric","NodalTPSSMinimal","nodal","Rerun",[2,3,4,5,6],[1,2,3,4],2)
#= test_flops_driver("skewsymmetric","NodalTPSSLGL","nodal","Rerun",[1,2,3,4,5,6,7,8],[6,8,10,12],2)
test_flops_driver("skewsymmetric","NodalTensor","nodal","Rerun",[1,2,3,4,5,6,7,8],[8,12,16,24],2)
test_flops_driver("skewsymmetric","ModalTensor","modal","Rerun",[1,2,3,4,5,6,7,8],[8,12,16,24],2)
test_flops_driver("skewsymmetric","NodalOmega","nodal","Rerun",[1,2,3,4,5,6,7,8],[8,12,16,24],2)
test_flops_driver("skewsymmetric","ModalOmega","modal","Rerun",[1,2,3,4,5,6,7,8],[8,12,16,24],2)
test_flops_driver("skewsymmetric","NodalDiagE","nodal","Rerun",[1,2,3,4,5,6,7,8],[8,12,16,24],2)
test_flops_driver("skewsymmetric","ModalDiagE","modal","Rerun",[1,2,3,4,5,6,7,8],[8,12,16,24],2)
test_flops_driver("skewsymmetric","NodalTPSSOptimal","nodal","Rerun",[2,3,4,5,6],[1,3,5,7,9],2)
test_flops_driver("skewsymmetric","NodalTPSSMinimal","nodal","Rerun",[2,3,4,5,6],[1,3,5,7,9],2)

test_flops_driver("skewsymmetric","NodalTPSSLGL","nodal","Rerun",[1,2,3,4,5],[1,2,3,4],3)
test_flops_driver("skewsymmetric","NodalTensor","nodal","Rerun",[1,2,3,4,5],[2,4,6,8],3)
test_flops_driver("skewsymmetric","ModalTensor","modal","Rerun",[1,2,3,4,5],[2,4,6,8],3)
test_flops_driver("skewsymmetric","NodalOmega","nodal","Rerun",[1,2,3,4,5],[2,4,6,8],3)
test_flops_driver("skewsymmetric","ModalOmega","modal","Rerun",[1,2,3,4,5],[2,4,6,8],3)
test_flops_driver("skewsymmetric","NodalDiagE","nodal","Rerun",[1,2,3,4,5],[2,4,6,8],3)
test_flops_driver("skewsymmetric","ModalDiagE","modal","Rerun",[1,2,3,4,5],[2,4,6,8],3)

test_flops_driver("fluxdifferencing","ModalTensor","modal","nonDiagE",[1,2,3,4],[8,12,16,24],2)
test_flops_driver("fluxdifferencing","NodalTensor","nodal","nonDiagE",[1,2,3,4],[8,12,16,24],2)
test_flops_driver("fluxdifferencing","ModalOmega","modal","nonDiagE",[1,2,3,4],[8,12,16,24],2)
test_flops_driver("fluxdifferencing","NodalOmega","nodal","nonDiagE",[1,2,3,4],[8,12,16,24],2)
test_flops_driver("fluxdifferencing","ModalDiagE","modal","diagE",[1,2,3,4],[8,12,16,24],2)
test_flops_driver("fluxdifferencing","NodalDiagE","nodal","diagE",[1,2,3,4],[8,12,16,24],2)
test_flops_driver("fluxdifferencing","ModalTensor","modal","nonDiagE",[5,6,7,8],[8,10,12,16],2)
test_flops_driver("fluxdifferencing","NodalTensor","nodal","nonDiagE",[5,6,7,8],[8,10,12,16],2)
test_flops_driver("fluxdifferencing","ModalOmega","modal","nonDiagE",[5,6,7,8],[8,10,12,16],2)
test_flops_driver("fluxdifferencing","NodalOmega","nodal","nonDiagE",[5,6,7,8],[8,10,12,16],2)
test_flops_driver("fluxdifferencing","ModalDiagE","modal","diagE",[5,6,7,8],[8,10,12,16],2)
test_flops_driver("fluxdifferencing","NodalDiagE","nodal","diagE",[5,6,7,8],[8,10,12,16],2)
test_flops_driver("fluxdifferencing","NodalTPSSLGL","nodal","diagE",[1,2,3,4],[4,6,8,12],2)
test_flops_driver("fluxdifferencing","NodalTPSSLGL","nodal","diagE",[5,6,7,8],[4,5,6,8],2)
test_flops_driver("fluxdifferencing","NodalTPSSOptimal","nodal","diagE",[2,3,4,5,6],[1,2,3,4],2)

test_flops_driver("fluxdifferencing","NodalTPSSLGL","nodal","diagE",[1,2,3,4,5],[1,2,3,4],3)
test_flops_driver("fluxdifferencing","NodalTensor","nodal","nondiagE",[1,2,3,4,5],[2,4,6,8],3)
test_flops_driver("fluxdifferencing","ModalTensor","modal","nondiagE",[1,2,3,4,5],[2,4,6,8],3)
test_flops_driver("fluxdifferencing","NodalOmega","nodal","nondiagE",[1,2,3,4,5],[2,4,6,8],3)
test_flops_driver("fluxdifferencing","ModalOmega","modal","nondiagE",[1,2,3,4,5],[2,4,6,8],3)
test_flops_driver("fluxdifferencing","NodalDiagE","nodal","diagE",[1,2,3,4,5],[2,4,6,8],3)
test_flops_driver("fluxdifferencing","ModalDiagE","modal","diagE",[1,2,3,4,5],[2,4,6,8],3) =#